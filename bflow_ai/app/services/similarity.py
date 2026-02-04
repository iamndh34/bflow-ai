"""
Hybrid Similarity - Kết hợp Full Sentence và Keywords

Ý tưởng:
- Similarity trên cả câu hoàn chỉnh + từ khóa chính
- Kết hợp 2 scores để ra quyết định cuối cùng

Formula:
  final_score = α * sentence_score + (1-α) * keyword_score

  Với α = 0.7 (ưu tiên sentence meaning, nhưng vẫn xem xét keywords)
"""
import re
import numpy as np
from typing import List, Tuple, Optional, Dict

from app.core.config import settings
from app.core.embeddings import get_embed_model, encode_batch


def extract_keywords(text: str) -> List[str]:
    """
    Trích xuất từ khóa chính từ câu hỏi.

    Keywords bao gồm:
    - Số tài khoản (3-5 chữ số)
    - Từ quan trọng (hàng hóa, tiền mặt, hạch toán, etc.)

    Args:
        text: Câu hỏi

    Returns:
        List of keywords
    """
    keywords = set()

    # 1. Số tài khoản (3-5 chữ số)
    account_numbers = re.findall(r'\b\d{3,5}\b', text)
    keywords.update(account_numbers)

    # 2. Từ khóa quan trọng trong kế toán
    important_terms = [
        'hàng hóa', 'tiền mặt', 'phải thu', 'phải trả',
        'hạch toán', 'định khoản', 'bút toán', 'ghi nhận',
        'doanh thu', 'chi phí', 'lợi nhuận', 'nguyên vật liệu',
        'tài sản', 'nợ phải trả', 'vốn chủ sở hữu',
        'thuế', 'gtgt', 'tncn', 'tdcn', 'khấu hao',
        'nhập kho', 'xuất kho', 'bán hàng', 'mua hàng'
    ]

    text_lower = text.lower()
    for term in important_terms:
        if term in text_lower:
            keywords.add(term)

    # 3. Các từ viết tắt
    abbreviations = re.findall(r'\b[A-Z]{2,5}\b', text)
    keywords.update(abbreviations)

    return list(keywords)


def compute_hybrid_similarity(
    query: str,
    history_questions: List[str],
    alpha: float = 0.7,
    model=None
) -> List[Tuple[int, float, float, float]]:
    """
    Tính hybrid similarity kết hợp sentence và keywords.

    Args:
        query: Câu hỏi hiện tại
        history_questions: List các câu hỏi trong history
        alpha: Trọng số cho sentence similarity (0.7 = 70% sentence, 30% keywords)
        model: Embedding model

    Returns:
        List of (index, sentence_sim, keyword_sim, final_score) tuples
    """
    if model is None:
        model = get_embed_model()

    # 1. Sentence similarity (full text)
    all_texts = history_questions + [query]
    sentence_embs = encode_batch(all_texts, normalize=True)

    query_sent_emb = sentence_embs[-1]
    history_sent_embs = sentence_embs[:-1]

    sentence_sims = np.dot(history_sent_embs, query_sent_emb)

    # 2. Keyword similarity
    query_keywords = extract_keywords(query)
    keyword_texts = [" ".join(extract_keywords(q)) for q in history_questions]

    if query_keywords and any(keyword_texts):
        # Encode keywords
        all_kw_texts = keyword_texts + [" ".join(query_keywords)]
        kw_embs = encode_batch(all_kw_texts, normalize=True)

        query_kw_emb = kw_embs[-1]
        history_kw_embs = kw_embs[:-1]

        keyword_sims = np.dot(history_kw_embs, query_kw_emb)
    else:
        # Không có keywords → dùng sentence similarity
        keyword_sims = sentence_sims.copy()

    # 3. Combine scores
    results = []
    for i, (sent_sim, kw_sim) in enumerate(zip(sentence_sims, keyword_sims)):
        final_sim = alpha * sent_sim + (1 - alpha) * kw_sim
        results.append((i, float(sent_sim), float(kw_sim), float(final_sim)))

    # Sort by final score
    results.sort(key=lambda x: -x[3])

    return results


class HybridSemanticCache:
    """
    Semantic Cache với Hybrid Similarity.

    Kết hợp full sentence + keywords để match tốt hơn.
    """

    def __init__(self, alpha: float = 0.7):
        """
        Initialize hybrid cache.

        Args:
            alpha: Trọng số sentence similarity (default: 0.7)
                   - 1.0 = Chỉ dùng sentence (original)
                   - 0.7 = 70% sentence, 30% keywords (balanced)
                   - 0.5 = 50-50 balance
                   - 0.3 = 30% sentence, 70% keywords (keyword-focused)
        """
        self.alpha = alpha
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = get_embed_model()
        return self._model

    def find_hybrid(
        self,
        query: str,
        history_questions: List[str],
        threshold: float = 0.85
    ) -> Tuple[Optional[int], Optional[str], dict]:
        """
        Tìm câu hỏi tương tự bằng hybrid similarity.

        Args:
            query: Câu hỏi hiện tại
            history_questions: List câu hỏi trong history
            threshold: Ngưỡng final score

        Returns:
            (index, matched_question, details) hoặc (None, None, {})
        """
        if not history_questions:
            return None, None, {}

        results = compute_hybrid_similarity(
            query,
            history_questions,
            alpha=self.alpha,
            model=self._get_model()
        )

        if not results:
            return None, None, {}

        best_idx, sent_sim, kw_sim, final_sim = results[0]

        details = {
            "sentence_similarity": sent_sim,
            "keyword_similarity": kw_sim,
            "final_score": final_sim,
            "threshold": threshold,
            "alpha": self.alpha,
            "matched": final_sim >= threshold
        }

        if final_sim >= threshold:
            return best_idx, history_questions[best_idx], details

        return None, None, details


# Convenience function
def find_with_hybrid_similarity(
    query: str,
    history: list,
    threshold: float = None,
    alpha: float = 0.7
) -> Optional[str]:
    """
    Tìm response bằng hybrid similarity.

    Args:
        query: Câu hỏi
        history: List of {question, response} dicts
        threshold: Ngưỡng similarity (None → dùng từ config)
        alpha: Trọng số sentence similarity

    Returns:
        Response nếu tìm thấy, None nếu không
    """
    threshold = threshold or settings.SEMANTIC_SIMILARITY_THRESHOLD

    if not history:
        return None

    history_questions = [item["question"] for item in history]
    history_responses = {item["question"]: item["response"] for item in history}

    cache = HybridSemanticCache(alpha=alpha)
    idx, matched, details = cache.find_hybrid(query, history_questions, threshold)

    if matched is not None:
        print(f"[HybridSimilarity] Sentence: {details['sentence_similarity']:.3f}, "
              f"Keyword: {details['keyword_similarity']:.3f}, "
              f"Final: {details['final_score']:.3f}")
        print(f"[HybridSimilarity] ✓ Matched: \"{matched[:50]}...\"")
        return history_responses[matched]

    print(f"[HybridSimilarity] No match (best: {details.get('final_score', 0):.3f} < {threshold})")
    return None
