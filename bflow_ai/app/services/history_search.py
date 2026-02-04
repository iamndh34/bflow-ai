"""
Semantic History Cache - Cache thông minh truy xuất từ lịch sử

Hỗ trợ 3 modes:
1. Sentence-only: Similarity trên cả câu hoàn chỉnh
2. Keyword-only: Similarity trên từ khóa chính
3. Hybrid: Kết hợp cả hai (TỐT NHẤT)

Formula cho hybrid:
  final_score = α * sentence_score + (1-α) * keyword_score
"""
import time
from typing import Optional, List
import numpy as np

from app.core.config import settings
from app.core.embeddings import get_embed_model, encode_batch
from app.services.session_manager import get_session_manager
from app.services.similarity import (
    extract_keywords,
    compute_hybrid_similarity,
    HybridSemanticCache
)


class SemanticHistoryCache:
    """
    Cache thông minh dựa trên semantic search trong history.

    Hỗ trợ 3 modes: sentence, keyword, hybrid.
    """

    def __init__(self):
        """Initialize semantic history cache với config từ settings."""
        self._model = None
        self._mode = settings.SEMANTIC_MODE
        self._alpha = settings.SEMANTIC_ALPHA

    def _get_model(self):
        """Lazy load embedding model"""
        if self._model is None:
            self._model = get_embed_model()
        return self._model

    def _get_threshold(self) -> float:
        """Get threshold từ config"""
        return settings.SEMANTIC_SIMILARITY_THRESHOLD

    def _get_mode(self) -> str:
        """Get mode từ config"""
        return settings.SEMANTIC_MODE

    def find_similar_response(
        self,
        question: str,
        session_id: str,
        chat_type: str = "thinking",
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        Tìm câu hỏi tương tự trong history.

        Args:
            question: Câu hỏi hiện tại
            session_id: Session ID
            chat_type: Loại chat
            threshold: Similarity threshold (nếu None, dùng từ config)

        Returns:
            Response nếu tìm thấy, None nếu không
        """
        threshold = threshold or self._get_threshold()
        mode = self._get_mode()

        # Get history from session
        sm = get_session_manager(chat_type)
        history = sm.get_history(session_id, max_count=50)

        if not history:
            return None

        past_questions = [item["question"] for item in history]
        past_responses = {item["question"]: item["response"] for item in history}

        # Tìm similar question theo mode
        matched_idx, matched_question, score = self._find_by_mode(
            question, past_questions, mode, threshold
        )

        if matched_idx is not None:
            response = past_responses[matched_question]
            print(f"[SemanticHistory] ✓ Found ({mode}): \"{matched_question[:50]}...\" (score: {score:.3f})")
            return response

        return None

    def find_with_agent_hint(
        self,
        question: str,
        session_id: str,
        agent_name: str,
        chat_type: str = "thinking",
        threshold: Optional[float] = None
    ) -> Optional[str]:
        """
        Tìm câu hỏi tương tự CÙNG agent category.

        Args:
            question: Câu hỏi hiện tại
            session_id: Session ID
            agent_name: Tên agent
            chat_type: Loại chat
            threshold: Similarity threshold

        Returns:
            Response nếu tìm thấy, None nếu không
        """
        threshold = threshold or self._get_threshold()
        mode = self._get_mode()

        # Get history
        sm = get_session_manager(chat_type)
        history = sm.get_history(session_id, max_count=50)

        if not history:
            return None

        # Filter by agent name
        agent_history = [
            item for item in history
            if item.get("category") == agent_name
        ]

        if not agent_history:
            return None

        past_questions = [item["question"] for item in agent_history]
        past_responses = {item["question"]: item["response"] for item in agent_history}

        # Tìm similar question theo mode
        matched_idx, matched_question, score = self._find_by_mode(
            question, past_questions, mode, threshold
        )

        if matched_idx is not None:
            response = past_responses[matched_question]
            print(f"[SemanticHistory] ✓ Found in {agent_name} ({mode}): \"{matched_question[:50]}...\" (score: {score:.3f})")
            return response

        return None

    def _find_by_mode(
        self,
        query: str,
        history_questions: List[str],
        mode: str,
        threshold: float
    ) -> tuple:
        """
        Tìm câu hỏi tương tự theo mode.

        Returns:
            (index, matched_question, score) hoặc (None, None, 0.0)
        """
        if not history_questions:
            return None, None, 0.0

        model = self._get_model()

        if mode == "keyword":
            # Keyword-only mode
            return self._find_by_keyword(query, history_questions, threshold, model)
        elif mode == "hybrid":
            # Hybrid mode (sentence + keyword)
            return self._find_by_hybrid(query, history_questions, threshold, model)
        else:
            # Sentence-only mode (default)
            return self._find_by_sentence(query, history_questions, threshold, model)

    def _find_by_sentence(
        self,
        query: str,
        history_questions: List[str],
        threshold: float,
        model
    ) -> tuple:
        """Tìm bằng sentence similarity"""
        all_embeddings = encode_batch(history_questions + [query], normalize=True)

        past_embs = all_embeddings[:-1]
        query_emb = all_embeddings[-1]

        similarities = np.dot(past_embs, query_emb)

        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        print(f"[SemanticHistory-Sentence] Max similarity: {max_sim:.3f} (threshold: {threshold})")

        if max_sim >= threshold:
            return max_idx, history_questions[max_idx], max_sim

        return None, None, max_sim

    def _find_by_keyword(
        self,
        query: str,
        history_questions: List[str],
        threshold: float,
        model
    ) -> tuple:
        """Tìm bằng keyword similarity"""
        # Extract keywords
        query_keywords = extract_keywords(query)
        keyword_texts = [" ".join(extract_keywords(q)) for q in history_questions]

        if not query_keywords:
            # Fallback sang sentence
            print("[SemanticHistory-Keyword] No keywords found, using sentence")
            return self._find_by_sentence(query, history_questions, threshold, model)

        # Encode keywords
        all_kw_texts = keyword_texts + [" ".join(query_keywords)]
        kw_embs = encode_batch(all_kw_texts, normalize=True)

        query_kw_emb = kw_embs[-1]
        history_kw_embs = kw_embs[:-1]

        similarities = np.dot(history_kw_embs, query_kw_emb)

        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        print(f"[SemanticHistory-Keyword] Max similarity: {max_sim:.3f} (threshold: {threshold})")

        if max_sim >= threshold:
            return max_idx, history_questions[max_idx], max_sim

        return None, None, max_sim

    def _find_by_hybrid(
        self,
        query: str,
        history_questions: List[str],
        threshold: float,
        model
    ) -> tuple:
        """Tìm bằng hybrid similarity (sentence + keyword)"""
        results = compute_hybrid_similarity(
            query,
            history_questions,
            alpha=self._alpha,
            model=model
        )

        if not results:
            return None, None, 0.0

        best_idx, sent_sim, kw_sim, final_sim = results[0]

        print(f"[SemanticHistory-Hybrid] Sentence: {sent_sim:.3f}, Keyword: {kw_sim:.3f}, Final: {final_sim:.3f} (threshold: {threshold})")

        if final_sim >= threshold:
            return best_idx, history_questions[best_idx], final_sim

        return None, None, final_sim


# Global instance
_semantic_history_cache = None


def get_semantic_history_cache() -> SemanticHistoryCache:
    """Get singleton semantic history cache"""
    global _semantic_history_cache
    if _semantic_history_cache is None:
        _semantic_history_cache = SemanticHistoryCache()
    return _semantic_history_cache


# =============================================================================
# Convenience function để dùng trong orchestrator
# =============================================================================

def find_in_history_before_llm(
    question: str,
    session_id: str,
    agent_name: str,
    chat_type: str = "thinking"
) -> Optional[str]:
    """
    Tìm trong history TRƯỚC KHI gọi LLM.

    Args:
        question: Câu hỏi
        session_id: Session ID
        agent_name: Tên agent
        chat_type: Loại chat

    Returns:
        Response từ history nếu có similar question, None nếu không
    """
    cache = get_semantic_history_cache()

    # Tìm trong same agent
    response = cache.find_with_agent_hint(
        question=question,
        session_id=session_id,
        agent_name=agent_name,
        chat_type=chat_type
    )

    return response
