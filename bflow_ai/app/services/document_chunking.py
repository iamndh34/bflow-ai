"""
Document Chunking for RAG - Semantic Chunking

Chunk văn bản dài thành các phần nhỏ dựa trên ý nghĩa ngữ nghĩa.
Phù hợp cho RAG với các tài liệu kế toán, quy trình, etc.

Methods:
1. SemanticChunking: Chia dựa trên cosine similarity giữa sentences
2. RecursiveChunking: Chia đệ quy theo paragraph → sentence
3. FixedSizeChunking: Chia theo số tokens/characters (fallback)
"""

import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from app.core.embeddings import get_embed_model, encode_batch


@dataclass
class Chunk:
    """Represent một chunk văn bản"""
    content: str
    metadata: Dict[str, Any]
    index: int

    def __repr__(self):
        return f"Chunk(index={self.index}, len={len(self.content)})"


class SemanticChunker:
    """
    Semantic Chunking - Chia văn bản dựa trên ý nghĩa ngữ nghĩa.

    Cách hoạt động:
    1. Chia văn bản thành sentences
    2. Tính embeddings cho các sentences
    3. Gom sentences có similarity cao vào cùng chunk
    4. Bắt đầu chunk mới khi similarity thấp (ngắt ý)

    Args:
        similarity_threshold: Ngưỡng similarity để gom sentence (0-1)
                           - Cao hơn: Chunk lớn hơn, ít chunk hơn
                           - Thấp hơn: Chunk nhỏ hơn, nhiều chunk hơn
        max_chunk_size: Số characters tối đa mỗi chunk (hard limit)
        min_chunk_size: Số characters tối thiểu mỗi chunk
        embedding_model: Model embedding (default: singleton từ embeddings.py)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        embedding_model=None
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self._embed_model = embedding_model

    def _get_model(self):
        """Lazy load embedding model"""
        if self._embed_model is None:
            self._embed_model = get_embed_model()
        return self._embed_model

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chia văn bản thành các chunks dựa trên semantic similarity.

        Args:
            text: Văn bản đầu vào (có thể dài, nhiều段落)
            metadata: Metadata bổ sung (source, type, etc.)

        Returns:
            List of Chunk objects
        """
        # 1. Split thành sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return [Chunk(content=text, metadata=metadata or {}, index=0)]

        # 2. Tính embeddings cho tất cả sentences
        embeddings = encode_batch(sentences, normalize=True)

        # 3. Gom sentences thành chunks dựa trên similarity
        chunks = self._group_by_similarity(sentences, embeddings, metadata)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Chia văn bản thành sentences.

        Hỗ trợ tiếng Việt:
        - Kết thúc bằng . ! ? followed by space/newline
        - Kết thúc bằng dấu chấm và xuống dòng
        """
        # Pattern cho sentence ending
        sentence_endings = re.compile(r'([.!?]+)\s+|([.!?]+)\n|\n\n+')

        # Split nhưng giữ nguyên delimiter
        parts = []
        last_idx = 0

        for match in sentence_endings.finditer(text):
            end_idx = match.end()
            sentence = text[last_idx:end_idx].strip()

            if sentence:
                parts.append(sentence)

            last_idx = end_idx

        # Phần còn lại
        if last_idx < len(text):
            remaining = text[last_idx:].strip()
            if remaining:
                parts.append(remaining)

        return parts if parts else [text]

    def _group_by_similarity(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        base_metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Gom sentences thành chunks dựa trên cosine similarity.

        Logic:
        - Bắt đầu chunk với sentence đầu tiên
        - Thêm sentence tiếp theo nếu similarity >= threshold
        - Nếu similarity < threshold, bắt đầu chunk mới
        - Nếu chunk đạt max_chunk_size, bắt đầu chunk mới
        """
        chunks = []
        current_chunk_sentences = []
        current_length = 0

        for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            sentence_len = len(sentence)

            # Kiểm tra nếu thêm câu này sẽ vượt quá max size
            if current_chunk_sentences and current_length + sentence_len > self.max_chunk_size:
                # Finalize chunk hiện tại
                chunks.append(self._create_chunk(
                    current_chunk_sentences,
                    chunks,
                    base_metadata
                ))
                current_chunk_sentences = []
                current_length = 0

            # Nếu chưa có sentence nào, thêm câu đầu tiên
            if not current_chunk_sentences:
                current_chunk_sentences.append(sentence)
                current_length = sentence_len
                continue

            # Tính similarity với sentence cuối cùng trong chunk
            last_embedding = embeddings[idx - 1]
            similarity = float(np.dot(last_embedding, embedding))

            # Quyết định: thêm vào chunk hay tạo chunk mới?
            if similarity >= self.similarity_threshold:
                # Similar cao, cùng ý nghĩa → thêm vào chunk
                current_chunk_sentences.append(sentence)
                current_length += sentence_len
            else:
                # Similar thấp, ngắt ý → kết thúc chunk hiện tại
                if len(current_chunk_sentences) > 0:
                    chunks.append(self._create_chunk(
                        current_chunk_sentences,
                        chunks,
                        base_metadata
                    ))

                # Bắt đầu chunk mới với sentence hiện tại
                current_chunk_sentences = [sentence]
                current_length = sentence_len

        # Thêm chunk cuối cùng
        if current_chunk_sentences:
            chunks.append(self._create_chunk(
                current_chunk_sentences,
                chunks,
                base_metadata
            ))

        # Filter chunks quá ngắn
        filtered_chunks = [
            c for c in chunks
            if len(c.content) >= self.min_chunk_size
        ]

        return filtered_chunks if filtered_chunks else chunks

    def _create_chunk(
        self,
        sentences: List[str],
        existing_chunks: List[Chunk],
        base_metadata: Optional[Dict] = None
    ) -> Chunk:
        """Tạo Chunk object từ list of sentences"""
        content = " ".join(sentences).strip()

        # Combine base metadata với chunk-specific metadata
        metadata = (base_metadata or {}).copy()
        metadata.update({
            "chunk_index": len(existing_chunks),
            "sentence_count": len(sentences),
            "char_count": len(content),
            "chunking_method": "semantic"
        })

        return Chunk(
            content=content,
            metadata=metadata,
            index=len(existing_chunks)
        )


class RecursiveChunker:
    """
    Recursive Character Chunking - Chia đệ quy theo hierarchy.

    Hierarchy: Paragraph → Sentence → Word

    Args:
        max_chunk_size: Số characters tối đa mỗi chunk
        chunk_overlap: Số characters overlap giữa các chunks
        separators: List of separators theo thứ tự ưu tiên
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators cho tiếng Việt: paragraph → sentence → word
        self.separators = separators or ["\n\n+", "\n", ". ", "! ", "? ", " ", ""]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chia văn bản sử dụng recursive chunking.

        Args:
            text: Văn bản đầu vào
            metadata: Metadata bổ sung

        Returns:
            List of Chunk objects
        """
        chunks = []

        def _split_recursive(text: str, separators: List[str], depth: int = 0):
            """Recursive split theo separators"""
            if depth >= len(separators):
                return [text]

            separator = separators[depth]

            if separator:
                parts = text.split(separator)
            else:
                # Fallback: split theo character
                parts = list(text)

            # Gom các parts thành chunks
            current_chunk = ""
            chunks_split = []

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Nếu thêm part này vượt quá max size
                if len(current_chunk) + len(part) > self.max_chunk_size:
                    # Có chunk hiện tại, lưu lại
                    if current_chunk:
                        chunks_split.append(current_chunk.strip())

                    # Nếu part đơn cũng vượt quá max size, chia nhỏ hơn
                    if len(part) > self.max_chunk_size:
                        sub_chunks = _split_recursive(part, separators, depth + 1)
                        chunks_split.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part
                else:
                    # Thêm vào chunk hiện tại (với separator nếu cần)
                    if current_chunk and separator and separator not in ["\n", "\n\n"]:
                        current_chunk += separator + part
                    elif separator in ["\n", "\n\n"]:
                        current_chunk += separator + part
                    else:
                        current_chunk += part

            # Thêm chunk cuối cùng
            if current_chunk:
                chunks_split.append(current_chunk.strip())

            # Add overlap
            if len(chunks_split) > 1 and self.chunk_overlap > 0:
                chunks_split = self._add_overlap(chunks_split)

            return chunks_split

        # Thực hiện split
        split_texts = _split_recursive(text, self.separators)

        # Tạo Chunk objects
        for idx, chunk_text in enumerate(split_texts):
            if len(chunk_text) >= self.min_chunk_size if hasattr(self, 'min_chunk_size') else True:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    "chunking_method": "recursive"
                })
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    index=idx
                ))

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap giữa các chunks liên tiếp"""
        overlapped = []

        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                # Lấy overlap từ chunk trước
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]
                overlapped.append(overlap_text + " " + chunk)
            else:
                overlapped.append(chunk)

        return overlapped


class FixedSizeChunker:
    """
    Fixed Size Chunking - Chia theo số characters cố định.

    Args:
        chunk_size: Số characters mỗi chunk
        chunk_overlap: Số characters overlap giữa các chunks
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Chia văn bản thành các chunks có kích thước cố định.

        Args:
            text: Văn bản đầu vào
            metadata: Metadata bổ sung

        Returns:
            List of Chunk objects
        """
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    "chunking_method": "fixed_size"
                })
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    index=idx
                ))
                idx += 1

            start = end - self.chunk_overlap

        return chunks


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_chunker(
    method: str = "semantic",
    **kwargs
) -> SemanticChunker | RecursiveChunker | FixedSizeChunker:
    """
    Factory function để tạo chunker.

    Args:
        method: Chọn phương pháp chunking
                - "semantic": SemanticChunking (default, best cho RAG)
                - "recursive": RecursiveChunker (balance)
                - "fixed": FixedSizeChunker (simple, fast)
        **kwargs: Additional arguments cho chunker cụ thể

    Returns:
        Chunker instance

    Examples:
        >>> chunker = get_chunker("semantic", similarity_threshold=0.75)
        >>> chunks = chunker.chunk(long_text)

        >>> chunker = get_chunker("recursive", max_chunk_size=1500)
        >>> chunks = chunker.chunk(long_text)
    """
    if method == "semantic":
        return SemanticChunker(**kwargs)
    elif method == "recursive":
        return RecursiveChunker(**kwargs)
    elif method == "fixed":
        return FixedSizeChunker(**kwargs)
    else:
        raise ValueError(f"Unknown chunking method: {method}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def chunk_text(
    text: str,
    method: str = "semantic",
    metadata: Optional[Dict] = None,
    **kwargs
) -> List[Chunk]:
    """
    Convenience function để chunk text.

    Args:
        text: Văn bản đầu vào
        method: Phương pháp chunking ("semantic", "recursive", "fixed")
        metadata: Metadata bổ sung
        **kwargs: Arguments cho chunker

    Returns:
        List of Chunk objects
    """
    chunker = get_chunker(method, **kwargs)
    return chunker.chunk(text, metadata)


def chunk_documents(
    documents: List[Dict[str, str]],
    method: str = "semantic",
    text_field: str = "content",
    **kwargs
) -> List[Chunk]:
    """
    Chunk nhiều documents cùng lúc.

    Args:
        documents: List of dicts với ít nhất một text field
        method: Phương pháp chunking
        text_field: Field name chứa text content
        **kwargs: Arguments cho chunker

    Returns:
        List of Chunk objects với metadata từ documents

    Examples:
        >>> docs = [
        ...     {"content": "Text 1...", "source": "file1.pdf"},
        ...     {"content": "Text 2...", "source": "file2.pdf"}
        ... ]
        >>> chunks = chunk_documents(docs, method="semantic")
    """
    all_chunks = []

    for doc_idx, doc in enumerate(documents):
        text = doc.get(text_field, "")
        if not text:
            continue

        # Merge metadata
        metadata = {k: v for k, v in doc.items() if k != text_field}
        metadata["document_index"] = doc_idx

        chunks = chunk_text(text, method=method, metadata=metadata, **kwargs)
        all_chunks.extend(chunks)

    return all_chunks
