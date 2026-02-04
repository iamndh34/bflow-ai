"""
Optimized Utility functions for streaming responses.

Optimizations:
1. Larger buffer size (từ word-level sang phrase-level)
2. Reduced yield frequency
3. Better chunking for Vietnamese text
4. Handle both dict (Ollama format) and string formats

Đã optimize từ phiên bản trước để giảm overhead.
"""
import re
from typing import Iterator, Union


def stream_by_sentence(ollama_stream, buffer_size_words: int = 5):
    """
    Optimized streaming: yield theo cụm từ thay vì từng từ.

    Giảm số lần yield từ ~100 xuống ~10-20 cho cùng response.
    Tốc độ truyền nhanh hơn do ít overhead.

    Args:
        ollama_stream: Iterator từ ollama.chat(stream=True)
                      Hoặc generator yielding strings
        buffer_size_words: Số từ tối đa trong mỗi buffer (default: 5)

    Yields:
        str: Cụm từ hoàn chỉnh
    """
    buffer = ""
    word_count = 0

    # Pattern detect sentence ending: . ? ! followed by space/newline
    sentence_end_pattern = re.compile(r'([.!?]+\s+|\n\n+)')

    for chunk in ollama_stream:
        # Handle ChatResponse objects from ollama library
        if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
            content = chunk.message.content
        # Handle dict format (backward compatibility)
        elif isinstance(chunk, dict):
            content = chunk.get("message", {}).get("content", "")
        # Handle string format
        elif isinstance(chunk, str):
            content = chunk
        else:
            continue

        if not content:
            continue

        buffer += content

        # Đếm số từ trong buffer
        words_in_buffer = len([w for w in buffer.split() if w])

        # Yield khi:
        # 1. Đến cuối câu
        # 2. Hoặc buffer đã đủ lớn
        should_yield = False

        if sentence_end_pattern.search(buffer):
            should_yield = True
        elif words_in_buffer >= buffer_size_words:
            should_yield = True

        if should_yield and buffer.strip():
            yield buffer
            buffer = ""
            word_count = 0

    # Yield phần còn lại
    if buffer.strip():
        yield buffer


def stream_by_phrase(ollama_stream, phrases_per_yield: int = 2):
    """
    Yield theo phrase (cụm từ) thay vì từng từ.

    Args:
        ollama_stream: Iterator từ ollama.chat(stream=True)
        phrases_per_yield: Số cụm từ mỗi lần yield

    Yields:
        str: Cụm từ
    """
    buffer = ""
    phrase_count = 0

    # Pattern detect phrase boundary
    phrase_boundary = re.compile(r'[,.;!?]+\s+|\n')

    for chunk in ollama_stream:
        # Handle both dict format (Ollama) and string format
        if isinstance(chunk, dict):
            content = chunk.get("message", {}).get("content", "")
        elif isinstance(chunk, str):
            content = chunk
        else:
            continue

        if not content:
            continue

        buffer += content

        # Đếm số phrase delimiter trong buffer
        found_phrases = len(phrase_boundary.findall(buffer))

        if found_phrases >= phrases_per_yield:
            yield buffer
            buffer = ""
            phrase_count = 0

    if buffer.strip():
        yield buffer


def stream_by_char(ollama_stream):
    """
    Yield từng chữ một (true character-by-character streaming).

    Args:
        ollama_stream: Iterator từ ollama.chat(stream=True)

    Yields:
        str: Từng chữ
    """
    for chunk in ollama_stream:
        # Handle ChatResponse objects from ollama library
        if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
            content = chunk.message.content
        # Handle dict format (backward compatibility)
        elif isinstance(chunk, dict):
            content = chunk.get("message", {}).get("content", "")
        # Handle string format
        elif isinstance(chunk, str):
            content = chunk
        else:
            continue

        if content:
            # Yield each character individually
            for char in content:
                yield char


def stream_by_word(ollama_stream):
    """
    Yield từng từ một (fallback mode).

    Args:
        ollama_stream: Iterator từ ollama.chat(stream=True)

    Yields:
        str: Từng từ
    """
    for chunk in ollama_stream:
        # Handle ChatResponse objects from ollama library
        if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
            content = chunk.message.content
        # Handle dict format (backward compatibility)
        elif isinstance(chunk, dict):
            content = chunk.get("message", {}).get("content", "")
        # Handle string format
        elif isinstance(chunk, str):
            content = chunk
        else:
            continue

        if content:
            # Split by whitespace and yield each word
            words = content.split()
            for word in words:
                yield word + " "  # Add space after each word


def create_sentence_streamer(client, model, messages, system_prompt=None):
    """
    Wrapper để tạo sentence streamer từ ollama client.

    Args:
        client: ollama.Client instance
        model: Model name
        messages: List of messages hoặc single user message string
        system_prompt: Optional system prompt

    Yields:
        str: Từng câu hoàn chỉnh
    """
    if isinstance(messages, str):
        msg_list = [{"role": "user", "content": messages}]
    else:
        msg_list = messages

    if system_prompt:
        msg_list = [{"role": "system", "content": system_prompt}] + msg_list

    stream = client.chat(
        model=model,
        messages=msg_list,
        stream=True
    )

    yield from stream_by_sentence(stream)
