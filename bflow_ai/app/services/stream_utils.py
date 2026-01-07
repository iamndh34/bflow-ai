"""
Utility functions for streaming responses.
Yield theo từng từ để tránh lỗi encoding tiếng Việt.
"""

import re


def stream_by_sentence(ollama_stream):
    """
    Buffer ollama stream và yield theo từng từ hoàn chỉnh.

    Cắt khi gặp:
    - Space (khoảng trắng)
    - Newline (\n)

    Args:
        ollama_stream: Iterator từ ollama.chat(stream=True)

    Yields:
        str: Từng từ hoàn chỉnh (kèm space/newline phía sau)
    """
    buffer = ""

    # Pattern để detect điểm cắt từ: space hoặc newline
    word_end_pattern = re.compile(r'([ \t]+|\n)')

    for chunk in ollama_stream:
        content = chunk.get("message", {}).get("content", "")
        if not content:
            continue

        buffer += content

        # Tìm và yield các từ hoàn chỉnh trong buffer
        while True:
            match = word_end_pattern.search(buffer)
            if match:
                # Lấy vị trí kết thúc từ (bao gồm cả space/newline)
                end_pos = match.end()

                # Yield từ hoàn chỉnh
                word = buffer[:end_pos]
                if word:
                    yield word

                # Giữ lại phần còn lại trong buffer
                buffer = buffer[end_pos:]
            else:
                # Không tìm thấy điểm cắt, giữ nguyên buffer
                break

    # Yield phần còn lại trong buffer (nếu có)
    if buffer:
        yield buffer


def create_sentence_streamer(client, model, messages, system_prompt=None):
    """
    Wrapper để tạo sentence streamer từ ollama client.

    Args:
        client: ollama.Client instance
        model: Model name (e.g., "qwen2.5:3b")
        messages: List of messages hoặc single user message string
        system_prompt: Optional system prompt

    Yields:
        str: Từng câu hoàn chỉnh
    """
    # Build messages list
    if isinstance(messages, str):
        msg_list = [{"role": "user", "content": messages}]
    else:
        msg_list = messages

    # Add system prompt if provided
    if system_prompt:
        msg_list = [{"role": "system", "content": system_prompt}] + msg_list

    # Create stream
    stream = client.chat(
        model=model,
        messages=msg_list,
        stream=True
    )

    # Yield by sentence
    yield from stream_by_sentence(stream)
