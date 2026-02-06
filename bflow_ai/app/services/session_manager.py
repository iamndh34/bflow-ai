import json
import os
import uuid
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "services", "rag_json", "sessions")


class SessionManager:
    """Quản lý chat sessions - mỗi session là 1 file riêng."""

    def __init__(self, chat_type: str = "thinking"):
        self.chat_type = chat_type
        self.sessions_dir = os.path.join(SESSIONS_DIR, chat_type)
        self._ensure_dir()

    def _ensure_dir(self):
        """Tạo thư mục nếu chưa tồn tại."""
        os.makedirs(self.sessions_dir, exist_ok=True)

    def _get_session_path(self, session_id: str) -> str:
        """Lấy đường dẫn file của session."""
        return os.path.join(self.sessions_dir, f"{session_id}.json")

    def _load_session(self, session_id: str) -> dict:
        """Load session từ file."""
        path = self._get_session_path(session_id)
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[SessionManager] Error loading session {session_id}: {e}")
        return None

    def _save_session(self, session_id: str, data: dict):
        """Lưu session ra file."""
        path = self._get_session_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SessionManager] Error saving session {session_id}: {e}")

    def create_session(self, user_id: str = None) -> str:
        """Tạo session mới, trả về session_id."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        data = {
            "id": session_id,
            "user_id": user_id,
            "chat_type": self.chat_type,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "",
            "history": []
        }
        self._save_session(session_id, data)
        print(f"[SessionManager] Created session: {session_id} (user: {user_id})")
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Lấy thông tin session."""
        return self._load_session(session_id)

    def add_message(self, session_id: str, question: str, response: str, category: str = "GENERAL", user_id: str = None):
        """Thêm cặp Q&A vào session."""
        data = self._load_session(session_id)
        if not data:
            # Tự tạo session nếu chưa có
            session_id = self.create_session(user_id=user_id)
            data = self._load_session(session_id)

        data["history"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "response": response,
            "category": category
        })
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Cập nhật user_id nếu có
        if user_id:
            data["user_id"] = user_id

        # Cập nhật title theo câu hỏi gần nhất
        data["title"] = question[:50] + "..." if len(question) > 50 else question

        self._save_session(session_id, data)
        return session_id

    def get_history(self, session_id: str, max_count: int = 10) -> list:
        """Lấy history của session (N câu gần nhất)."""
        data = self._load_session(session_id)
        if not data:
            return []
        history = data.get("history", [])
        return history[-max_count:] if len(history) > max_count else history

    def get_messages_format(self, session_id: str, max_count: int = 10) -> list:
        """Chuyển history thành format messages cho Ollama."""
        history = self.get_history(session_id, max_count)
        messages = []
        for item in history:
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item["response"]})
        return messages

    def delete_session(self, session_id: str) -> bool:
        """Xóa session."""
        path = self._get_session_path(session_id)
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[SessionManager] Deleted session: {session_id}")
                return True
        except Exception as e:
            print(f"[SessionManager] Error deleting session {session_id}: {e}")
        return False

    def list_sessions(self, user_id: str = None) -> list:
        """Liệt kê tất cả sessions, sắp xếp theo updated_at mới nhất.

        Args:
            user_id: Nếu có, chỉ trả về sessions của user đó
        """
        sessions = []
        try:
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith(".json"):
                    session_id = filename[:-5]  # Bỏ .json
                    data = self._load_session(session_id)
                    if data:
                        # Lọc theo user_id nếu có
                        if user_id and data.get("user_id") != user_id:
                            continue

                        sessions.append({
                            "id": data.get("id", session_id),
                            "user_id": data.get("user_id", ""),
                            "title": data.get("title", "Untitled"),
                            "chat_type": data.get("chat_type", self.chat_type),
                            "created_at": data.get("created_at", ""),
                            "updated_at": data.get("updated_at", ""),
                            "message_count": len(data.get("history", []))
                        })
        except Exception as e:
            print(f"[SessionManager] Error listing sessions: {e}")

        # Sắp xếp theo updated_at mới nhất
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    def clear_session(self, session_id: str):
        """Xóa history của session nhưng giữ session."""
        data = self._load_session(session_id)
        if data:
            data["history"] = []
            data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_session(session_id, data)


# Singleton instances
THINKING_SESSION_MANAGER = SessionManager(chat_type="thinking")
FREE_SESSION_MANAGER = SessionManager(chat_type="free")


def get_session_manager(chat_type: str = "thinking") -> SessionManager:
    """Lấy SessionManager theo chat_type."""
    if chat_type == "free":
        return FREE_SESSION_MANAGER
    return THINKING_SESSION_MANAGER
