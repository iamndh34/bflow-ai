import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HistoryManager:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history_file = os.path.join(BASE_DIR, "services", "rag_json", "conversation_history.json")
        self.history = self._load_from_file()

    def _load_from_file(self) -> list:
        """Load tất cả lịch sử từ file JSON"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"[HistoryManager] Error loading history: {e}")
        return []

    def _save_to_file(self):
        """Lưu lịch sử ra file JSON"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HistoryManager] Error saving history: {e}")

    def add(self, question: str, response: str, category: str = "POSTING_ENGINE"):
        """Thêm câu hỏi và câu trả lời vào lịch sử (lưu tất cả, không giới hạn)"""
        self.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "response": response,
            "category": category
        })
        self._save_to_file()

    def get_recent(self, count: int = None) -> list:
        """Lấy N câu hỏi gần nhất (mặc định là max_history)"""
        n = count if count is not None else self.max_history
        return self.history[-n:] if len(self.history) > n else self.history

    def get_last_category(self) -> str:
        """Lấy category của câu hỏi trước"""
        return self.history[-1]["category"] if self.history else None

    def reload(self):
        """Reload lịch sử từ file"""
        self.history = self._load_from_file()
        print(f"[HistoryManager] Reloaded {len(self.history)} items")

    def clear(self, count: int = None):
        """Xóa N câu hỏi gần nhất (mặc định là max_history). Giữ lại phần còn lại."""
        n = count if count is not None else self.max_history
        if len(self.history) <= n:
            self.history = []
        else:
            self.history = self.history[:-n]
        self._save_to_file()
        print(f"[HistoryManager] Cleared {n} recent items, {len(self.history)} remaining")


# Singleton instance
HISTORY_MANAGER = HistoryManager()
