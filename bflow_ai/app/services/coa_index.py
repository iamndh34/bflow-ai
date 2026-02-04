"""
COA Index Service - Indexed data lookup cho Chart of Accounts

Thay vì linear search qua 2000+ accounts,
build index để lookup O(1).

Features:
1. O(1) lookup by code
2. Fast keyword search
3. Pre-built indexes
4. Lazy loading

Usage:
    from app.services.coa_index import get_coa_index

    idx = get_coa_index()
    acc = idx.get_by_code("156")
    results = idx.search_by_keyword("hàng hóa")
"""
import json
import os
from typing import Optional, List, Dict
from collections import defaultdict


class COAIndex:
    """Indexed COA data service"""

    def __init__(self):
        self._loaded = False
        self._data_99 = []
        self._data_200 = []
        self._compare_data = []

        # Indexes
        self._by_code_99 = {}
        self._by_code_200 = {}
        self._by_type_99 = defaultdict(list)
        self._by_type_200 = defaultdict(list)
        self._compare_by_type = defaultdict(list)

        # Keyword index - map keyword -> list of accounts
        self._keyword_index_99 = defaultdict(list)
        self._keyword_index_200 = defaultdict(list)

    def _load_data(self):
        """Lazy load data từ JSON files"""
        if self._loaded:
            return

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        COA_99_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_99.json")
        COA_200_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_200.json")
        COA_COMPARE_FILE = os.path.join(BASE_DIR, "services", "rag_json", "coa_compare_99_vs_200.json")

        # Load TT99
        if os.path.exists(COA_99_FILE):
            with open(COA_99_FILE, "r", encoding="utf-8") as f:
                self._data_99 = json.load(f)
        else:
            print(f"[WARN] {COA_99_FILE} not found")

        # Load TT200
        if os.path.exists(COA_200_FILE):
            with open(COA_200_FILE, "r", encoding="utf-8") as f:
                self._data_200 = json.load(f)
        else:
            print(f"[WARN] {COA_200_FILE} not found")

        # Load Compare
        if os.path.exists(COA_COMPARE_FILE):
            with open(COA_COMPARE_FILE, "r", encoding="utf-8") as f:
                self._compare_data = json.load(f)
        else:
            print(f"[WARN] {COA_COMPARE_FILE} not found")

        # Build indexes
        self._build_indexes()
        self._loaded = True
        print(f"[COAIndex] Loaded: {len(self._data_99)} TT99, {len(self._data_200)} TT200")

    def _build_indexes(self):
        """Build all indexes"""
        # Index by code cho TT99
        for acc in self._data_99:
            code = acc["code"]
            self._by_code_99[code] = acc
            self._by_type_99[acc["type_name"]].append(acc)

            # Keyword index - extract keywords from name
            self._index_keywords(acc, self._keyword_index_99)

        # Index by code cho TT200
        for acc in self._data_200:
            code = acc["code"]
            self._by_code_200[code] = acc
            self._by_type_200[acc["type_name"]].append(acc)
            self._index_keywords(acc, self._keyword_index_200)

        # Compare data index
        for item in self._compare_data:
            self._compare_by_type[item["change_type"]].append(item)

    def _index_keywords(self, acc: dict, index: dict):
        """Index keywords từ account name"""
        name_lower = acc["name"].lower()
        name_en_lower = acc.get("name_en", "").lower()

        # Extract keywords (split by space)
        for word in name_lower.split():
            if len(word) > 2:  # Skip quá ngắn
                index[word].append(acc)

        # English keywords
        for word in name_en_lower.split():
            if len(word) > 2:
                index[word].append(acc)

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def get_by_code(self, code: str, use_tt200: bool = False) -> Optional[dict]:
        """
        O(1) lookup by account code.

        Args:
            code: Account code (e.g., "156")
            use_tt200: If True, search TT200, otherwise TT99

        Returns:
            Account dict or None
        """
        self._load_data()
        index = self._by_code_200 if use_tt200 else self._by_code_99
        return index.get(code)

    def get_by_type(self, type_name: str, use_tt200: bool = False) -> List[dict]:
        """
        Get accounts by type name.

        Args:
            type_name: Type name (e.g., "Tài sản ngắn hạn")
            use_tt200: If True, search TT200

        Returns:
            List of accounts
        """
        self._load_data()
        index = self._by_type_200 if use_tt200 else self._by_type_99
        return index.get(type_name, [])

    def search_by_keyword(self, keyword: str, use_tt200: bool = False, limit: int = 20) -> List[dict]:
        """
        Fast keyword search using pre-built index.

        Args:
            keyword: Keyword to search
            use_tt200: If True, search TT200
            limit: Max results

        Returns:
            List of matching accounts
        """
        self._load_data()
        keyword_lower = keyword.lower()
        index = self._keyword_index_200 if use_tt200 else self._keyword_index_99

        # Direct keyword match from index
        if keyword_lower in index:
            results = index[keyword_lower]
            # Deduplicate by code
            seen = set()
            unique = []
            for acc in results:
                if acc["code"] not in seen:
                    seen.add(acc["code"])
                    unique.append(acc)
            return unique[:limit]

        # Fallback: substring search (slower but still indexed)
        return self._substring_search(keyword_lower, use_tt200, limit)

    def _substring_search(self, keyword_lower: str, use_tt200: bool, limit: int) -> List[dict]:
        """Fallback substring search với optimization"""
        data = self._data_200 if use_tt200 else self._data_99

        results = []
        for acc in data:
            if keyword_lower in acc["name"].lower() or keyword_lower in acc.get("name_en", "").lower():
                results.append(acc)
                if len(results) >= limit:
                    break
        return results

    def get_compare_by_type(self, change_type: str) -> List[dict]:
        """Get compare data by change type"""
        self._load_data()
        return self._compare_by_type.get(change_type, [])

    def get_all_compare_summary(self) -> dict:
        """Get summary of all changes"""
        self._load_data()
        return {
            "total_changes": len(self._compare_data),
            "by_type": {k: len(v) for k, v in self._compare_by_type.items()}
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_coa_index_instance: Optional[COAIndex] = None


def get_coa_index() -> COAIndex:
    """Get singleton COA index instance"""
    global _coa_index_instance
    if _coa_index_instance is None:
        _coa_index_instance = COAIndex()
    return _coa_index_instance
