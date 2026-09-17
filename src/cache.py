import os
import json
import pickle
import hashlib
from src.config import LLM_CACHE_PATH, EMBEDDING_CACHE_PATH

class DiskCache:
    def __init__(self, json_path=LLM_CACHE_PATH, pkl_path=EMBEDDING_CACHE_PATH):
        self.json_path = json_path
        self.pkl_path = pkl_path
        self.llm_cache = self._load_json()
        self.embedding_cache = self._load_pkl()

    def _load_json(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_json(self):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.llm_cache, f, indent=2, ensure_ascii=False)

    def _load_pkl(self):
        if os.path.exists(self.pkl_path):
            try:
                with open(self.pkl_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_pkl(self):
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)

    def get_llm_response(self, prompt: str):
        key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        return self.llm_cache.get(key)

    def set_llm_response(self, prompt: str, response: str):
        key = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        self.llm_cache[key] = response
        self._save_json()

    def get_embedding(self, text: str):
        key = hashlib.md5(text.encode('utf-8')).hexdigest()
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, embedding):
        key = hashlib.md5(text.encode('utf-8')).hexdigest()
        self.embedding_cache[key] = embedding
        self._save_pkl()
