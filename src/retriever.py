import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import RETRIEVAL_INDEX_PATH
from src.cache import DiskCache

_SHARED_RETRIEVAL_INDEX = None

class HistoricalResolutionRetriever:
    def __init__(self, index_path=RETRIEVAL_INDEX_PATH):
        global _SHARED_RETRIEVAL_INDEX
        if _SHARED_RETRIEVAL_INDEX is None:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Retrieval index file not found at: {index_path}")
            
            df = pd.read_csv(index_path)
            texts = df['customer_text'].fillna('').tolist()
            vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            _SHARED_RETRIEVAL_INDEX = {
                'df': df,
                'texts': texts,
                'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix
            }
            
        self.df = _SHARED_RETRIEVAL_INDEX['df']
        self.vectorizer = _SHARED_RETRIEVAL_INDEX['vectorizer']
        self.tfidf_matrix = _SHARED_RETRIEVAL_INDEX['tfidf_matrix']
        self.cache = DiskCache()

    def retrieve(self, query_text: str, top_k: int = 3):
        if not query_text or not isinstance(query_text, str):
            return []
            
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            intent_val = row.get('candidate_intent', row.get('primary_intent', 'general_inquiry_other'))
            results.append({
                'tweet_id': str(row['tweet_id']),
                'customer_text': str(row['customer_text']),
                'brand_response': str(row['brand_response']),
                'intent': str(intent_val),
                'similarity_score': float(sims[idx])
            })
        return results
