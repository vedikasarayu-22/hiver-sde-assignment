import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.config import FINAL_TAXONOMY, RETRIEVAL_INDEX_PATH
from src.cache import DiskCache

class MajorityBaselineClassifier:
    def __init__(self, majority_class="software_update_bugs"):
        self.majority_class = majority_class

    def predict(self, text: str) -> str:
        return self.majority_class


class SimpleTfidfClassifier:
    def __init__(self, index_path=RETRIEVAL_INDEX_PATH):
        df = pd.read_csv(index_path)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = self.vectorizer.fit_transform(df['customer_text'].fillna(''))
        
        intent_col = 'primary_intent' if 'primary_intent' in df.columns else 'candidate_intent'
        y = df[intent_col].fillna('general_inquiry_other')
        
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X, y)

    def predict(self, text: str) -> str:
        if not text:
            return "general_inquiry_other"
        vec = self.vectorizer.transform([text])
        return self.model.predict(vec)[0]


class RuleLLMClassifier:
    def __init__(self):
        self.cache = DiskCache()
        self.patterns = {
            'device_hardware_battery': [r'battery', r'charge', r'charging', r'drain', r'overheat', r'screen', r'display', r'crack', r'camera', r'hardware'],
            'software_update_bugs': [r'update', r'ios', r'ios[0-9]+', r'software', r'bug', r'freeze', r'frozen', r'lag', r'crash', r'restart', r'stuck', r'apple logo'],
            'account_appleid_security': [r'apple id', r'password', r'lock', r'locked', r'2fa', r'verification', r'sign in', r'account', r'icloud', r'security', r'hacked'],
            'billing_subscriptions_refunds': [r'bill', r'purchase', r'refund', r'subscription', r'apple pay', r'app store charge', r'money', r'payment', r'receipt'],
            'connectivity_network_bluetooth': [r'wifi', r'wi-fi', r'bluetooth', r'cellular', r'signal', r'service', r'pairing', r'network'],
            'app_thirdparty_behavior': [r'whatsapp', r'twitter', r'facebook', r'instagram', r'spotify', r'youtube', r'app store', r'download app', r'keyboard', r'emoji'],
            'audio_accessory_hardware': [r'airpod', r'headphone', r'earbud', r'volume', r'sound', r'audio', r'macbook', r'watch', r'apple watch']
        }

    def predict(self, text: str) -> str:
        if not text:
            return "general_inquiry_other"
        text_lower = text.lower()
        
        matches = []
        for intent, regex_list in self.patterns.items():
            for pat in regex_list:
                if re.search(r'\b' + pat + r'\b', text_lower):
                    matches.append(intent)
                    break
                    
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            for priority in ['account_appleid_security', 'billing_subscriptions_refunds', 'device_hardware_battery', 'audio_accessory_hardware', 'connectivity_network_bluetooth', 'app_thirdparty_behavior']:
                if priority in matches:
                    return priority
            return matches[0]
            
        return "general_inquiry_other"
