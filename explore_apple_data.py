import os
import sys
import re
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import APPLE_RAW_TWEETS_PATH, RESOLUTION_PAIRS_PATH

def explore_data():
    if not os.path.exists(APPLE_RAW_TWEETS_PATH):
        print(f"Error: {APPLE_RAW_TWEETS_PATH} not found.")
        return

    df = pd.read_csv(APPLE_RAW_TWEETS_PATH)
    print("=" * 70)
    print("1. EMPIRICAL DATASET OVERVIEW & RECONSTRUCTION (@AppleSupport)")
    print("=" * 70)
    print(f"Total @AppleSupport related tweets: {len(df):,}")
    
    inbound_df = df[df['inbound'] == True].copy()
    print(f"Inbound customer tweets: {len(inbound_df):,}")
    
    outbound_df = df[df['inbound'] == False].copy()
    print(f"Outbound @AppleSupport tweets: {len(outbound_df):,}")
    
    initial_queries = inbound_df[inbound_df['in_response_to_tweet_id'].isna()].copy()
    print(f"Initial customer query tweets (Root conversations): {len(initial_queries):,}")
    
    outbound_map = {}
    for idx, row in outbound_df.iterrows():
        parent_id = row['in_response_to_tweet_id']
        if pd.notna(parent_id):
            parent_id = str(int(parent_id)) if isinstance(parent_id, float) else str(parent_id)
            if parent_id not in outbound_map:
                outbound_map[parent_id] = row['text']
                
    initial_queries['tweet_id_str'] = initial_queries['tweet_id'].astype(str)
    initial_queries['brand_response'] = initial_queries['tweet_id_str'].map(outbound_map)
    
    paired_df = initial_queries[initial_queries['brand_response'].notna()].copy()
    print(f"Initial queries with matched @AppleSupport resolution reply: {len(paired_df):,}")
    
    print("\n" + "=" * 70)
    print("2. EMPIRICAL TAXONOMY VERIFICATION & CATEGORY DISCOVERY")
    print("=" * 70)
    
    intent_patterns = {
        'device_hardware_battery': [
            r'battery', r'charge', r'charging', r'drain', r'power off', r'overheat', 
            r'screen', r'display', r'crack', r'camera', r'speaker', r'mic', r'hardware'
        ],
        'software_update_bugs': [
            r'update', r'ios', r'ios[0-9]+', r'software', r'bug', r'freeze', r'frozen', 
            r'lag', r'crash', r'restart', r'stuck', r'apple logo', r'restore', r'backup'
        ],
        'account_appleid_security': [
            r'apple id', r'password', r'lock', r'locked', r'2fa', r'verification', 
            r'sign in', r'account', r'icloud', r'security', r'hacked', r'disabled'
        ],
        'billing_subscriptions_refunds': [
            r'bill', r'charge', r'purchase', r'refund', r'subscription', r'apple pay', 
            r'app store', r'card', r'money', r'payment', r'receipt', r'cancel'
        ],
        'connectivity_network_bluetooth': [
            r'wifi', r'wi-fi', r'bluetooth', r'cellular', r'signal', r'service', 
            r'connect', r'pairing', r'network', r'airdrop'
        ],
        'app_thirdparty_behavior': [
            r'whatsapp', r'twitter', r'facebook', r'instagram', r'spotify', r'youtube',
            r'app store', r'download', r'install', r'app crash', r'keyboard', r'emoji', r'notification'
        ],
        'audio_accessory_hardware': [
            r'airpod', r'headphone', r'earbud', r'earphone', r'volume', r'sound', r'audio', 
            r'macbook', r'ipad', r'watch', r'apple watch', r'dongle', r'adapter'
        ]
    }
    
    def match_intents(text):
        if not isinstance(text, str):
            return ['general_inquiry_other']
        text_lower = text.lower()
        matched = []
        for intent, patterns in intent_patterns.items():
            for pat in patterns:
                if re.search(r'\b' + pat + r'\b', text_lower):
                    matched.append(intent)
                    break
        if not matched:
            return ['general_inquiry_other']
        return matched

    paired_df['intents'] = paired_df['text'].apply(match_intents)
    paired_df['intent_count'] = paired_df['intents'].apply(len)
    paired_df['primary_intent'] = paired_df['intents'].apply(lambda x: x[0])
    
    print("\nPrimary Intent Distribution (After Taxonomy Refinement):")
    intent_counts = paired_df['primary_intent'].value_counts()
    for intent, count in intent_counts.items():
        pct = (count / len(paired_df)) * 100
        print(f"  • {intent:35s}: {count:5d} ({pct:5.1f}%)")
        
    print(f"\nCategory Overlap Breakdown:")
    multi_matches = (paired_df['intent_count'] > 1).sum()
    print(f"  • Total queries matching >1 intent: {multi_matches:,} ({(multi_matches/len(paired_df))*100:.1f}%)")

    paired_clean = paired_df[['tweet_id_str', 'author_id', 'created_at', 'text', 'brand_response', 'primary_intent']].copy()
    paired_clean.rename(columns={'tweet_id_str': 'tweet_id', 'text': 'customer_text'}, inplace=True)
    
    paired_clean.to_csv(RESOLUTION_PAIRS_PATH, index=False, encoding='utf-8')
    print(f"\nSaved {len(paired_clean):,} clean paired conversations to {RESOLUTION_PAIRS_PATH}")

if __name__ == "__main__":
    explore_data()
