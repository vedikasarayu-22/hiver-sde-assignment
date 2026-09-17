import os
import sys
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CANDIDATE_PATH = os.path.join(DATA_DIR, "candidate_golden_set.csv")

def print_taxonomy_guide():
    print("=" * 80)
    print("OFFICIAL INTENT ANNOTATION TAXONOMY & ESCALATION POLICY REFERENCE GUIDE")
    print("=" * 80)
    
    df = pd.read_csv(CANDIDATE_PATH)

    categories = [
        {
            "name": "1. device_hardware_battery",
            "definition": "Issues related to physical hardware, battery degradation, charging cables/ports, screen damage, camera, speaker, mic, overheating, or power failures.",
            "keywords": ["battery", "charge", "charging", "drain", "power off", "overheat", "screen", "display", "crack", "camera", "speaker", "mic", "hardware"],
            "intent_key": "device_hardware_battery"
        },
        {
            "name": "2. software_update_bugs",
            "definition": "Issues related to iOS/macOS/watchOS system updates, software glitches, app freezes, lag, system crashes, restart loops, stuck on Apple logo, storage backup/restore errors.",
            "keywords": ["update", "ios", "ios11", "software", "bug", "freeze", "frozen", "lag", "crash", "restart", "stuck", "apple logo", "restore", "backup"],
            "intent_key": "software_update_bugs"
        },
        {
            "name": "3. account_appleid_security",
            "definition": "Issues regarding locked Apple ID accounts, password resets, 2-Factor Authentication (2FA), account recovery delays, unauthorized sign-ins, phishing alerts, or security locks.",
            "keywords": ["apple id", "password", "lock", "locked", "2fa", "verification", "sign in", "account", "icloud", "security", "hacked", "disabled"],
            "intent_key": "account_appleid_security"
        },
        {
            "name": "4. billing_subscriptions_refunds",
            "definition": "Monetary transactions, App Store charges, iTunes gift card code delivery, iCloud subscription cancellation, refund requests, or payment card verification.",
            "keywords": ["bill", "purchase", "refund", "subscription", "apple pay", "app store charge", "money", "payment", "receipt", "cancel", "itunes card"],
            "intent_key": "billing_subscriptions_refunds"
        },
        {
            "name": "5. connectivity_network_bluetooth",
            "definition": "Problems connecting to Wi-Fi networks, Bluetooth accessory pairing failures, cellular data/signal dropouts, or AirDrop transfer errors.",
            "keywords": ["wifi", "wi-fi", "bluetooth", "cellular", "signal", "service", "connect", "pairing", "network", "airdrop"],
            "intent_key": "connectivity_network_bluetooth"
        },
        {
            "name": "6. app_thirdparty_behavior",
            "definition": "Specific third-party app issues (YouTube, Spotify, WhatsApp, Twitter, Instagram), App Store app download/install failures, keyboard typing bugs, or emoji rendering bugs.",
            "keywords": ["whatsapp", "twitter", "facebook", "instagram", "spotify", "youtube", "app store", "download app", "install", "app crash", "keyboard", "emoji"],
            "intent_key": "app_thirdparty_behavior"
        },
        {
            "name": "7. audio_accessory_hardware",
            "definition": "Problems with AirPods, headphones, earbuds, volume controls, ringer buttons, Apple Watch accessories, dongles, or physical adapters.",
            "keywords": ["airpod", "headphone", "earbud", "earphone", "volume", "sound", "audio", "macbook", "ipad", "watch", "apple watch", "dongle", "adapter"],
            "intent_key": "audio_accessory_hardware"
        },
        {
            "name": "8. general_inquiry_other",
            "definition": "Generic support line availability, store opening hours, vague complaints, feature requests, or customer tweets without sufficient technical context.",
            "keywords": ["store hours", "lines closed", "help", "hello", "lines open", "support hours"],
            "intent_key": "general_inquiry_other"
        }
    ]

    for cat in categories:
        print("\n" + "=" * 80)
        print(f"CATEGORY: {cat['name']}")
        print("=" * 80)
        print(f"• DEFINITION  : {cat['definition']}")
        print(f"• KEYWORDS    : {', '.join(cat['keywords'])}")
        print("• REAL EXAMPLES FROM DATASET:")

        sample_rows = df[df['candidate_intent'] == cat['intent_key']].head(3)
        if len(sample_rows) == 0:
            sample_rows = df.head(3)

        for idx, r in sample_rows.iterrows():
            clean_txt = str(r['customer_text']).replace('\n', ' ')
            clean_resp = str(r['brand_response']).replace('\n', ' ')
            print(f"\n  [Ex {idx+1}] Query: {clean_txt[:120]}...")
            print(f"         Resolution: {clean_resp[:100]}...")

    print("\n" + "=" * 80)
    print("HUMAN REVIEW ESCALATION POLICY & RULES")
    print("=" * 80)
    print("Mark HUMAN_FINAL_ESCALATION = True (ESCALATE TO HUMAN) if ANY of the following apply:")
    print("  1. Financial / Monetary Dispute: Refund requests, unauthorized App Store charges, payment card issues, unreceived iTunes gift card codes.")
    print("  2. Account Security & Compromise: Locked Apple ID, forgotten password, 2FA verification failure, account recovery delays, phishing/hacked accounts.")
    print("  3. Physical Hardware Damage: Shattered screens, swollen batteries, water damage, broken hardware requiring physical Genius Bar inspection.")
    print("  4. Explicit Private Information Requirement: Customer inquiry requires exchanging PII, serial numbers, order IDs, or private account lookup via DM.")
    print("\nMark HUMAN_FINAL_ESCALATION = False (AUTO-HANDLE) if:")
    print("  • The query is a standard technical troubleshooting question (e.g. Wi-Fi reset steps, iOS update check, app restart) that can be auto-handled with standard public guidance/links without private account inspection.")
    print("=" * 80)

if __name__ == "__main__":
    print_taxonomy_guide()
