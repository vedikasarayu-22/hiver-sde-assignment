import os
import sys
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SOURCE_PATH = os.path.join(DATA_DIR, "human_review.csv")
DEST_PATH = os.path.join(DATA_DIR, "human_verified_golden_set.csv")

INTENT_OPTIONS = [
    "device_hardware_battery",
    "software_update_bugs",
    "account_appleid_security",
    "billing_subscriptions_refunds",
    "connectivity_network_bluetooth",
    "app_thirdparty_behavior",
    "audio_accessory_hardware",
    "general_inquiry_other"
]

def load_data():
    if not os.path.exists(SOURCE_PATH):
        print(f"Error: Source file {SOURCE_PATH} not found.")
        sys.exit(1)

    source_df = pd.read_csv(SOURCE_PATH)

    if os.path.exists(DEST_PATH):
        verified_df = pd.read_csv(DEST_PATH)
        reviewed_ids = set(verified_df[verified_df['human_verified'] == True]['tweet_id'].astype(str))
    else:
        verified_df = pd.DataFrame(columns=[
            'tweet_id', 'customer_text', 'brand_response', 'human_final_intent',
            'human_final_escalation', 'difficulty', 'reviewer_notes', 'human_verified'
        ])
        reviewed_ids = set()

    return source_df, verified_df, reviewed_ids

def save_decision(verified_df, record):
    tweet_id_str = str(record['tweet_id'])
    verified_df['tweet_id'] = verified_df['tweet_id'].astype(str)
    
    if len(verified_df) > 0 and tweet_id_str in verified_df['tweet_id'].values:
        idx = verified_df[verified_df['tweet_id'] == tweet_id_str].index[0]
        for col, val in record.items():
            verified_df.at[idx, col] = val
    else:
        verified_df = pd.concat([verified_df, pd.DataFrame([record])], ignore_index=True)

    verified_df.to_csv(DEST_PATH, index=False, encoding='utf-8')
    return verified_df

def run_fast_cli():
    source_df, verified_df, reviewed_ids = load_data()
    total_count = len(source_df)

    print("=" * 80)
    print("FAST HUMAN GOLDEN SET REVIEW INTERFACE (CLI)")
    print("=" * 80)
    print("KEYBOARD CONTROLS:")
    print("  • Press Enter               -> Confirm & Save current suggested selection")
    print("  • 1 to 8                    -> Select Intent category (1:Battery, 2:Software, 3:Security...)")
    print("  • A / a                     -> Set Escalation to AUTO-HANDLE (False)")
    print("  • E / e                     -> Set Escalation to ESCALATE TO HUMAN (True)")
    print("  • Combine (e.g., '3e')      -> Set Intent 3 + Escalate and Save")
    print("  • N / n                     -> Add optional reviewer note")
    print("  • Q / q                     -> Pause & exit (progress is saved automatically)")
    print("=" * 80)

    unreviewed = source_df[~source_df['tweet_id'].astype(str).isin(reviewed_ids)]

    if len(unreviewed) == 0:
        print(f"\nAll {total_count} examples have been verified by human! Destination: {DEST_PATH}")
        return

    for _, row in unreviewed.iterrows():
        reviewed_count = len(reviewed_ids)
        pct = (reviewed_count / total_count) * 100

        # Preselect candidate suggestions
        current_intent = row['candidate_intent'] if row['candidate_intent'] in INTENT_OPTIONS else INTENT_OPTIONS[0]
        current_esc = True if str(row['candidate_escalation']).lower() == 'true' else False
        notes = ""

        while True:
            print("\n" + "=" * 80)
            print(f"PROGRESS: {reviewed_count} / {total_count} reviewed ({pct:.1f}%) | Resuming at Item #{reviewed_count + 1}")
            print("=" * 80)
            print(f"Tweet ID   : {row['tweet_id']} | Difficulty: {row['difficulty']}")
            print(f"\n[Customer Query]:\n{row['customer_text']}")
            print(f"\n[Reference Resolution]:\n{row['brand_response']}")
            print("\nCURRENT SELECTION (Preselected from candidate suggestion):")
            
            # Show intent menu with active highlight
            print("  Intents:")
            for i, opt in enumerate(INTENT_OPTIONS, 1):
                marker = "  ===> ACTIVE" if opt == current_intent else ""
                print(f"    [{i}] {opt}{marker}")
                
            esc_str = "ESCALATE TO HUMAN (True)" if current_esc else "AUTO-HANDLE (False)"
            print(f"  Escalation : ===> ACTIVE: {esc_str}")
            if notes:
                print(f"  Notes      : {notes}")
            print("-" * 80)

            cmd = input("\n[1-8: Intent | A: Auto-Handle | E: Escalate | Enter: SAVE & NEXT | Q: Quit] > ").strip()

            if cmd == "":
                # Save immediately and advance
                record = {
                    'tweet_id': str(row['tweet_id']),
                    'customer_text': str(row['customer_text']),
                    'brand_response': str(row['brand_response']),
                    'human_final_intent': current_intent,
                    'human_final_escalation': current_esc,
                    'difficulty': str(row['difficulty']),
                    'reviewer_notes': notes,
                    'human_verified': True
                }
                verified_df = save_decision(verified_df, record)
                reviewed_ids.add(str(row['tweet_id']))
                print(f"✅ Saved Tweet ID {row['tweet_id']} -> Intent: {current_intent} | Escalation: {current_esc}")
                break

            cmd_lower = cmd.lower()
            if cmd_lower == 'q':
                print(f"\nReview session paused. Progress saved ({len(reviewed_ids)} / {total_count} verified).")
                print("Run python review_cli.py anytime to resume where you left off!")
                sys.exit(0)

            # Check single/combined keys
            # Handle number keys 1-8
            for ch in cmd:
                if ch.isdigit():
                    num = int(ch)
                    if 1 <= num <= 8:
                        current_intent = INTENT_OPTIONS[num - 1]
                elif ch.lower() == 'a':
                    current_esc = False
                elif ch.lower() == 'e':
                    current_esc = True
                elif ch.lower() == 'n':
                    notes = input("Enter reviewer notes: ").strip()

            # If user entered quick combo (like '3e' or '2a'), auto-confirm if length > 1
            if len(cmd) > 1 and not cmd.startswith('n'):
                record = {
                    'tweet_id': str(row['tweet_id']),
                    'customer_text': str(row['customer_text']),
                    'brand_response': str(row['brand_response']),
                    'human_final_intent': current_intent,
                    'human_final_escalation': current_esc,
                    'difficulty': str(row['difficulty']),
                    'reviewer_notes': notes,
                    'human_verified': True
                }
                verified_df = save_decision(verified_df, record)
                reviewed_ids.add(str(row['tweet_id']))
                print(f"✅ Saved Tweet ID {row['tweet_id']} -> Intent: {current_intent} | Escalation: {current_esc}")
                break

if __name__ == "__main__":
    run_fast_cli()
