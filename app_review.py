import os
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="Fast Human Golden Set Review", page_icon="⚡", layout="wide")

@st.cache_data(ttl=1)
def load_source_data():
    if not os.path.exists(SOURCE_PATH):
        st.error(f"Source file not found at: {SOURCE_PATH}")
        st.stop()
    return pd.read_csv(SOURCE_PATH)

def load_dest_data():
    if os.path.exists(DEST_PATH):
        df = pd.read_csv(DEST_PATH)
        df['tweet_id'] = df['tweet_id'].astype(str)
        return df
    return pd.DataFrame(columns=[
        'tweet_id', 'customer_text', 'brand_response', 'human_final_intent',
        'human_final_escalation', 'difficulty', 'reviewer_notes', 'human_verified'
    ])

def save_record(record):
    df = load_dest_data()
    t_id = str(record['tweet_id'])
    
    if len(df) > 0 and t_id in df['tweet_id'].values:
        idx = df[df['tweet_id'] == t_id].index[0]
        for k, v in record.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        
    df.to_csv(DEST_PATH, index=False, encoding='utf-8')
    return df

source_df = load_source_data()
dest_df = load_dest_data()

total_count = len(source_df)
reviewed_ids = set(dest_df[dest_df['human_verified'] == True]['tweet_id'].astype(str)) if len(dest_df) > 0 else set()
reviewed_count = len(reviewed_ids)

st.title("⚡ Fast Human Golden Set Review Interface")
st.markdown("**Keyboard Shortcuts / Fast Controls**: 1-8 for Intent | A for Auto-Handle | E for Escalate | Press **Enter / Space** to Save & Next")

progress_pct = reviewed_count / total_count if total_count > 0 else 0
st.progress(progress_pct)
st.subheader(f"Progress: {reviewed_count} / {total_count} reviewed ({progress_pct*100:.1f}%)")

if 'current_idx' not in st.session_state:
    # Auto-resume to first unreviewed item
    for idx, r in source_df.iterrows():
        if str(r['tweet_id']) not in reviewed_ids:
            st.session_state.current_idx = idx
            break
    else:
        st.session_state.current_idx = 0

col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])
with col_nav1:
    if st.button("⬅️ Previous") and st.session_state.current_idx > 0:
        st.session_state.current_idx -= 1
        st.rerun()

with col_nav3:
    if st.button("Next ➡️") and st.session_state.current_idx < total_count - 1:
        st.session_state.current_idx += 1
        st.rerun()

row = source_df.iloc[st.session_state.current_idx]
tweet_id_str = str(row['tweet_id'])

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown(f"### Example {st.session_state.current_idx + 1} of {total_count}")
    st.markdown(f"**Tweet ID**: `{tweet_id_str}` | **Difficulty**: `{row['difficulty']}`")
    
    st.info(f"**Customer Query**:\n\n{row['customer_text']}")
    st.success(f"**Reference Brand Resolution**:\n\n{row['brand_response']}")
    
    cand_esc_str = "ESCALATE (True)" if str(row['candidate_escalation']).lower() == 'true' else "AUTO-HANDLE (False)"
    st.warning(f"**Preselected Suggestions**:\n\n• **Suggested Intent**: `{row['candidate_intent']}`\n\n• **Suggested Escalation**: `{cand_esc_str}`")

with col_right:
    st.markdown("### Human Final Verification")
    
    existing_record = dest_df[dest_df['tweet_id'] == tweet_id_str] if len(dest_df) > 0 else pd.DataFrame()
    
    if len(existing_record) > 0:
        default_intent = existing_record.iloc[0]['human_final_intent']
        default_esc = "ESCALATE TO HUMAN (True)" if bool(existing_record.iloc[0]['human_final_escalation']) else "AUTO-HANDLE (False)"
        default_notes = str(existing_record.iloc[0].get('reviewer_notes', ''))
        if default_notes == 'nan': default_notes = ""
    else:
        default_intent = row['candidate_intent'] if row['candidate_intent'] in INTENT_OPTIONS else INTENT_OPTIONS[0]
        default_esc = "ESCALATE TO HUMAN (True)" if str(row['candidate_escalation']).lower() == 'true' else "AUTO-HANDLE (False)"
        default_notes = ""

    intent_idx = INTENT_OPTIONS.index(default_intent) if default_intent in INTENT_OPTIONS else 0
    
    # Form for instant submission via Enter key
    with st.form(key="fast_review_form"):
        final_intent = st.radio("Final Intent Category (Keys 1-8):", INTENT_OPTIONS, index=intent_idx)
        
        esc_opts = ["AUTO-HANDLE (False) [Key: A]", "ESCALATE TO HUMAN (True) [Key: E]"]
        esc_idx = 1 if "ESCALATE" in default_esc else 0
        final_esc_str = st.radio("Escalation Decision:", esc_opts, index=esc_idx)
        final_escalation = True if "ESCALATE" in final_esc_str else False

        reviewer_notes = st.text_input("Reviewer Notes (Optional):", value=default_notes)

        submit = st.form_submit_button("✅ SAVE & NEXT (Press Enter)", type="primary")

    if submit:
        record = {
            'tweet_id': tweet_id_str,
            'customer_text': str(row['customer_text']),
            'brand_response': str(row['brand_response']),
            'human_final_intent': final_intent,
            'human_final_escalation': final_escalation,
            'difficulty': str(row['difficulty']),
            'reviewer_notes': reviewer_notes,
            'human_verified': True
        }
        save_record(record)
        st.toast(f"Saved decision for Tweet ID {tweet_id_str}!")
        if st.session_state.current_idx < total_count - 1:
            st.session_state.current_idx += 1
        st.rerun()
