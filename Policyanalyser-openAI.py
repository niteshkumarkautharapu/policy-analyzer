import streamlit as st
import pdfplumber
from openai import OpenAI
import json
import tempfile
import os

# ---------------------------
# CONFIG
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-5-mini"


# ---------------------------
# Extract text from PDF
# ---------------------------
def extract_text(file):
    text = ""

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    return text


# ---------------------------
# Clean Output
# ---------------------------
def clean_json(text):
    return text.replace("```json", "").replace("```", "").strip()


# ---------------------------
# Validate JSON
# ---------------------------
def validate_json(json_str):
    try:
        json_str = clean_json(json_str)
        return json.loads(json_str)
    except:
        return None


# ---------------------------
# OpenAI Call
# ---------------------------
def call_gpt(prompt):

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )

    return response.output_text


# ---------------------------
# EXTRACTION PROMPT
# ---------------------------
def run_extraction(text):
    prompt = f"""
You are an insurance policy data extraction engine.

You are given raw insurance policy text.

Your task is to extract structured data in strict JSON format.

CORE RULES:

Output ONLY valid JSON
Do NOT explain anything
Do NOT add commentary
Do NOT infer beyond what is explicitly mentioned
If a field is missing → return "Not specified"
Use exact numeric values where possible (no ranges unless given)
Standardize currency in INR (numbers only, no symbols)
Keep structure EXACTLY as defined below

OUTPUT JSON STRUCTURE:

{{
"policy_name": "",
"insurer": "",
"policy_number": "",
"policy_type": "",
"policy_classification_hint": "",

"sum_insured": "",
"sum_insured_type": "",
"deductible": "",
"deductible_type": "",
"copay": "",
"room_rent_limit": "",

"premium": "",
"policy_start_date": "",
"policy_end_date": "",
"policy_tenure": "",

"members_count": "",
"members": [
{{
"name": "",
"age": "",
"relation": ""
}}
],

"coverage": {{
"inpatient": "",
"pre_hospitalization_days": "",
"post_hospitalization_days": "",
"day_care": "",
"ayush": "",
"domiciliary": "",
"ambulance_limit": "",
"donor_expense": ""
}},

"exclusions": {{
"maternity": "",
"opd": "",
"permanent_exclusions": []
}},

"waiting_periods": {{
"initial_days": "",
"specified_disease_months": "",
"pre_existing_months": ""
}},

"sublimits": {{
"cataract_per_eye": ""
}},

"special_conditions": [
]
}}

Return ONLY JSON

INPUT:
{text}
"""
    return call_gpt(prompt)


# ---------------------------
# RETRY
# ---------------------------
def extract_with_retry(text):
    for _ in range(2):
        output = run_extraction(text)
        parsed = validate_json(output)
        if parsed:
            return parsed
    return None


# ---------------------------
# HIGHLIGHTS
# ---------------------------
def generate_highlights(json_data):

    prompt = f"""
You are an insurance policy transparency expert.

Generate:

• Activation insight
• Risk insight
• Coverage insight
• Hidden cost insight

Plain English
Short bullets
Financial impact only

INPUT:
{json.dumps(json_data)}
"""

    return call_gpt(prompt)


# ---------------------------
# BASIC SUMMARY
# ---------------------------
def generate_basic_summary(json_data):

    prompt = f"""
Write 2-3 lines explaining:

Policy type
Deductible behaviour
Coverage nature

No judgement

INPUT:
{json.dumps(json_data)}
"""

    return call_gpt(prompt)


# ---------------------------
# FULL DETAILED ANALYSIS
# ---------------------------
def run_analysis(json_data):

    prompt = f"""
Prompt for Json analysis

You are an insurance policy transparency expert.

You are given structured insurance policy data in JSON format.

Your task is NOT to rate, score, judge, recommend, or advise.

Your goal is to clearly explain:

How this policy behaves in real life

What the user will actually pay

When the policy helps and when it does not

What conditions affect payouts

How financial responsibility shifts between user and insurer


CORE RULES:

Do NOT assign scores or ratings

Do NOT say "good", "bad", "best", "recommended"

Do NOT suggest actions

Do NOT repeat raw JSON

Do NOT use technical insurance jargon

Translate policy clauses into real-life financial impact

If any data is missing → say "Not specified in policy"


OUTPUT FORMAT (STRICT MARKDOWN)

POLICY ANALYSIS REPORT

[Policy Name] — [Policy Number]

------------------------------------------------

📋 POLICY SUMMARY

Field | Details
Policy Name |
Insurer |
Policy Type |
Policy Classification |
Sum Insured |
Deductible |
Co-Pay |
Room Rent Limit |
Premium Paid |
Policy Period |
Members Covered |

------------------------------------------------

🔴 ACTIVATION BARRIER

Explain clearly:

If deductible exists  
How much user pays first

If none  
State clearly

------------------------------------------------

🔍 TOP 3 REALITY CHECKS

3 bullets

Biggest financial truths

------------------------------------------------

💰 HOW THIS WORKS IN REAL LIFE

Scenario | Estimated Bill | Insurance Pays | You Pay

Small claim  
Medium claim  
Near deductible  
Large claim

Use Indian cost ranges

------------------------------------------------

⚠️ KEY LIMITATIONS & CONDITIONS

Waiting periods  
Restrictions  
Coverage limits

------------------------------------------------

⚠️ HIDDEN COST LIMITS

Procedure caps  
Room rent  
Sublimits

------------------------------------------------

⏳ WHEN YOU CANNOT USE THIS POLICY

Waiting periods  
Pre-existing  
Initial wait

------------------------------------------------

✅ WHEN THIS POLICY HELPS

Large bills  
Serious illness  

------------------------------------------------

❌ WHEN THIS POLICY DOES NOT HELP

Small claims  
Frequent usage  

------------------------------------------------

🧠 PLAIN ENGLISH SUMMARY

2-3 lines

------------------------------------------------

🧾 PRACTICAL INTERPRETATION

User financial responsibility

------------------------------------------------

🔎 FINANCIAL RISK AREAS

User still pays

------------------------------------------------

💡 CLAIM BEHAVIOR INSIGHT

Claim flow explanation

INPUT JSON:
{json.dumps(json_data)}
"""

    return call_gpt(prompt)


# ---------------------------
# STREAMLIT UI
# ---------------------------

st.set_page_config(
    page_title="Check Your Policy",
    page_icon="🛡️",
    layout="wide"
)

title_col, nav_col = st.columns([3,2])

with title_col:
    st.title("🛡️ Check Your Policy")
    st.caption("Understand your insurance policy coverage, risks and limitations instantly")

with nav_col:

    col1, col2, col3, col4 = st.columns(4)

    if "menu" not in st.session_state:
        st.session_state.menu = None

    with col1:
        if st.button("🎯 Vision"):
            st.session_state.menu = None if st.session_state.menu == "vision" else "vision"

    with col2:
        if st.button("ℹ️ How To"):
            st.session_state.menu = None if st.session_state.menu == "howto" else "howto"

    with col3:
        if st.button("🔒 Privacy"):
            st.session_state.menu = None if st.session_state.menu == "privacy" else "privacy"

    with col4:
        if st.button("🚧 Upcoming"):
            st.session_state.menu = None if st.session_state.menu == "upcoming" else "upcoming"

if st.session_state.menu == "vision":
    st.markdown("### 🎯 Vision")
    st.info(
    "Insurance policies are complex and often difficult to understand. "
    "CheckYourPolicy simplifies policies and helps users understand coverage, risks, "
    "and claim behaviour."
    )

elif st.session_state.menu == "howto":
    st.markdown("### ℹ️ How To")
    st.info("""
1. Upload your insurance policy or policy draft (PDF)  
2. Review quick understanding and highlights  
3. Generate detailed report (optional)
""")

elif st.session_state.menu == "privacy":
    st.markdown("### 🔒 Privacy")
    st.info(
    "Your uploaded documents are processed securely and are not stored permanently."
    )

elif st.session_state.menu == "upcoming":
    st.markdown("### 🚧 Upcoming Features")
    st.info("""
• Policy comparison across insurers  
• Claim scenario simulation  
• Coverage gap detection  
• Advisor connect  
• Renewal alerts  
• Family policy optimization  
""")
 
st.markdown("---")
uploaded_file = st.file_uploader("Upload Policy PDF to start analysis", type="pdf")

if uploaded_file:

    if st.button("Get Basic Summary"):

        with st.spinner("Analyzing policy..."):

            text = extract_text(uploaded_file)

            parsed_json = extract_with_retry(text)

            if not parsed_json:
                st.error("Extraction failed")
            else:

                highlights = generate_highlights(parsed_json)
                summary = generate_basic_summary(parsed_json)

                st.markdown("## 🛡️ Policy Snapshot")

                st.markdown(f"""
Policy Name: {parsed_json.get('policy_name')}  
Insurer: {parsed_json.get('insurer')}  
Policy Type: {parsed_json.get('policy_type')}  
Sum Insured: {parsed_json.get('sum_insured')}  
Deductible: {parsed_json.get('deductible')}  
Co-Pay: {parsed_json.get('copay')}  
Room Rent: {parsed_json.get('room_rent_limit')}  
Members: {parsed_json.get('members_count')}
""")
                st.markdown("## 🧠 Quick Understanding")
                st.markdown(summary)

                st.markdown("## ⭐ Key Highlights")
                st.markdown(highlights)

                st.session_state["policy_json"] = parsed_json

if "policy_json" in st.session_state:

    if st.button("Generate Detailed Report"):

        with st.spinner("Generating detailed report..."):

            report = run_analysis(st.session_state["policy_json"])

            st.markdown(report)
st.caption("Supports Health, Life, Motor Insurance Policies")
st.markdown("---")

st.caption(
"CheckYourPolicy analyzes insurance policies to highlight coverage, limitations, and financial risks."
)

st.caption(
"This tool does not replace your insurer or advisor. Always refer to original policy wording."
)

st.caption("© 2026 CheckYourPolicy. All Rights Reserved.")
