import os
import json
import time
import tempfile

import pdfplumber
import streamlit as st
from openai import OpenAI
from datetime import datetime
from notion_client import Client

def save_feedback(policy, report, feedback, comment):

    try:
        notion = Client(auth=st.secrets["NOTION_TOKEN"])

        notion.pages.create(
            parent={"database_id": st.secrets["NOTION_DATABASE_ID"]},
            properties={
                "Title": {
                    "title": [
                        {
                            "text": {
                                "content": str(policy)
                            }
                        }
                    ]
                },
                "Report": {
                    "select": {
                        "name": str(report)
                    }
                },
                "Feedback": {
                    "select": {
                        "name": str(feedback)
                    }
                },
                "Comment": {
                    "rich_text": [
                        {
                            "text": {
                                "content": str(comment)
                            }
                        }
                    ]
                },
                "Date": {
                    "date": {
                        "start": datetime.now().isoformat()
                    }
                }
            }
        )

    except Exception as e:
        st.error(f"Notion Error: {e}")
# ---------------------------
# CONFIG
# ---------------------------

# FIX: validate API key at startup instead of silently passing None
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY environment variable is not set.")
    st.stop()

client = OpenAI(api_key=api_key)

# FIX: gpt-5-mini does not exist — use gpt-4o for analysis
EXTRACTION_MODEL = "gpt-4o-mini"
ANALYSIS_MODEL = "gpt-4o"

# ---------------------------
# Session State Initialization
# ---------------------------

if "show_basic" not in st.session_state:
    st.session_state.show_basic = False

if "show_detailed" not in st.session_state:
    st.session_state.show_detailed = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "detailed_report" not in st.session_state:
    st.session_state.detailed_report = None

if "highlights" not in st.session_state:
    st.session_state.highlights = None

if "summary" not in st.session_state:
    st.session_state.summary = None

# FIX: basic_report was used but never initialised in session state
if "basic_report" not in st.session_state:
    st.session_state.basic_report = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "menu" not in st.session_state:
    st.session_state.menu = None

if "feedback_submitted_basic" not in st.session_state:
    st.session_state.feedback_submitted_basic = False
if "feedback_value_basic" not in st.session_state:
    st.session_state.feedback_value_basic = None
if "feedback_comment_basic" not in st.session_state:
    st.session_state.feedback_comment_basic = ""

if "feedback_submitted_detailed" not in st.session_state:
    st.session_state.feedback_submitted_detailed = False
if "feedback_value_detailed" not in st.session_state:
    st.session_state.feedback_value_detailed = None
if "feedback_comment_detailed" not in st.session_state:
    st.session_state.feedback_comment_detailed = ""




# ---------------------------
# Extract text from PDF
# ---------------------------

def extract_text(file):
    # FIX: seek(0) so re-runs don't read from EOF
    file.seek(0)
    text = ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    finally:
        # FIX: always delete the temp file to avoid accumulation
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
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
    # FIX: catch only JSONDecodeError, not every possible exception
    try:
        json_str = clean_json(json_str)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ---------------------------
# OpenAI Call
# ---------------------------

def call_gpt(prompt, model):
    # FIX: use client.chat.completions.create — client.responses does not exist
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


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

WAITING PERIOD INTERPRETATION RULES:

Health insurance waiting periods behave differently for new policies and renewals.

IMPORTANT:

• Initial waiting period (30 days) applies only if policy is new
• Pre-existing disease waiting period applies only if not already completed
• Specified disease waiting period applies only if not already completed
• If renewal status is not specified, treat waiting periods as conditional
• Do NOT assume policy is new
• Do NOT state waiting periods as absolute unless explicitly mentioned

Always present waiting periods as conditional:

Example:

"If this is a new policy, 30-day waiting period applies"

"If waiting period already completed, this restriction may not apply"

Avoid definitive statements that may mislead users.

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
    return call_gpt(prompt, EXTRACTION_MODEL)


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
# Basic Report
# ---------------------------

def generate_basic_report(json_data):
    prompt = f"""
You are an insurance policy transparency expert.

Generate a Basic Policy Summary using the following structure:

------------------------------------------------

## 🧠 Quick Understanding

Explain in plain English:

• What this policy is  
• Who it protects  
• What kind of medical situations it is useful for  

Write in short natural paragraphs (not bullet lists).

------------------------------------------------

## 📌 What This Policy Means For You

Explain:

• When this policy is most useful  
• How coverage behaves in real-life  

Focus on practical understanding rather than policy terminology.

------------------------------------------------

## ⚠️ How Costs Are Shared

Explain clearly:

• Deductible / Copay / Floater behaviour  
• When insurance starts paying  
• How this impacts real-world claims  

Keep explanation simple and practical.

------------------------------------------------

## ⭐ Key Highlights

Generate 5-6 short bullets only:

• Activation insight  
• Risk insight  
• Coverage insight  
• Hidden cost insight  

Focus only on meaningful insights. Avoid repetition.

------------------------------------------------

IMPORTANT RULES:

• Use medium-level English  
• Avoid technical jargon  
• Avoid judgement words  
• Avoid recommendations  
• Avoid listing too many coverage features  
• Prefer short paragraphs instead of long bullet lists  
• Avoid repeating content across sections  
• Focus on user understanding  
• Avoid assumptions  
• Keep tone neutral and informative  

Do NOT include:

• Waiting period details  
• Sublimit details  
• Detailed exclusions  
• Premium vs value comparisons  

These belong to detailed report.

INPUT:
{json.dumps(json_data)}

Return structured markdown output.
"""
    return call_gpt(prompt, ANALYSIS_MODEL)

# ---------------------------
# FULL DETAILED ANALYSIS
# ---------------------------

def run_analysis(json_data):
    prompt = f"""
You are an insurance policy behaviour analysis expert.

IMPORTANT RULES:

- Use medium-level English
- Use simple, clear sentences
- Avoid technical jargon and complex words
- Prefer tables wherever applicable (except interpretation sections)
- Focus on real-life interpretation
- Avoid repetition across sections
- If any data missing → say "Not specified in policy"


INSURANCE DOMAIN RULES:

- Do not assume policy is good or bad
- Do not use judgement words like strong, weak, good, poor
- Focus on behaviour, not recommendation
- Explain financial impact wherever possible
- Use numbers instead of vague statements
- Use policy-specific interpretation only
- Clearly explain deductible behaviour
- Explain when insurance starts paying after deductible


WAITING PERIOD RULES:

- Initial waiting period applies mainly if policy is new
- Pre-existing disease waiting applies only if waiting period not already completed
- Specified disease waiting applies only if waiting period not completed
- If renewal status not mentioned, state applicability conditionally
- Avoid absolute statements


INLINE NOTE RULES:

- For waiting period related items, add short inline note
- Mention whether applicable mainly for new policy or renewal
- Keep note short and simple


SECTION DIFFERENTIATION RULES:

Policy Snapshot  
- Only factual policy data  
- No interpretation  

What This Policy Really Means  
- High level interpretation  
- 2-4 lines only  

Real-Life Claim Behaviour  
- Scenario-based table  
- Show financial behaviour  

Where This Policy Helps — And Where It Doesn't  
- Situational comparison only  

Your Financial Exposure  
- Out-of-pocket risks only  
- Deductible  
- Copay  
- Sublimits  
- Floater risk  

Key Policy Constraints  
- Waiting periods  
- Eligibility  
- Exclusions  

Understanding Your Coverage  
- Coverage behaviour explanation  


REAL-LIFE SCENARIO RULES:

- Use realistic Indian hospital costs
- Include minor, medium, large and critical scenarios
- Show insurance pays vs you pay
- Show deductible impact


ABBREVIATION RULES:

- Avoid abbreviations like SI, OOP, PED, OPD, IPD
- Use full terms
- If abbreviation used, explain once


OUTPUT RULES:

- Do not ask questions
- Do not include suggestions
- Do not repeat sections
- End report cleanly
- Short explanatory paragraph
- Structured table where applicable
- Optional interpretation sentence

Return STRICT MARKDOWN FORMAT
------------------------------------------------

POLICY ANALYSIS REPORT

[Policy Name] — [Policy Number]

------------------------------------------------

## 📊 Policy Snapshot

Create a table:

Field | Details
Policy Name |
Insurer |
Policy Type |
Policy Classification |
Sum Insured |
Deductible |
Co-Pay |
Room Rent Limit |
Premium |
Policy Period |
Members Covered |

------------------------------------------------

## 🧠 What This Policy Really Means

What This Policy Really Means  
- 4-6 explanatory sentences  
- Explain real-world behaviour  
- Avoid bullet format

Plain English only

------------------------------------------------

## 💰 Real-Life Claim Behaviour

Add disclaimer:

Based on common claim settlement patterns observed in Indian health insurance and publicly available insurance data, the following scenarios illustrate how claims may typically be paid, partially paid, or rejected depending on policy conditions. Actual claim outcomes depend on insurer assessment and policy terms.

Use policy-specific information only.

------------------------------------------------

## 💰 Real-Life Claim Behaviour

Add disclaimer:

Based on common claim settlement patterns observed in Indian health insurance and IRDAI-reported claim behaviours, the following scenarios illustrate how claims may typically be paid, partially paid, or rejected depending on policy conditions. Actual claim outcomes depend on insurer assessment and policy terms.

Use policy-specific information only.

Analyze policy information:

• Sum insured  
• Deductible  
• Co-pay  
• Waiting periods  
• Sublimits  
• Room rent limits  
• Coverage scope  
• Exclusions  
• Special conditions  
• Members covered  

IMPORTANT REASONING RULE:

• Consider at least 10 realistic scenarios internally  
• Select the most relevant 5 scenarios based on policy conditions  
• Avoid generic or repetitive scenarios  
• Prioritize scenarios with financial impact  

------------------------------------------------

### 🟢 Claims Typically Paid

Create table:

Scenario | Why Claim Usually Paid | Real-Life Example | Financial Outcome

Reasoning Examples:

• Large hospitalization after waiting period  
• Accident hospitalization  
• Emergency hospitalization  
• ICU admission  
• Covered surgeries  
• Network hospital admission  
• Day-care procedures covered  
• Critical illness hospitalization  
• Covered inpatient treatment  
• Post hospitalization claims  

Output:

• Select Top 5 most relevant scenarios  
• Use policy-specific coverage  

------------------------------------------------

### 🟡 Claims Partially Paid

Create table:

Scenario | Why Partially Paid | Real-Life Example | Financial Outcome

Reasoning Examples:

• Deductible applicable  
• Room rent limit exceeded  
• Sublimit applicable  
• Copay applicable  
• Non-medical expenses excluded  
• Floater sum insured sharing  
• Procedure limits  
• Consumable exclusions  
• Day-care limit  
• Network hospital differences  

Output:

• Select Top 5 most relevant scenarios  
• Focus on financial impact  

------------------------------------------------

### 🔴 Claims That May Be Rejected

Create table:

Scenario | Why Claim May Be Rejected | Real-Life Example | Financial Outcome

Reasoning Based on Indian Claim Trends:

• Waiting period not completed  
• Pre-existing disease waiting  
• Non-disclosure risk  
• Policy expired  
• Non-covered treatment  
• Non-medically necessary hospitalization  
• Documentation issues  
• Policy condition violation  
• Coverage exhaustion  
• Experimental treatment  

Output:

• Select Top 5 most relevant scenarios  
• Avoid generic scenarios  
• Focus on realistic claim rejection  

------------------------------------------------

IMPORTANT RULES:

• Use medium-level English  
• Avoid technical jargon  
• Avoid abbreviations  
• Avoid duplication across sections  
• Avoid generic insurance explanations  
• Use Indian healthcare cost examples  
• If information missing → say "Depends on insurer claim policy"

Return structured tables only

------------------------------------------------

## ⚖️ Where This Policy Helps — And Where It Doesn't

Create comparison table:

Where This Policy Helps | Where This Policy Doesn't Help

Output:

List atleast top 10 relevant scenarios
------------------------------------------------

## ⚠️ Where You May Have To Pay From Your Pocket

Create table:

Risk Area | Why This Matters | Financial Impact

------------------------------------------------

## 🚧 Key Policy Constraints

Create table:

Constraint | What Policy Says | Impact

Include:

Waiting periods (mention conditional applicability for renewals)
Pre-existing disease rules (mention if waiting already completed)  
Specific disease waiting  
Coverage restrictions  
Eligibility conditions  

------------------------------------------------

## 🔎 Understanding Your Coverage

Create table:

Coverage Element | What Policy Says | What It Means

Include:

List atleast top 10 relevant scenarios

------------------------------------------------

Provide short explanatory context where helpful
Avoid overly long paragraphs
Maintain readability and clarity

INPUT JSON:
{json.dumps(json_data)}
"""
    return call_gpt(prompt, ANALYSIS_MODEL)

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.set_page_config(
    page_title="Check Your Policy",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("""
<style>
table {
    width: 100% !important;
}
th, td {
    white-space: normal !important;
    word-wrap: break-word !important;
}
</style>
""", unsafe_allow_html=True)
title_col, nav_col = st.columns([3, 2])

with title_col:
    st.title("🛡️ Check Your Policy")
    st.caption("Understand your insurance policy coverage, risks and limitations instantly.")

# FIX: nav_col was empty — no buttons existed to set st.session_state.menu
with nav_col:
    st.markdown("<div style='padding-top:1.2rem'></div>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("Vision", use_container_width=True):
            st.session_state.menu = None if st.session_state.menu == "vision" else "vision"
    with n2:
        if st.button("About", use_container_width=True):
            st.session_state.menu = None if st.session_state.menu == "about" else "about"
    with n3:
        if st.button("Premium", use_container_width=True):
            st.session_state.menu = None if st.session_state.menu == "upcoming" else "upcoming"

# ---------------------------
# NAVIGATION CONTENT
# ---------------------------

menu_placeholder = st.container()

with menu_placeholder:

    if st.session_state.menu == "vision":
        st.markdown("### 🎯 Vision")

        st.info(
            "CheckYourPolicy aims to make insurance transparent and easy to understand, "
            "helping people clearly see how their policy behaves, where risks exist, "
            "and what to expect in real claim situations."
        )

        st.success("Help people understand their insurance before they need it.")

    elif st.session_state.menu == "about":
        st.markdown("### ℹ️ What is CheckYourPolicy")

        st.info(
            "CheckYourPolicy analyzes your insurance document using AI to explain "
            "coverage behaviour, hidden clauses, financial risks, and real-world claim impact "
            "in simple, structured insights."
        )

        st.markdown("""
### 🔎 What You Can Understand

• What is covered — and what is not  
• Hidden clauses that may affect claims  
• Deductible and cost-sharing behaviour  
• Financial exposure during hospitalization  
• Real-life claim scenarios  
• Common claim rejection risks  
• Policy limitations that are easy to miss  

---

Instead of reading lengthy policy documents, you get **clear insights** into how your insurance actually behaves in real situations.
""")

    elif st.session_state.menu == "upcoming":
        st.markdown("### 🔒 Premium Detailed Report")
        st.caption("""
• Clause-by-clause breakdown  
• Hidden conditions detection  
• Claim rejection risk analysis  
• Coverage gap identification  
• Sum insured adequacy analysis  
• Personalized risk insights  
• Financial risk explanation  
""")

        st.markdown("### 🚀 More Features")
        st.caption("""
• Motor and Life Insurance Category  
• Report Download  
• Multi Policy Comparison  
""")

st.markdown("---")

# ---------------------------
# Upload Section
# ---------------------------

st.markdown("### Upload your policy")

col1, col2, spacer = st.columns([2, 1, 3])

with col1:
    uploaded_file = st.file_uploader(
        "Upload policy",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        key=f"policy_uploader_{st.session_state.uploader_key}"
    )

with col2:
    clear_disabled = (
        uploaded_file is None
        and "policy_json" not in st.session_state
    )
    if st.button(
        "Reset",
        help="Clear uploaded policy",
        disabled=clear_disabled,
        use_container_width=True
    ):
        st.session_state.uploader_key += 1
        st.session_state.show_basic = False
        st.session_state.show_detailed = False
        st.session_state.file_uploaded = False
        st.session_state.detailed_report = None
        st.session_state.basic_report = None
        st.session_state.feedback_submitted_basic = False
        st.session_state.feedback_value_basic = None
        st.session_state.feedback_comment_basic = ""
        st.session_state.feedback_submitted_detailed = False
        st.session_state.feedback_value_detailed = None
        st.session_state.feedback_comment_detailed = ""
        st.session_state.pop("policy_json", None)
        st.session_state.pop("highlights", None)
        st.session_state.pop("summary", None)
        st.session_state.pop("last_uploaded", None)
        st.rerun()

# ---------------------------
# FIX: single upload/removal detection block (was duplicated)
# ---------------------------

if uploaded_file is not None:
    st.session_state.file_uploaded = True
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = uploaded_file.name
    elif uploaded_file.name != st.session_state.last_uploaded:
        st.session_state.show_basic = False
        st.session_state.show_detailed = False
        st.session_state.detailed_report = None
        st.session_state.basic_report = None
        st.session_state.feedback_submitted_basic = False
        st.session_state.feedback_value_basic = None
        st.session_state.feedback_comment_basic = ""
        st.session_state.feedback_submitted_detailed = False
        st.session_state.feedback_value_detailed = None
        st.session_state.feedback_comment_detailed = ""
        st.session_state.pop("policy_json", None)
        st.session_state.pop("highlights", None)
        st.session_state.pop("summary", None)
        st.session_state.last_uploaded = uploaded_file.name
else:
    st.session_state.show_basic = False
    st.session_state.show_detailed = False
    st.session_state.file_uploaded = False
    st.session_state.detailed_report = None
    st.session_state.basic_report = None
    st.session_state.pop("policy_json", None)
    st.session_state.pop("highlights", None)
    st.session_state.pop("summary", None)

# ---------------------------
# Basic Summary
# ---------------------------

if uploaded_file:
    st.success("✅ Policy uploaded successfully! Click **Basic Summary** to analyse your policy.")
    if st.button("Basic Summary"):
        st.session_state.show_basic = True

# FIX: single show_basic block (was split into two misaligned blocks causing IndentationError)
if st.session_state.show_basic and uploaded_file:

    if "policy_json" not in st.session_state:

        progress = st.progress(0)
        status = st.empty()

        status.info("📄 Reading policy document...")
        progress.progress(10)

        text = extract_text(uploaded_file)
        time.sleep(0.2)

        status.info("🔍 Extracting policy details, might take a few seconds...")
        progress.progress(35)

        parsed_json = extract_with_retry(text)

        if not parsed_json:
            status.error("Extraction failed. Please try again.")
            progress.empty()
            st.stop()

        st.session_state["policy_json"] = parsed_json
        time.sleep(0.2)

        status.info("🧠 Generating policy summary, please wait...")
        progress.progress(60)

        basic_report = generate_basic_report(parsed_json)
        st.session_state["basic_report"] = basic_report

        time.sleep(0.2)

        status.info("📊 Preparing report view...")
        progress.progress(85)
        time.sleep(0.3)

        progress.progress(100)
        status.success("✅ Basic summary ready")
        time.sleep(0.5)

        progress.empty()
        status.empty()

    parsed_json = st.session_state["policy_json"]
    basic_report = st.session_state["basic_report"]

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
    basic_report = basic_report.replace("```markdown", "").replace("```", "")
    st.markdown(basic_report)
    st.markdown("---")

    # ---------------------------
    # FEEDBACK BLOCK — Basic Report
    # ---------------------------

    st.markdown("#### Was this summary helpful?")

    if not st.session_state.feedback_submitted_basic:

        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 5])

        with fb_col1:
            if st.button("👍 Yes", key="basic_thumbs_up", use_container_width=True):
                save_feedback(
                    parsed_json.get("policy_name", "Unknown"),
                    "Basic Report",
                    "Helpful",
                    ""
                )
                st.session_state.feedback_submitted_basic = True
                st.rerun()

        with fb_col2:
            if st.button("👎 No", key="basic_thumbs_down", use_container_width=True):
                save_feedback(
                    parsed_json.get("policy_name", "Unknown"),
                    "Basic Report",
                    "Not Helpful",
                    ""
                )
                st.session_state.feedback_submitted_basic = True
                st.rerun()

    else:
        st.success("✅ Thank you for your feedback!")

    st.markdown("---")

    st.markdown("## 🔎 Want Deeper Analysis?")
    st.markdown("""
The detailed report provides deeper insights into how your policy behaves in real claim situations.

### Detailed Report Includes:

- Real-life claim rejection scenarios  
- Hidden clauses that impact claims  
- Financial risk areas and out-of-pocket exposure  
- When insurance actually pays vs when it doesn't  
- Waiting period impact (new vs renewal)  
- Deductible and sublimit behaviour  
- Coverage gaps and limitations  
- Practical interpretation of policy conditions  

This helps you understand **where your policy protects you — and where it may not.**
""")

    if st.button("🔒 Generate Detailed Report"):
        st.session_state.show_detailed = True
# ---------------------------
# Detailed Report
# ---------------------------

if st.session_state.show_detailed and "policy_json" in st.session_state:

    if st.session_state.detailed_report is None:

        progress = st.progress(0)
        status = st.empty()

        status.info("📊 Building policy snapshot...")
        progress.progress(10)
        time.sleep(0.3)

        status.info("🔍 Analyzing coverage behaviour...")
        progress.progress(25)
        time.sleep(0.3)

        status.info("⚠️ Identifying financial risks...")
        progress.progress(40)
        time.sleep(0.3)

        status.info("🚧 Evaluating policy constraints...")
        progress.progress(55)
        time.sleep(0.3)

        status.info("🧠 Interpreting real-life claim behaviour...")
        progress.progress(70)
        time.sleep(0.3)

        status.info("📄 Generating detailed report, please wait...")
        progress.progress(85)

        detailed = run_analysis(st.session_state["policy_json"])
        st.session_state.detailed_report = detailed

        progress.progress(100)
        status.success("✅ Detailed report ready")
        time.sleep(0.6)

        progress.empty()
        status.empty()

    report = st.session_state.detailed_report
    report = report.replace("```markdown", "").replace("```", "")

    st.markdown(report)
    st.markdown("---")

    # ---------------------------
    # FEEDBACK BLOCK — Detailed Report
    # ---------------------------

    st.markdown("#### Was this detailed report helpful?")

    if not st.session_state.feedback_submitted_detailed:

        fd_col1, fd_col2, fd_col3 = st.columns([1, 1, 5])

        with fd_col1:
            if st.button("👍 Yes", key="detailed_thumbs_up", use_container_width=True):
                save_feedback(
                    st.session_state["policy_json"].get("policy_name", "Unknown"),
                    "Detailed Report",
                    "Helpful",
                    ""
                )
                st.session_state.feedback_submitted_detailed = True
                st.rerun()

        with fd_col2:
            if st.button("👎 No", key="detailed_thumbs_down", use_container_width=True):
                save_feedback(
                    st.session_state["policy_json"].get("policy_name", "Unknown"),
                    "Detailed Report",
                    "Not Helpful",
                    ""
                )
                st.session_state.feedback_submitted_detailed = True
                st.rerun()

    else:
        st.success("✅ Thank you for your feedback!")


# ---------------------------
# Footer
# ---------------------------

st.caption("Supports Health Insurance Policies")
st.markdown("---")

with st.expander("📘 How To Use"):

   st.markdown("""
### 📄 Upload The Right Document

✅ That Work Best - Documents with **more coverage details** provide **better analysis and more accurate insights**.

• Policy schedule or certificate  
• Policy wording document  
• Renewal document with coverage details  

🚫 Avoid uploading: These may not contain full coverage details.

• Premium receipts  
• Emails  
• Incomplete screenshots  
• Payment confirmations  

### 🚀 How It Works

1. Upload your policy document  
2. Generate **Basic Summary**  
3. Review key coverage behaviour  
4. Generate **Detailed Report** (optional)  
5. Understand risks, exclusions and claim scenarios

---

### 🔒 Your Data Privacy

• Your document is used only for analysis  
• No policy data is stored permanently  
• No personal information is shared  

""")

if "footer" not in st.session_state:
    st.session_state.footer = None

footer_col1, footer_col2 = st.columns([1, 6])

with footer_col1:
    btn1, btn2 = st.columns([1, 1])
    with btn1:
        if st.button("Privacy", key="privacy_footer"):
            st.session_state.footer = None if st.session_state.footer == "privacy" else "privacy"
    with btn2:
        if st.button("Terms", key="terms_footer"):
            st.session_state.footer = None if st.session_state.footer == "terms" else "terms"

with footer_col2:
    st.markdown(
        """
        <div style="text-align:right; font-size:11px; color:#999;">
        © 2026 CheckYourPolicy
        </div>
        """,
        unsafe_allow_html=True
    )

if st.session_state.footer == "privacy":
    st.markdown("---")
    st.info("""
Privacy Policy

1. Document Security  
Uploaded insurance documents are processed securely using encrypted connections.  
We do not permanently store uploaded policy documents.

2. Data Usage  
Uploaded documents are used only for:

• Policy extraction  
• Analysis generation  
• Report creation  

Documents are not used for training models.

3. No Third-Party Sharing  
We do not sell, share, or distribute user documents or personal data to third parties.

4. Temporary Processing  
Documents may be temporarily processed in secure systems to generate reports and are discarded afterward.

5. Personal Information  
We do not require:

• Name  
• Phone number  
• Email  
• Identity information  

Unless voluntarily provided by users.

6. Payment Information (Premium Reports)  
Payments are handled by secure payment providers.  
We do not store card or banking details.

7. Data Retention  
Generated reports may be temporarily cached for performance but not permanently stored.

8. User Responsibility  
Users should avoid uploading documents containing unrelated sensitive personal information.

9. Policy Updates  
Privacy policy may be updated periodically to improve transparency and compliance.

10. Contact  
For privacy concerns, contact: support@checkyourpolicy.com
""")

if st.session_state.footer == "terms":
    st.markdown("---")
    st.info("""
Terms & Conditions

1. Informational Purpose Only  
CheckYourPolicy provides AI-generated analysis of insurance policy documents for informational purposes only.  
This does not constitute financial, legal, or insurance advice.

2. No Guarantee of Accuracy  
While we strive for accuracy, AI-generated insights may contain errors or omissions.  
Users must verify policy details directly with insurers or advisors.

3. No Liability  
CheckYourPolicy shall not be liable for:

• Claim rejection  
• Financial loss  
• Coverage misunderstanding  
• Policy purchase decisions  

Users are solely responsible for decisions made using this report.

4. Document Handling  
Uploaded documents are processed securely and not permanently stored.  
We do not share user documents with third parties.

5. Premium Reports  
Premium reports provide deeper analysis but do not guarantee:

• Claim approval  
• Coverage suitability  
• Financial outcome  

Premium reports remain informational.

6. Service Availability  
We reserve the right to:

• Modify features  
• Change pricing  
• Limit access  
• Suspend service  

Without prior notice.

7. Intellectual Property  
Reports generated are for personal use only.  
Commercial redistribution is not permitted.

8. Acceptance of Terms  
By using CheckYourPolicy, users agree to these Terms & Conditions.
""")
