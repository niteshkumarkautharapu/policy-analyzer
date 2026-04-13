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
"policy_category": "",
"policy_classification_hint": "",

"is_individual_policy": "",
"is_family_floater": "",
"is_group_policy": "",
"is_topup": "",
"restoration_type": "",
"specified_diseases": [],
"proportionate_deduction": "",
"network_hospital_count": "",
"continuity_benefit": "",
"premium_loading": "",
"is_super_topup": "",
"is_critical_illness": "",
"is_personal_accident": "",
"is_senior_citizen_policy": "",

"sum_insured": "",
"sum_insured_type": "",

"deductible": "",
"deductible_type": "",
"copay": "",

"room_rent_limit": "",
"room_rent_type": "",

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
"pre_post_conditions": "",
"day_care": "",
"ayush": "",
"domiciliary": "",
"ambulance_limit": "",
"donor_expense": ""
}},

"coverage_features": {{
"restoration_benefit": "",
"restoration_conditions": "",
"no_claim_bonus": {{
"available": "",
"bonus_percent": "",
"max_bonus": "",
"reset_conditions": ""
}},
"cashless_network": "",
"network_hospitals": "",
"tpa": "",
"cashless_scope": "",
"modern_treatment": {{
"covered": "",
"sublimit": "",
"conditions": ""
}},
"organ_donor": "",
"home_care": ""
}},

"exclusions": {{
"maternity": {{
"covered": "",
"waiting_period": "",
"limit": "",
"newborn_cover": ""
}},
"opd": {{
"covered": "",
"limit": "",
"conditions": ""
}},
"cosmetic": "",
"non_medical": "",
"permanent_exclusions": []
}},

"waiting_periods": {{
"initial_days": "",
"specified_disease_months": "",
"pre_existing_months": "",
"maternity_waiting": ""
}},

"sublimits": {{
"room_rent": "",
"icu_limit": "",
"cataract_per_eye": "",
"maternity_limit": "",
"ambulance_limit": ""
}},

"financial_conditions": {{
"copay_conditions": "",
"deductible_conditions": "",
"floater_risk": ""
}},

"age_conditions": {{
"entry_age": "",
"exit_age": "",
"renewability": ""
}},

"renewal_conditions": "",

"insurer_metrics": {{
"claim_settlement_ratio": "",
"grievance_ratio": ""
}},

"special_conditions": []
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

Your goal:
Help users quickly understand:

• Where they are covered  
• Where they have risks  
• When insurance becomes useful  

Avoid technical insurance terminology and Use the following structure:

------------------------------------------------

## 🧠 Quick Understanding

Create a simple table:

Item | Summary
Coverage Type |
Who is Covered |
Total Coverage |
Insurance Starts |
Best For |

Rules:

• Use simple plain english, non-technical language  
• Convert insurance terms into plain English  
• Policy type must describe behaviour, not classification  
• Mention deductible if present  
• Mention family sharing if applicable  
• Focus on real-world meaning  
• Keep summaries short and intuitive  

------------------------------------------------

## 🟢 You are covered for

List 7-8 short bullets:

Each bullet must follow:

Scenario — What it means for user

Examples:

• Large hospitalization — Insurance becomes useful after deductible  
• Surgeries — Major procedures covered within total coverage  
• Emergency admission — Covered within policy limits  
• ICU treatment — High-cost care covered after deductible  

Rules:

• Keep bullets short  
• Avoid technical jargon  
• Avoid waiting period details  
• Avoid long explanations  

------------------------------------------------

## 🔴 Be Careful when

List 5–7 short bullet points:

• Deductible exposure  
• Coverage limitations  
• Shared coverage risks  
• High-level exclusions  
• Financial exposure  

Examples:

• Small hospitalization — First expenses paid by you  
• Multiple family claims — Shared coverage reduces faster  
• Limited ambulance — Small portion covered  

Rules:

• Avoid deep exclusions  
• Avoid waiting period details  
• Focus on financial risk  

------------------------------------------------

## 💡 What This Means

Write 2-3 short sentences:

Explain:

• When policy helps most  
• Where user may still pay  

Plain English only  
No recommendations  
No judgement  

------------------------------------------------
CONTENT STYLE RULES:

• Each bullet should have short explanation  
• Use format: bullet — short explanation  
• Avoid long paragraphs  
• Avoid single-word bullets  
• Keep explanation under one line  

------------------------------------------------

READABILITY RULES:

• Use tables where helpful  
• Use short bullet points  
• Avoid paragraphs  
• Maximum 1–2 lines per bullet  
• Avoid dense text  
• Make output easy to scan  

------------------------------------------------

SIMPLIFICATION RULES:

• Convert technical insurance terms into plain English  
• Avoid words like indemnity, floater, benefit unless explained  
• Focus on user understanding  
• Focus on real-world behaviour  

------------------------------------------------

IMPORTANT RULES:

• Use medium-level English  
• Avoid judgement words  
• Avoid recommendations  
• Avoid repeating content  
• Avoid assumptions  
• Keep tone neutral  

------------------------------------------------

Do NOT include:

• Waiting period details  
• Sublimits  
• Detailed exclusions  
• Premium vs value comparisons  

These belong to detailed report.

------------------------------------------------

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
Your task is to analyze the insurance policy JSON and generate a clear, decision-focused report.

The report must help users understand:

• Will insurance actually pay
• How much user may still pay
• Hidden risks
• Overall protection behaviour

Use policy-specific reasoning only.

IMPORTANT RULES:

• Use medium-level English
• Use simple, clear sentences
• Avoid technical jargon
• Avoid generic insurance education
• Avoid advisory language
• Avoid judgement words
• Avoid assumptions
• Use policy-specific reasoning only
• Focus on financial impact

If information missing:

Write:

"Not specified in policy"
or
"Depends on insurer claim assessment"

------------------------------------------------

UNCERTAINTY RULE

If policy data is unclear:

Do NOT assume risk

Instead say:

"Depends on policy wording"

Avoid speculative risks.

-----------------------------------------------------

NON-DUPLICATION RULE

Do not repeat:

• Scenario risks in financial risks
• Financial risks in hidden risks
• Hidden risks already explained

Each section must provide new insights.

------------------------------------------------------

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

Rules:

• Convert technical terms to plain English
• Avoid insurance jargon
• Keep explanations short
------------------------------------------------

## 🟢 How much Insurance will pay

Add disclaimer:

Based on common claim settlement patterns observed in Indian health insurance and IRDAI-reported claim behaviours, the following scenarios illustrate how claims may typically be paid, partially paid, or rejected depending on policy conditions. Actual claim outcomes depend on insurer assessment and policy terms.

Create one combined table:

Scenario | Claim Outcome | What Happens | Why | Financial Impact

Claim Outcome must be:

• Approved 
• Partially Approved 
• May Be Rejected 

SCENARIO DERIVATION RULE

Derive scenarios dynamically from policy data.

Analyze:

• Deductible 
• Copay 
• Waiting periods 
• Coverage scope 
• Sublimits 
• Room rent limits 
• Coverage type 
• Floater structure 
• Sum insured 
• Exclusions 
• Eligibility rules 
• Restoration rules 
• Network rules 
• Member structure 
• Policy tenure 
• Special conditions 


REAL-WORLD CLAIM SCENARIO CATEGORIES TO CONSIDER

Based on IRDAI and insurer claim behaviour:

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

SCENARIO PRIORITIZATION RULE

Consider at least 10 internally

Output Top 8 scenarios

Prioritize:

• Highest financial impact
• Most realistic scenarios
• Most likely situations
• Clearly classify claim outcome 
• Use realistic Indian healthcare costs 
• Use policy-specific reasoning only 

Avoid:

• Generic scenarios
• Minor scenarios
• Duplicate scenarios

--------------------------------------------------

CLAIM OUTCOME CLASSIFICATION RULE

Use "Approved" only when:
• Covered treatment
• No deductible impact
• No major limitations

Use "Partially Approved" when:
• Deductible applies
• Room rent limits apply
• Sublimits apply
• Coverage shared (floater)
• Non-medical exclusions apply
• Copay applies

Use "May Be Rejected" only when:
• Explicit exclusion in policy
• Waiting period not completed
• Policy expired or invalid
• Non-covered treatment
• Eligibility conditions not met

Do not classify as "Rejected" unless supported by policy data.

------------------------------------------------

## ⚠️ How Much You May Still Pay

Analyze the policy and identify financial risks based ONLY on policy data.

Create table:

Risk Category | Risk Scenario | When It Happens | Financial Impact | Policy Reference

------------------------------------------------

RISK DERIVATION RULE

Derive financial risks dynamically from policy JSON.

Analyze the entire policy including:

• Coverage details  
• Financial conditions  
• Waiting periods  
• Exclusions  
• Sublimits  
• Room rent limits  
• Deductible  
• Copay  
• Sum insured structure  
• Floater structure  
• Eligibility rules  
• Restoration conditions  
• Network hospital rules  
• Special conditions  
• Any other coverage limitations  

------------------------------------------------

RISK IDENTIFICATION LOGIC

Identify risks that may:

• Delay claim payment  
• Reduce claim payment  
• Reject claim  
• Limit coverage  
• Create out-of-pocket exposure  

Generate risks only if supported by policy data.

------------------------------------------------

STRICT GUARDRAILS

Do NOT:

• Assume risks not mentioned in policy  
• Generate generic insurance risks  
• Use industry assumptions  
• Create hypothetical risks  
• Provide advice or recommendations  
• Use judgement words (good, bad, weak, strong)

If information not available:
Do not generate that risk

------------------------------------------------

POLICY-ONLY RULE

Each risk must:

• Be derived directly from policy JSON  
• Reference policy condition  
• Avoid generic insurance explanations  
• Avoid assumptions  

Bad Example:
"Consumables may not be covered"

Good Example:
"Non-medical expenses excluded — Consumables paid by user"

------------------------------------------------

FINANCIAL IMPACT RULE

Each risk must include:

• When risk happens  
• Financial impact  
• Real-world behaviour  

Avoid vague statements.

Bad:
"Partial payment possible"

Good:
"Room rent limit ₹5,000 — Higher room leads to proportionate deduction"

------------------------------------------------

RISK PRIORITY RULE

Generate:

Maximum 6-8 risks

Prioritize:

• Highest financial exposure
• Most realistic risks
• User decision-impact risks

Avoid:

• Minor risks
• Generic risks
• Duplicate risks

------------------------------------------------

##⚠️ Hidden Surprises / Risks In Your Policy

Create table:

Hidden Risk | Why It Matters | Possible Impact

Focus on:

• Hidden clauses 
• Coverage gaps 
• Financial exposure 
• Lesser-known limitations 

Examples :

• Deductible behaviour 
• Family floater impact or sharing
• Restoration conditions 
• Policy activation timing 
• Coverage exhaustion 
• Non-covered items 
• Documentation conditions 
• Eligibility conditions 
• Subtle exclusions 
• High deductible vs sum insured

RULES:

• Must be policy-specific 
• Avoid repeating Financial Risk section 
• Avoid generic insurance explanations 
• Focus on surprises users may miss 

HIDDEN RISK IDENTIFICATION RULE

Hidden risks should identify:

• Policy behaviour users may not expect
• Financial exposure not obvious
• Coverage gaps created by structure
• Interaction between policy conditions
• Non-obvious risks

Avoid obvious risks already covered in previous section.

------------------------------------------------

## 5. 🧠 Overall Summary

Write 4-6 short sentences explaining:

• When this policy helps most 
• When it may not help 
• Key financial behaviour 
• Overall protection behaviour 

RULES:

• Plain English 
• No bullet points 
• No recommendations 
• No judgement words 
• No advisory language 

------------------------------------------------

IMPORTANT GUARDRAILS

• Use medium-level English 
• Avoid technical jargon 
• Avoid abbreviations 
• Avoid recommendations 
• Avoid advisory tone 
• Avoid judgement words 
• Avoid assumptions 
• Avoid generic insurance explanations 
• Use policy-specific reasoning only 

If information missing:

Write:
"Depends on insurer claim assessment"
or
"Not specified in policy"

------------------------------------------------

READABILITY RULES

• Use tables wherever possible 
• Avoid long paragraphs 
• Use short sentences 
• Make content scannable 
• Avoid dense blocks of text 

------------------------------------------------

ABBREVIATION RULES

Avoid abbreviations:

Do not use:

• SI 
• OOP 
• PED 
• IPD 
• OPD 

Use full forms instead.

------------------------------------------------

INPUT JSON:
{json.dumps(json_data)}
"""
    return call_gpt(prompt, ANALYSIS_MODEL)

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.set_page_config(
    page_title="CheckYourPolicy",
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
    st.caption("Understand your insurance policy coverage, risks and limitations.")

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
        if st.button("Upcoming", use_container_width=True):
            st.session_state.menu = None if st.session_state.menu == "Upcoming" else "Upcoming"

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
            "Help people understand their insurance before they need it."
        )

       
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

    elif st.session_state.menu == "Upcoming":
 
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
        type=["pdf"],
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

    basic_report = basic_report.replace("```markdown", "").replace("```", "")
    st.markdown(basic_report)
    st.markdown("---")


    # ---------------------------
    # FEEDBACK BLOCK — Basic Report
    # ---------------------------

    st.markdown("#### Was this summary helpful?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "👍 Yes",
            key="basic_yes",
            type="primary" if st.session_state.feedback_value_basic == "Helpful" else "secondary"
        ):
            st.session_state.feedback_value_basic = "Helpful"

    with col2:
        if st.button(
            "👎 No",
            key="basic_no",
            type="primary" if st.session_state.feedback_value_basic == "Not Helpful" else "secondary"
        ):
            st.session_state.feedback_value_basic = "Not Helpful"


    if st.session_state.feedback_value_basic:

        comment = st.text_area(
            "Tell us more (optional)",
            placeholder="What worked well or what can be improved?",
            key="basic_comment_box"
        )

        if st.button("Submit Feedback", key="basic_submit"):

            save_feedback(
                parsed_json.get("policy_name", "Unknown"),
                "Basic Report",
                st.session_state.feedback_value_basic,
                comment
            )

            st.session_state.feedback_submitted_basic = True
            st.success("✅ Thanks for your feedback!")


    st.markdown("---")


    # ---------------------------
    # Detailed Report Nudge
    # ---------------------------

    st.markdown("## 🔎 Want Deeper Analysis?")
    st.markdown("""
Based on your policy details, the **Detailed Report** helps you understand:

• How your policy behaves in real-life claim situations  
• Claim rejection scenarios based on policy conditions  
• When insurance actually pays vs when you may still pay  
• Hidden costs and financial exposure areas  
• Coverage gaps that are not obvious in summary  

This helps you understand **how your policy may perform when you actually need it.**
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

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Yes", key="detailed_yes"):
                st.session_state.feedback_value_detailed = "Helpful"

        with col2:
            if st.button("👎 No", key="detailed_no"):
                st.session_state.feedback_value_detailed = "Not Helpful"


    if st.session_state.feedback_value_detailed:

        comment = st.text_area(
            "Tell us more (optional)",
            placeholder="What can be improved or what was useful?",
            key="detailed_comment_box"
        )

        if st.button("Submit Feedback", key="detailed_submit"):

            save_feedback(
                st.session_state["policy_json"].get("policy_name", "Unknown"),
                "Detailed Report",
                st.session_state.feedback_value_detailed,
                comment
            )

            st.session_state.feedback_submitted_detailed = True
            st.success("Thanks for your feedback!")

    st.markdown("---")
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
