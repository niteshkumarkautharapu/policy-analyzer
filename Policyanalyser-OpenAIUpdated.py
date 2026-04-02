import streamlit as st
import pdfplumber
from openai import OpenAI
import json
import tempfile
import os

# Session State Initialization
if "show_basic" not in st.session_state:
    st.session_state.show_basic = False

if "show_detailed" not in st.session_state:
    st.session_state.show_detailed = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
    
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
# ---------------------------
# FULL DETAILED ANALYSIS
# ---------------------------
def run_analysis(json_data):

    prompt = f"""
You are an insurance policy behaviour analysis expert.

You are given structured insurance policy data in JSON format.

Your job is NOT to:

• Rate the policy  
• Recommend policy  
• Judge policy  
• Compare policy  

Your job is to:

Explain clearly:

• How this policy behaves
• When it helps
• When it does not help
• What financial risks exist
• What coverage behaviour exists

Translate insurance language into real-life understanding.

IMPORTANT RULES:

• Use simple plain English
• Avoid technical jargon
• Use tables wherever applicable
• Avoid repeating JSON content
• Avoid long paragraphs
• Focus on real-life interpretation

If any data missing → say "Not specified in policy"

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

Write 2-4 lines explaining:

• What type of protection this policy provides
• Whether useful for small or large claims
• Overall behaviour

Plain English only

------------------------------------------------

## 💰 Real-Life Claim Behaviour

Create a table:

Scenario | Estimated Bill | Insurance Pays | You Pay | Behaviour

Include:

• Small hospitalization
• Medium hospitalization
• Large hospitalization
• Critical illness

Use Indian healthcare cost ranges

------------------------------------------------

## ⚖️ Where This Policy Helps — And Where It Doesn't

Create comparison table:

Where This Policy Helps | Where This Policy Doesn't Help

Examples:

Large hospitalization  
Small claims  
Critical illness  
Routine treatment  
Emergency cases  
Frequent claims  

------------------------------------------------

## ⚠️ Your Financial Exposure

Create table:

Risk Area | Why This Matters | Financial Impact

Examples:

Deductible  
Sublimits  
Waiting periods  
Room rent limits  
Copay  

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

Include if available:

Room rent  
Deductible  
Restoration benefit  
Day care coverage  
Pre hospitalization  
Post hospitalization  
AYUSH  
Network hospital  
Procedure limits  
Ambulance coverage  

------------------------------------------------

Keep explanations short, structured and clear.

Avoid long narrative paragraphs.

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
    st.caption("Understand your insurance policy coverage, risks and limitations instantly.")

with nav_col:

    col1, col2, col3 = st.columns(3)

    if "menu" not in st.session_state:
        st.session_state.menu = None

    with col1:
        if st.button("🎯 Vision"):
            st.session_state.menu = None if st.session_state.menu == "vision" else "vision"

    with col2:
        if st.button("ℹ️ About"):
            st.session_state.menu = None if st.session_state.menu == "about" else "about"

    with col3:
        if st.button("🚧 Upcoming"):
            st.session_state.menu = None if st.session_state.menu == "upcoming" else "upcoming"

menu_placeholder = st.container()
# ---------------------------
# NAVIGATION CONTENT
# ---------------------------

menu_placeholder = st.container()

with menu_placeholder:

    if st.session_state.menu == "vision":

        st.markdown("### 🎯 Vision")

        st.info(
        "Insurance policies are complex and difficult to understand. "
        "CheckYourPolicy simplifies insurance documents and highlights coverage, risks, "
        "limitations, and real-world claim behaviour."
        )


    elif st.session_state.menu == "about":

        st.markdown("### ℹ️ What is CheckYourPolicy")

        st.info(
        "CheckYourPolicy analyzes your insurance document using AI to identify coverage details, "
        "hidden clauses, exclusions, financial risks, and real-world claim impact."
        )

        st.caption("""
• Understand what is covered and what is not  
• Identify hidden clauses  
• Highlight financial risks  
• Avoid claim surprises  
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
• Multi Policy Comparision  
""")

st.markdown("---")
uploaded_file = st.file_uploader("Upload your policy", type=["pdf", "docx"])

# Detect File Upload
if uploaded_file is not None:
    st.session_state.file_uploaded = True

# Detect File Removal
if uploaded_file is None:
    st.session_state.show_basic = False
    st.session_state.show_detailed = False
    st.session_state.file_uploaded = False

if uploaded_file:

    if st.button("Basic Summary"):
        st.session_state.show_basic = True

if st.session_state.show_basic and uploaded_file:

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


if st.session_state.show_basic and "policy_json" in st.session_state:

    if st.button("🔒 Generate Detailed Report"):
        st.session_state.show_detailed = True


if st.session_state.show_detailed and "policy_json" in st.session_state:

    with st.spinner("Generating detailed report..."):

        report = run_analysis(st.session_state["policy_json"])

        st.markdown(report)


if st.session_state.show_detailed and "policy_json" in st.session_state:

    with st.spinner("Generating detailed report..."):

        report = run_analysis(st.session_state["policy_json"])

        st.markdown(report)
# Footer State
if "footer_section" not in st.session_state:
    st.session_state.footer_section = None

st.caption("Supports Health Insurance Policies")

st.markdown("---")

# ---------------------------
# HOW TO USE (Always Visible)
# ---------------------------
with st.expander("📘 How To Use"):

    st.info(
    "Upload your insurance policy PDF to get an AI-powered summary of coverage, exclusions, risks, and key highlights."
    )

    st.caption("""
1. Upload your policy document (PDF)  
2. AI analyzes coverage, exclusions and risks  
3. Basic Report : Key highlights and findings  
4. Generate detailed report 
""")

# Footer state
if "footer" not in st.session_state:
    st.session_state.footer = None

footer_col1, footer_col2 = st.columns([1,6])

with footer_col1:
    btn1, btn2 = st.columns([1,1])

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
