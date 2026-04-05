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

if "detailed_report" not in st.session_state:
    st.session_state.detailed_report = None

if "highlights" not in st.session_state:
    st.session_state.highlights = None

if "summary" not in st.session_state:
    st.session_state.summary = None
    
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "menu" not in st.session_state:
    st.session_state.menu = None
    
# ---------------------------
# CONFIG
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Models
EXTRACTION_MODEL = "gpt-4o-mini"
ANALYSIS_MODEL = "gpt-5-mini"

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
def call_gpt(prompt, model):

    response = client.responses.create(
        model=model,
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

    return call_gpt(prompt, ANALYSIS_MODEL)


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

    return call_gpt(prompt, ANALYSIS_MODEL)

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.set_page_config(
    page_title="Check Your Policy",
    page_icon="🛡️",
    layout="wide"
)

# CSS for Clear Button inside Upload Bar
st.markdown("""
<style>
.clear-upload {
    position: relative;
}

.clear-upload button {
    position: absolute;
    right: 10px;
    top: -38px;
}
</style>
""", unsafe_allow_html=True)


title_col, nav_col = st.columns([3,2])

with title_col:
    st.title("🛡️ Check Your Policy")
    st.caption("Understand your insurance policy coverage, risks and limitations instantly.")
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

# ---------------------------
# Upload Section
# ---------------------------

st.markdown("### Upload your policy")

upload_container = st.container()

with upload_container:

    col1, col2, spacer = st.columns([2,1,3])

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

            st.session_state.pop("policy_json", None)
            st.session_state.pop("highlights", None)
            st.session_state.pop("summary", None)
            st.session_state.pop("last_uploaded", None)

            st.rerun()

# ---------------------------
# Detect File Upload
# ---------------------------

if uploaded_file is not None:
    st.session_state.file_uploaded = True

if uploaded_file is not None and "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = uploaded_file.name

elif uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded:
    st.session_state.show_basic = False
    st.session_state.show_detailed = False
    st.session_state.detailed_report = None
    st.session_state.pop("policy_json", None)
    st.session_state.pop("highlights", None)
    st.session_state.pop("summary", None)
    st.session_state.last_uploaded = uploaded_file.name


# ---------------------------
# Detect File Removal
# ---------------------------

if uploaded_file is None:
    st.session_state.show_basic = False
    st.session_state.show_detailed = False
    st.session_state.file_uploaded = False
    st.session_state.detailed_report = None
    st.session_state.pop("policy_json", None)
    st.session_state.pop("highlights", None)
    st.session_state.pop("summary", None)

# Detect File Upload
if uploaded_file is not None:
    st.session_state.file_uploaded = True
if uploaded_file is not None and "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = uploaded_file.name

elif uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded:
    st.session_state.show_basic = False
    st.session_state.show_detailed = False
    st.session_state.detailed_report = None
    st.session_state.pop("policy_json", None)
    st.session_state.last_uploaded = uploaded_file.name

# Detect File Removal
if uploaded_file is None:
    st.session_state.show_basic = False
    st.session_state.show_detailed = False
    st.session_state.file_uploaded = False
    st.session_state.detailed_report = None
    st.session_state.pop("policy_json", None)

if uploaded_file:

    st.success("✅ Policy uploaded successfully! Click **Basic Summary** to analyse your policy.")

    if st.button("Basic Summary"):
        st.session_state.show_basic = True

if st.session_state.show_basic and uploaded_file:

    if "policy_json" not in st.session_state:

        import time

        progress = st.progress(0)
        status = st.empty()

        status.info("📄 Reading policy document...")
        progress.progress(10)

        text = extract_text(uploaded_file)
        time.sleep(0.2)

        status.info("🔍 Extracting policy details, might take few seconds...")
        progress.progress(35)

        parsed_json = extract_with_retry(text)

        if not parsed_json:
            status.error("Extraction failed")
            progress.empty()
            st.stop()

        st.session_state["policy_json"] = parsed_json
        time.sleep(0.2)

        status.info("🧠 Generating policy summary, Please wait, This may take few seconds...")
        progress.progress(60)

        highlights = generate_highlights(parsed_json)
        summary = generate_basic_summary(parsed_json)

        st.session_state["highlights"] = highlights
        st.session_state["summary"] = summary

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
    highlights = st.session_state["highlights"]
    summary = st.session_state["summary"]

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


if st.session_state.show_detailed and "policy_json" in st.session_state:

    if st.session_state.detailed_report is None:

        import time

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

        status.info("📄 Generating detailed report, Please wait, This may take few seconds...")
        progress.progress(85)

        detailed = run_analysis(
            st.session_state["policy_json"]
        )

        st.session_state.detailed_report = detailed

        progress.progress(100)
        status.success("✅ Detailed report ready")

        time.sleep(0.6)

        progress.empty()
        status.empty()

    st.markdown(st.session_state.detailed_report)
    
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
