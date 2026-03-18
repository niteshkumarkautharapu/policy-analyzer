import os
import streamlit as st
import pdfplumber
import google.generativeai as genai
import json

# 🔑 Add your Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
st.write("API KEY VALUE:", os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-pro")

st.title("Insurance Policy Analyzer")

uploaded_file = st.file_uploader("Upload Policy PDF", type="pdf")


# ---------------------------
# Extract text from PDF
# ---------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ---------------------------
# Clean Gemini output
# ---------------------------
def clean_json(text):
    text = text.replace("```json", "").replace("```", "")
    return text.strip()


# ---------------------------
# Safe JSON parse
# ---------------------------
def validate_json(json_str):
    try:
        json_str = clean_json(json_str)
        data = json.loads(json_str)
        return True, data
    except:
        return False, None


# ---------------------------
# YOUR EXTRACTION PROMPT (UNCHANGED)
# ---------------------------
def run_extraction(text):
    prompt = f"""
You are an insurance policy data extraction engine.

You are given raw insurance policy text.

Your task is to extract structured data in strict JSON format.

---

CORE RULES:

* Output ONLY valid JSON
* Do NOT explain anything
* Do NOT add commentary
* Do NOT infer beyond what is explicitly mentioned
* If a field is missing → return "Not specified"
* Use exact numeric values where possible (no ranges unless given)
* Standardize currency in INR (numbers only, no symbols)
* Keep structure EXACTLY as defined below

---

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

---

EXTRACTION INSTRUCTIONS:

* policy_classification_hint:
  → If deductible exists → "High deductible / top-up behavior"
  → Else → "Standard policy"

* deductible_type:
  → Mention if "aggregate per year" or "per claim"

* copay:
  → Extract % if mentioned, else "0" if NIL

* room_rent_limit:
  → "No limit" if explicitly stated

* premium:
  → Total premium paid (final amount)

* members:
  → Extract all listed insured persons

* ambulance_limit:
  → Extract numeric value

* maternity:
  → "Covered" or "Not covered"

* opd:
  → "Covered" or "Not covered"

* waiting_periods:
  → Convert into numeric values only

* sublimits:
  → Extract only clearly defined limits

* special_conditions:
  → Include:
  * procedure-specific copay
  * caps (e.g., cochlear implant)
  * important financial clauses

---

VALIDATION:

Before output:

* Ensure JSON is valid
* No trailing commas
* All keys present
* No missing fields

---

OUTPUT:

Return ONLY JSON

---

INPUT:
{text}
"""

    response = model.generate_content(prompt)
    return response.text


# ---------------------------
# Retry wrapper (NO prompt change)
# ---------------------------
def extract_with_retry(text):
    output = run_extraction(text)

    is_valid, parsed = validate_json(output)

    if is_valid:
        return parsed

    # Retry SAME prompt again (no modification)
    output = run_extraction(text)

    is_valid, parsed = validate_json(output)

    if is_valid:
        return parsed

    return None


# ---------------------------
# YOUR ANALYSIS PROMPT (UNCHANGED)
# ---------------------------
def run_analysis(json_data):
    prompt = f"""
You are an insurance policy transparency expert.

You are given structured insurance policy data in JSON format.

Your task is NOT to rate, score, judge, recommend, or advise.

Your goal is to clearly explain:

How this policy behaves in real life

What the user will actually pay

When the policy helps and when it does not

What conditions affect payouts

CORE RULES:

Do NOT assign scores or ratings

Do NOT say "good", "bad", "best", "recommended", or similar

Do NOT suggest actions or strategies

Do NOT repeat raw JSON

Do NOT use technical insurance jargon

Translate policy clauses into real-life financial impact

If any data is missing → say "Not specified in policy"

OUTPUT FORMAT (STRICT MARKDOWN)

POLICY ANALYSIS REPORT

[Policy Name] — [Policy Number]

POLICY SUMMARY

Field	Details
Policy Name	
Insurer	
Policy Type	
Policy Classification	[Explain based on deductible behavior]
Sum Insured	
Deductible	
Co-Pay	
Room Rent Limit	
Premium Paid	
Policy Period	
Members Covered	

🔴 ACTIVATION BARRIER

Explain in ONE clear sentence

🔍 TOP 3 REALITY CHECKS

3 bullet points

💰 HOW THIS WORKS IN REAL LIFE

Table

⚠️ KEY LIMITATIONS & CONDITIONS

Explain

⚠️ HIDDEN COST LIMITS

Explain

⏳ WHEN YOU CANNOT USE THIS POLICY

Explain

✅ WHEN THIS POLICY HELPS

Explain

❌ WHEN THIS POLICY DOES NOT HELP

Explain

🧠 PLAIN ENGLISH SUMMARY

2–3 lines

🧾 PRACTICAL INTERPRETATION

Explain

INPUT JSON:
{json.dumps(json_data)}
"""

    response = model.generate_content(prompt)
    return response.text


# ---------------------------
# MAIN FLOW
# ---------------------------
if uploaded_file:
    st.write("Reading PDF...")
    text = extract_text(uploaded_file)

    st.write("Extracting structured data...")
    parsed_json = extract_with_retry(text)

    if not parsed_json:
        st.error("Extraction failed after retry")
    else:
        st.success("Extraction successful")

        st.subheader("Extracted Data")
        st.json(parsed_json)

        st.write("Analyzing policy...")
        report = run_analysis(parsed_json)

        st.subheader("Policy Analysis")
        st.markdown(report)
