import os
import streamlit as st
import pdfplumber
import json
from google import genai

# ---------------------------
# CONFIGURE CLIENT
# ---------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.0-flash"   # you can try 2.5 later if enabled


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
    return text.replace("```json", "").replace("```", "").strip()


# ---------------------------
# Validate JSON
# ---------------------------
def validate_json(json_str):
    try:
        json_str = clean_json(json_str)
        data = json.loads(json_str)
        return True, data
    except Exception:
        return False, None


# ---------------------------
# Gemini call wrapper
# ---------------------------
def call_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


# ---------------------------
# EXTRACTION
# ---------------------------
def run_extraction(text):
    prompt = f"""<PASTE YOUR SAME EXTRACTION PROMPT HERE>
INPUT:
{text}
"""
    return call_gemini(prompt)


# ---------------------------
# RETRY
# ---------------------------
def extract_with_retry(text):
    for _ in range(2):
        output = run_extraction(text)
        is_valid, parsed = validate_json(output)
        if is_valid:
            return parsed
    return None


# ---------------------------
# ANALYSIS
# ---------------------------
def run_analysis(json_data):
    prompt = f"""<PASTE YOUR SAME ANALYSIS PROMPT HERE>

INPUT JSON:
{json.dumps(json_data)}
"""
    return call_gemini(prompt)


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


# ---------------------------
# DEBUG BUTTON
# ---------------------------
if st.button("Test Gemini 2.x"):
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents="Say hello in one line",
        )
        st.success(res.text)
    except Exception as e:
        st.error(str(e))
