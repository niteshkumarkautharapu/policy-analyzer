import json
import pdfplumber
import gradio as gr
from google import genai

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "gemini-2.0-flash"  # change to 2.5 if your key supports it

# IMPORTANT: for local testing, replace with your key
# For deployment (HF Spaces), we will use environment variables
import os
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file.name) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ---------------------------
# CLEAN + VALIDATE JSON
# ---------------------------
def clean_json(text):
    return text.replace("```json", "").replace("```", "").strip()


def validate_json(json_str):
    try:
        json_str = clean_json(json_str)
        return json.loads(json_str)
    except:
        return None


# ---------------------------
# GEMINI CALL
# ---------------------------
def call_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text


# ---------------------------
# EXTRACTION PROMPT
# ---------------------------
def run_extraction(text):
    prompt = f"""
You are an insurance policy data extraction engine.

Return ONLY valid JSON.

INPUT:
{text}
"""
    return call_gemini(prompt)


def extract_with_retry(text):
    for _ in range(2):
        output = run_extraction(text)
        parsed = validate_json(output)
        if parsed:
            return parsed
    return None


# ---------------------------
# ANALYSIS PROMPT
# ---------------------------
def run_analysis(json_data):
    prompt = f"""
Explain this insurance policy in plain English.

INPUT JSON:
{json.dumps(json_data)}
"""
    return call_gemini(prompt)


# ---------------------------
# MAIN FUNCTION (Gradio)
# ---------------------------
def analyze_policy(file):
    if file is None:
        return "Please upload a file", None

    try:
        text = extract_text(file)

        parsed_json = extract_with_retry(text)

        if not parsed_json:
            return "Extraction failed", None

        report = run_analysis(parsed_json)

        return report, parsed_json

    except Exception as e:
        return f"Error: {str(e)}", None


# ---------------------------
# GRADIO UI
# ---------------------------
with gr.Blocks(title="Policy Analyzer") as app:

    gr.Markdown("# 🛡️ Insurance Policy Analyzer")

    with gr.Row():
        file_input = gr.File(label="Upload Policy PDF")

    analyze_btn = gr.Button("Analyze Policy")

    output_text = gr.Markdown(label="Policy Analysis")
    output_json = gr.JSON(label="Extracted Data")

    analyze_btn.click(
        fn=analyze_policy,
        inputs=file_input,
        outputs=[output_text, output_json],
    )


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.launch()
