import streamlit as st
from groq import Groq
import pdfplumber
import json
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="PharmaDoc AI - Indian Pharmacy", layout="wide")
st.title("💊 PharmaDoc AI – Indian Pharmacy Document Wrapper")
st.write("Upload prescription or compounding record (PDF or Image) → Get structured data, simple summary in your language & approx. Indian price")

# Language options
language_options = {
    "English": "English",
    "Hindi": "Hindi (हिंदी)",
    "Kannada": "Kannada (ಕನ್ನಡ)",
    "Telugu": "Telugu (తెಲుగు)",
    "Tamil": "Tamil (தமிழ்)",
    "Marathi": "Marathi (मराठी)",
    "Gujarati": "Gujarati (ગુજરાતી)",
    "Bengali": "Bengali (বাংলা)"
}

selected_lang = st.selectbox("Choose Summary Language", list(language_options.keys()), index=0)

# Accept PDF + all common image formats
uploaded_file = st.file_uploader(
    "Upload PDF or Image (PNG, JPG, JPEG, GIF, BMP, WEBP)",
    type=["pdf", "png", "jpg", "jpeg", "gif", "bmp", "webp"]
)

if uploaded_file:
    file_type = uploaded_file.type

    # ===================== PDF Handling =====================
    if file_type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])

    # ===================== Image Handling =====================
    else:
        st.info("Image uploaded successfully. OCR (text extraction from image) will be added soon. For now, please use PDF.")
        text = ""

    st.subheader("Extracted Text (preview)")
    st.text_area("", text[:2000], height=150)

    if st.button("Analyze with AI"):
        prompt = f"""You are an expert Indian pharmacy assistant.
Extract the following from the document in strict JSON format:
{{
  "patient_name": "",
  "patient_dob": "",
  "drug_name": "",
  "strength": "",
  "dosage_form": "",
  "quantity": "",
  "directions": "",
  "prescriber_name": "",
  "prescriber_reg_no": "",
  "date_written": "",
  "refills": "",
  "notes": ""
}}

Then provide:
1. A short plain-English summary
2. The same summary translated into {selected_lang}
3. Approximate Indian pharmacy price estimate (generic vs brand, in INR)
4. Basic compliance checklist (Indian CDSCO / state pharmacy council rules)

Document text:
{text[:12000]}"""

        with st.spinner(f"Analyzing with Llama 3.1 in {selected_lang}..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-specdec",     # ← Latest working Groq model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )

        result = response.choices[0].message.content
        st.subheader("AI Output")

        # Try to parse JSON
        try:
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            data = json.loads(result[json_start:json_end])
            st.success("Structured Data Extracted")
            st.json(data)

            # CSV export
            df = pd.DataFrame([data])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download as CSV", csv, "prescription_data.csv", "text/csv")
        except:
            st.info("Could not parse JSON automatically – see full output below")

        st.markdown(result)
