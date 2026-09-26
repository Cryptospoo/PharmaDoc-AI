import streamlit as st
import ollama
import pdfplumber
import json
import pandas as pd

# Page Configuration
st.set_page_config(page_title="PharmaDoc AI - Indian Pharmacy", layout="wide")

st.title("💊 PharmaDoc AI – Local Pharmacy Document Wrapper")
st.write("Running locally with Ollama (Free & Private)")
st.write("Upload a prescription to get structured data, language summaries, and price estimates.")

# 1. Language Selection
language_options = {
    "English": "English",
    "Hindi": "Hindi (हिंदी)",
    "Kannada": "Kannada (ಕನ್ನಡ)",
    "Telugu": "Telugu (తెలుగు)",
    "Tamil": "Tamil (தமிழ்)",
    "Marathi": "Marathi (मराठी)",
    "Gujarati": "Gujarati (ગુજરાતી)",
    "Bengali": "Bengali (বাংলা)"
}

selected_lang = st.selectbox("Choose Summary Language", list(language_options.keys()), index=0)

# 2. File Uploader
uploaded_file = st.file_uploader(
    "Upload Prescription (PDF or Image)", 
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    file_type = uploaded_file.type
    text = ""

    # 3. Extract Text from PDF
    if file_type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    else:
        # Note: Image OCR requires additional libraries like EasyOCR. 
        # For now, we notify the user to use PDF for text extraction.
        st.info("Image uploaded. Note: Real OCR for handwritten images is best with PDF or specialized tools. Processing metadata...")
        text = "Image File Uploaded: " + uploaded_file.name

    # Show text preview
    if text.strip():
        st.subheader("Document Content (Preview)")
        st.text_area("Content", text[:2000], height=150, label_visibility="collapsed")
    else:
        st.warning("No text could be extracted. Please ensure the PDF is not just a scanned image.")

    # 4. Analyze Button
    if st.button("Analyze Document"):
        if not text.strip():
            st.error("Cannot analyze an empty document.")
        else:
            # The AI Prompt
            prompt = f"""
            You are an expert Indian pharmacy assistant. 
            Analyze the following text from a pharmacy document and provide:

            1. Extract this data in strict JSON format:
            {{
              "patient_name": "",
              "drug_name": "",
              "strength": "",
              "dosage": "",
              "quantity": "",
              "doctor_name": "",
              "date": ""
            }}

            2. Provide a short summary in {selected_lang}.
            
            3. Provide an approximate price estimate in Indian Rupees (INR) for these medicines 
               in the Indian market (mention both Generic and Brand prices).

            4. A quick compliance check for Indian Pharmacy rules.

            Document Text:
            {text}
            """

            try:
                with st.spinner(f"Ollama is thinking (Llama 3.1)..."):
                    # Call local Ollama
                    response = ollama.chat(
                        model="llama3.1:8b",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    full_response = response['message']['content']
                    
                    st.subheader("Results")
                    st.markdown(full_response)

                    # 5. Try to extract JSON for CSV download
                    try:
                        # Find the JSON part in the response
                        start_idx = full_response.find("{")
                        end_idx = full_response.rfind("}") + 1
                        json_str = full_response[start_idx:end_idx]
                        data = json.loads(json_str)
                        
                        # Show Table
                        st.subheader("Structured Data")
                        df = pd.DataFrame([data])
                        st.table(df)

                        # CSV Download Button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name="pharmacy_data.csv",
                            mime="text/csv"
                        )
                    except:
                        st.info("Note: AI did not format data as a table, but the text summary is above.")

            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}")
                st.info("Make sure you have run 'ollama pull llama3.1:8b' in your terminal.")

else:
    st.info("Please upload a file to begin.")
