import streamlit as st
from transformers import pipeline

st.title("AI Content Detector")
st.write("Check whether text is AI-generated or human-written.")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ahmediqbal/ai-text-detector-model")

classifier = load_model()

text = st.text_area("Enter text here to analyze:")
if st.button("Analyze Text"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = classifier(text)[0]
            label = result['label']
            score = result['score'] * 100
            st.success(f"Result: {label} ({score:.2f}% confidence)")
    else:
        st.warning("Please enter some text first.")
