import re
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="VIP AI Content Detector", page_icon="🔍", layout="wide"
)

st.title("🔍 VIP AI Content Detector (Sentence-Level Analysis)")
st.write(
    "Analyze your text sentence-by-sentence to see exactly which parts are"
    " flagged as AI-generated or human-written."
)


@st.cache_resource
def load_model():
  return pipeline(
      "text-classification", model="ahmediqbal/ai-text-detector-model"
  )


with st.spinner("Loading AI model into memory... Please wait a moment."):
  classifier = load_model()

text = st.text_area(
    "Enter your statement or essay here:",
    height=250,
    placeholder="Paste your essay here to check it sentence by sentence...",
)

if st.button("Run Deep Sentence Scan", type="primary"):
  if text.strip():
    # Split text into sentences using regex punctuation matching
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if not sentences:
      st.warning("Please enter valid sentences to analyze.")
    else:
      st.subheader("Detailed Sentence Breakdown")
      ai_sentences_count = 0

      progress_bar = st.progress(0)
      total_sentences = len(sentences)

      for i, sentence in enumerate(sentences):
        # Run classification on each individual sentence
        result = classifier(sentence)[0]
        label = result["label"]
        score = result["score"] * 100

        # Update progress bar
        progress_bar.progress((i + 1) / total_sentences)

        # Display color-coded results per sentence
        if "AI" in label.upper() and score > 50:
          ai_sentences_count += 1
          st.markdown(
              f"🔴 **[AI Flagged - {score:.1f}% confidence]** {sentence}"
          )
        else:
          st.markdown(
              f"🟢 **[Human-like - {100-score:.1f}% confidence]** {sentence}"
          )

      progress_bar.empty()
      st.divider()
      st.info(
          f"Scan complete! Analyzed {total_sentences} sentences. Flagged"
          f" {ai_sentences_count} sentence(s) as high probability AI."
      )
  else:
    st.warning("Please enter some text before clicking scan.")
