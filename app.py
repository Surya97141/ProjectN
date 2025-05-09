# 📌 Import necessary libraries
from transformers import pipeline
import gradio as gr
import torch  # For GPU support

# 🚀 Load the fact-checking model
try:
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    news_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
except Exception as e:
    raise RuntimeError(f"⚠️ Model failed to load: {str(e)}")

# 📊 Updated classification labels (better context)
classification_labels = ["factual", "misleading"]

# 🔍 Improved function for fake news detection
def classify_news(statement):
    """Processes news statements and ensures better classification accuracy."""
    if not statement.strip():
        return "⚠️ Please enter a valid news statement."
    try:
        prediction = news_classifier(statement, classification_labels)
        predicted_label = prediction["labels"][0]
        confidence_score = prediction["scores"][0]
    except Exception as e:
        return f"⚠️ Classification Error: {str(e)}"
    
    # 🚨 Threshold-based confidence filtering
    if confidence_score < 0.85:  # More strict accuracy requirement
        return f"⚠️ UNCERTAIN ({confidence_score * 100:.2f}%) - Verify with trusted sources!"
    elif predicted_label == "factual":
        return f"✅ FACTUAL ({confidence_score * 100:.2f}%)"
    else:
        return f"❌ MISLEADING ({confidence_score * 100:.2f}%)"

# 🌐 Gradio API for Browser Extension
def api_classify_news(statement):
    """Handles API requests via Gradio."""
    return {"fake_news": classify_news(statement)}

# 🎨 Gradio Interface for Manual Testing & API
news_checker = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news claim...", label="News Statement"),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector (Improved Accuracy)",
    description="Uses a zero-shot classification model (BART) to estimate truthfulness.\n"
                "⚠️ Always verify claims with trusted sources!",
    examples=[
        ["The Eiffel Tower is in France."],  # Should now return "✅ FACTUAL"
        ["Eating chocolate daily increases IQ by 50%."],
        ["COVID-19 vaccines reduce severe illness."],
        ["Drinking bleach cures infections."],
    ],
    allow_flagging="never",
    theme="default"
)

# 🌐 Gradio API Endpoint for Browser Extension
api_interface = gr.Interface(
    fn=api_classify_news,
    inputs=gr.Textbox(label="Enter News Statement"),
    outputs="json",
)

# 🚀 Launch Web UI & API
if __name__ == "__main__":
    gr.TabbedInterface([news_checker, api_interface], ["News Detector", "API"]).launch(share=True)