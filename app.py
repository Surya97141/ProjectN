# 📌 Import necessary libraries
from transformers import pipeline
import gradio as gr
import torch  # For GPU support

# 🚀 Load the zero-shot classification model (BART)
try:
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    news_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
except Exception as e:
    raise RuntimeError(f"⚠️ Model failed to load: {str(e)}")

# 📊 Define classification labels
classification_labels = ["true", "false"]

# 🔍 Function for fake news detection
def classify_news(statement):
    """Processes the news statement and returns a classification result."""
    if not statement.strip():
        return "⚠️ Please enter a valid news statement."
    try:
        prediction = news_classifier(statement, classification_labels)
        predicted_label = prediction["labels"][0]  
        confidence_score = prediction["scores"][0]  
    except Exception as e:
        return f"⚠️ Classification Error: {str(e)}"
    
    # 🚨 Apply confidence threshold corrections
    if confidence_score < 0.75:
        return f"⚠️ UNCERTAIN ({confidence_score * 100:.2f}%) - Please verify!"
    elif predicted_label == "true":
        return f"✅ LIKELY TRUE ({confidence_score * 100:.2f}%)"
    else:
        return f"❌ LIKELY FALSE ({confidence_score * 100:.2f}%)"

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
        ["Eating chocolate daily increases IQ by 50%."],
        ["The Eiffel Tower is in France."],
        ["COVID-19 vaccines reduce severe illness."],
        ["Drinking bleach cures infections."],
    ],
    allow_flagging="never",
    theme="default"
)

# 🌐 Gradio API Endpoint for Extension
api_interface = gr.Interface(
    fn=api_classify_news,
    inputs=gr.Textbox(label="Enter News Statement"),
    outputs="json",
)

# 🚀 Launch Both Web UI & API
if __name__ == "__main__":
    gr.TabbedInterface([news_checker, api_interface], ["News Detector", "API"]).launch(share=True)