# 📌 Import necessary libraries
from transformers import pipeline
import gradio as gr
import torch  # For GPU support
from flask import Flask, request, jsonify

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

# 🌐 Flask API for Browser Extension
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_fake_news():
    """Handles API requests from the browser extension."""
    data = request.json
    result = classify_news(data['text'])
    return jsonify({"fake_news": result})

# 🎨 Gradio Interface for Manual Testing
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

if __name__ == "__main__":
    # Launch both Gradio interface & Flask API
    news_checker.launch(share=True)
    app.run(host="0.0.0.0", port=8000)