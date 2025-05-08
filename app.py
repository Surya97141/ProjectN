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

# 🔍 Function for fake news detection with threshold handling
def classify_news(statement):
    """Processes the news statement and returns a classification result with improved accuracy."""
    
    # ⛔ Handle empty input
    if not statement.strip():
        return "⚠️ Please enter a valid news statement."
    
    # 🏗️ Run classification & handle errors
    try:
        prediction = news_classifier(statement, classification_labels)
        predicted_label = prediction["labels"][0]  
        confidence_score = prediction["scores"][0]  
    except Exception as e:
        return f"⚠️ Classification Error: {str(e)}"

    # 🚨 Apply confidence threshold corrections
    if confidence_score < 0.75:  # Uncertain predictions
        return f"⚠️ UNCERTAIN ({confidence_score * 100:.2f}%) - Please verify with trusted sources!"
    elif predicted_label == "true":
        return f"✅ LIKELY TRUE ({confidence_score * 100:.2f}%)"
    else:
        return f"❌ LIKELY FALSE ({confidence_score * 100:.2f}%)"

# 📚 Example news claims for testing
test_statements = [
    ["Eating chocolate daily increases IQ by 50%."],
    ["The Eiffel Tower is in France."],
    ["The moon is made of cheese."],
    ["COVID-19 vaccines reduce severe illness."],
    ["Drinking bleach cures infections."],
    ["Barack Obama served two terms as U.S. President."],
]

# 🎨 Create the Gradio interface
news_checker = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news claim...", label="News Statement"),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector (Improved Accuracy)",
    description="Uses a zero-shot classification model (BART) to estimate truthfulness.\n"
                "⚠️ Always verify claims with trusted sources—AI models can make errors!",
    examples=test_statements,
    allow_flagging="never",
    theme="default"
)

# 🌐 Launch the interactive app
news_checker.launch(share=True)