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

    # ✅ Format response based on prediction
    return (f"✅ LIKELY TRUE ({confidence_score * 100:.2f}%)"
            if predicted_label == "true" 
            else f"❌ LIKELY FALSE ({confidence_score * 100:.2f}%)")

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
    title="📰 Fake News Detector",
    description="Uses a zero-shot classification model (BART) to assess the likelihood of truthfulness.\n"
                "⚠️ This is not a certified fact-checking tool and should be used with caution!",
    examples=test_statements,
    allow_flagging="never",  # Removes unnecessary flagging button
    theme="default"  # Ensures clean UI styling
)

# 🌐 Launch the interactive app
news_checker.launch(share=True)