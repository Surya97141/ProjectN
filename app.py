# 📌 Import necessary libraries
from transformers import pipeline
import gradio as gr

# 🚀 Load the zero-shot classification model (BART)
try:
    news_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    raise RuntimeError(f"⚠️ Model failed to load: {str(e)}")

# 📊 Define classification labels
classification_labels = ["true", "false"]

# 🔍 Function for fake news detection
def classify_news(statement):
    # ⛔ Handle empty input
    if not statement.strip():
        return "⚠️ Please enter a valid news statement."
    
    # 🏗️ Run classification & handle errors
    try:
        prediction = news_classifier(statement, classification_labels)
    except Exception as e:
        return f"⚠️ Classification Error: {str(e)}"

    # 📊 Extract top prediction
    predicted_label = prediction["labels"][0]  
    confidence_score = prediction["scores"][0]  

    # ✅ Format response based on prediction
    if predicted_label == "true":
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
    fn=classify_news,  # Function to process input
    inputs=gr.Textbox(lines=3, placeholder="Enter a news claim...", label="News Statement"),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector",
    description="Uses a zero-shot classification model (BART) to assess the likelihood of truthfulness.\n⚠️ This is not a certified fact-checking tool and should be used with caution!",
    examples=test_statements
)

# 🌐 Launch the interactive app
news_checker.launch(share=True)