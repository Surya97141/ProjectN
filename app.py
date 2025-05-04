from transformers import pipeline
import gradio as gr

# Load the fake news detection model from Hugging Face
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Define prediction function
def detect_fake_news(text):
    if not text.strip():
        return "Please enter a news statement."
    result = classifier(text)[0]
    label = result['label'].lower()
    score = round(result['score'] * 100, 2)
    if "fake" in label:
        return f"🟥 FAKE NEWS ({score}%)"
    elif "real" in label:
        return f"🟩 REAL NEWS ({score}%)"
    else:
        return f"⚠️ Unknown label: {label} ({score}%)"

# Gradio UI
iface = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter a news statement here...", label="News Statement"),
    outputs=gr.Text(label="Prediction"),
    title="📰 Fake News Detector",
    description="Enter a news headline or claim to check whether it's likely FAKE or REAL using a Hugging Face model.",
)

# Launch the app with public sharing
iface.launch(share=True)
