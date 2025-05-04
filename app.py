from transformers import pipeline
import gradio as gr

# Load model
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Label decoding
label_map = {
    "LABEL_0": "REAL",
    "LABEL_1": "FAKE"
}

# Prediction function
def detect_fake_news(text):
    if not text.strip():
        return "Please enter a news statement."
    result = classifier(text)[0]
    label = result['label']
    prediction = label_map.get(label, "UNKNOWN")
    score = round(result['score'] * 100, 2)

    if prediction == "FAKE":
        return f"🟥 FAKE NEWS ({score}%)"
    elif prediction == "REAL":
        return f"🟩 REAL NEWS ({score}%)"
    else:
        return f"⚠️ Unknown label: {label} ({score}%)"

# Gradio UI
iface = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter a news statement here...", label="News Statement"),
    outputs=gr.Text(label="Prediction"),
    title="📰 Fake News Detector",
    description="Enter a news headline or statement to check whether it's likely FAKE or REAL using a Hugging Face BERT model.",
)

# Run the app with public link
iface.launch(share=True)
