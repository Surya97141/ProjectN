from transformers import pipeline
import gradio as gr

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define prediction function
def detect_fake_news(text):
    if not text.strip():
        return "⚠️ Please enter a news statement."
    
    labels = ["real news", "fake news"]
    result = classifier(text, labels)
    top_label = result["labels"][0]
    score = result["scores"][0]

    # Format output
    if top_label == "real news":
        return f"✅ REAL NEWS ({score * 100:.2f}%)"
    else:
        return f"❌ FAKE NEWS ({score * 100:.2f}%)"

# Build the Gradio interface
interface = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter a news headline or article...", label="News Statement"),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detection with Zero-Shot Learning",
    description="This demo uses a zero-shot classification model (facebook/bart-large-mnli) to determine whether a news statement is likely real or fake. Note: Accuracy is not guaranteed. Use for educational/demo purposes only.",
)

# Launch with public shareable link
interface.launch(share=True)
