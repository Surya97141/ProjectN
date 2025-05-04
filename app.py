from transformers import pipeline
import gradio as gr

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Updated labels for more accuracy
labels = ["true", "false"]

# Detection function
def detect_truth(text):
    if not text.strip():
        return "⚠️ Please enter a news statement."
    
    result = classifier(text, labels)
    top_label = result["labels"][0]
    score = result["scores"][0]

    if top_label == "true":
        return f"✅ LIKELY TRUE ({score * 100:.2f}%)"
    else:
        return f"❌ LIKELY FALSE ({score * 100:.2f}%)"

# Sample test cases
examples = [
    ["Eating chocolate daily causes a 50% increase in IQ, new study reveals."],
    ["The Eiffel Tower is located in France."],
    ["The moon is made of cheese."],
    ["COVID-19 vaccines are effective in reducing severe illness."],
    ["Drinking bleach can cure viral infections."],
    ["Barack Obama served two terms as U.S. President."],
]

# Gradio UI
interface = gr.Interface(
    fn=detect_truth,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news headline or claim...", label="News Statement"),
    outputs=gr.Textbox(label="Prediction"),
    title="🔍 Fake News / Truth Detector (Zero-Shot)",
    description="Uses zero-shot classification (BART model) to estimate whether a claim is likely true or false. ⚠️ This is not a fact-checking engine — results may be inaccurate.",
    examples=examples
)

# Launch with shareable link
interface.launch(share=True)
