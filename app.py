#  Import necessary libraries
from transformers import pipeline
import gradio as gr
import torch  # For GPU support

#  Load the fact-checking model
try:
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    news_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"⚠️ Model failed to load: {str(e)}")

#  Define classification labels (check model outputs)
classification_labels = ["factual", "misleading", "harmful"]

#  List of dangerous keywords for health claims
dangerous_keywords = [
    "bleach", "injection", "poison", "toxic", "acid", "consume", "disinfectant"
]

#  List of universally accepted scientific facts and known misleading claims
static_facts = {
    "the earth is round": "✅ FACTUAL - This is a universally accepted scientific fact.",
    "the earth revolves around the sun": "✅ FACTUAL - This is a universally accepted scientific fact.",
    "vaccines prevent diseases": "✅ FACTUAL - This is a scientifically proven fact.",
    "water boils at 100 degrees celsius": "✅ FACTUAL - Standard atmospheric pressure.",
    "water boils at 100 degrees centigrade": "✅ FACTUAL - Standard atmospheric pressure.",
    "water boils at 100 degrees": "✅ FACTUAL - Standard atmospheric pressure.",
    "humans need oxygen to survive": "✅ FACTUAL - Basic biological principle.",
    "the earth is flat": "❌ MISLEADING - This is a scientifically disproven claim.",
    "earth is flat": "❌ MISLEADING - This is a scientifically disproven claim."
}

#  Improved function for fake news detection
def classify_news(statement):
    """Processes news statements and ensures better classification accuracy."""
    
    statement_lower = statement.strip().lower()
    if not statement_lower or len(statement_lower) < 5:
        return "⚠️ Please enter a valid news statement (at least 5 characters)."

    #  Keyword-based quick detection for harmful content
    if any(keyword in statement_lower for keyword in dangerous_keywords):
        return "❌ HARMFUL - This statement is potentially dangerous!"

    #  Static fact-checking for well-known scientific facts and false claims
    if statement_lower in static_facts:
        return static_facts[statement_lower]

    try:
        # Run classification
        prediction = news_classifier(statement, classification_labels)

        # Edge case: If labels are not returned, handle it
        if not prediction["labels"]:
            return "⚠️ Classification Error: No labels returned. Please try again."

        # Extract the best prediction
        predicted_label = prediction["labels"][0]
        confidence_score = prediction["scores"][0]

    except Exception as e:
        return f"⚠️ Classification Error: {str(e)}"

    #  Improved confidence threshold handling
    if confidence_score < 0.60:  # Higher confidence threshold
        return f"⚠️ UNCERTAIN ({confidence_score * 100:.2f}%) - Verify with trusted sources!"
    elif predicted_label.lower() == "factual":
        return f"✅ FACTUAL ({confidence_score * 100:.2f}%)"
    elif predicted_label.lower() == "harmful":
        return f"❌ HARMFUL ({confidence_score * 100:.2f}%)"
    else:
        return f"❌ MISLEADING ({confidence_score * 100:.2f}%)"

#  Gradio API for Browser Extension
def api_classify_news(statement):
    """Handles API requests via Gradio."""
    return {"fake_news": classify_news(statement)}

#  Gradio Interface for Manual Testing & API
news_checker = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news claim...", label="News Statement"),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detector (Enhanced Safety Check)",
    description="Uses a zero-shot classification model (BART) to estimate truthfulness and detect harmful content.\n⚠️ Always verify claims with trusted sources!",
    examples=[
        ["The Eiffel Tower is in France."],
        ["Eating chocolate daily increases IQ by 50%."],
        ["COVID-19 vaccines reduce severe illness."],
        ["Drinking bleach cures infections."],
        ["The Earth is flat"],
        ["Water boils at 100 degrees Celsius"],
        ["Water boils at 100 degrees Centigrade"],
        ["The Earth is Round"]
    ],
    allow_flagging="never"
)

#  Gradio API Endpoint for Browser Extension
api_interface = gr.Interface(
    fn=api_classify_news,
    inputs=gr.Textbox(label="Enter News Statement"),
    outputs="json",
)

#  Launch Web UI & API
gr.TabbedInterface([news_checker, api_interface], ["News Detector", "API"]).launch()
