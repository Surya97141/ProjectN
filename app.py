import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_news(text):
    result = classifier(text)[0]
    label = result['label']
    return "REAL" if label == "POSITIVE" else "FAKE"

gr.Interface(fn=classify_news, inputs="text", outputs="text").launch()
