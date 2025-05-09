import requests
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import gradio as gr
import torch

# Initialize the model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Zero-shot classification model
device = 0 if torch.cuda.is_available() else -1
news_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=device)

# NewsAPI configuration
NEWS_API_KEY = 'YOUR_NEWSAPI_KEY'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

# Classification labels for zero-shot
classification_labels = ['factual', 'misleading', 'harmful']

dangerous_keywords = ['bleach', 'injection', 'poison', 'toxic', 'acid', 'consume', 'disinfectant']

# Predefined scientific facts and misleading claims
static_facts = {
    'the earth is round': '✅ FACTUAL - This is a universally accepted scientific fact.',
    'the earth revolves around the sun': '✅ FACTUAL - This is a universally accepted scientific fact.',
    'vaccines prevent diseases': '✅ FACTUAL - This is a scientifically proven fact.',
    'water boils at 100 degrees celsius': '✅ FACTUAL - Standard atmospheric pressure.',
    'humans need oxygen to survive': '✅ FACTUAL - Basic biological principle.',
    'the earth is flat': '❌ MISLEADING - This is a scientifically disproven claim.'
}

def fetch_real_time_news(query):
    params = {
        'q': query,
        'language': 'en',
        'apiKey': NEWS_API_KEY,
        'pageSize': 5,
        'sortBy': 'relevancy'
    }
    response = requests.get(NEWS_API_URL, params=params)
    articles = response.json().get('articles', [])
    return [(article['title'], article['description'], article['url']) for article in articles]

def semantic_similarity(query, articles):
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = []
    for title, description, url in articles:
        combined_text = f'{title}. {description}'
        article_embedding = model.encode(combined_text, convert_to_tensor=True)
        similarity = util.cos_sim(query_embedding, article_embedding).item()
        results.append((title, description, url, similarity))
    results.sort(key=lambda x: x[3], reverse=True)
    return results

def verify_news(query):
    articles = fetch_real_time_news(query)
    if not articles:
        return 'No articles found', []

    results = semantic_similarity(query, articles)
    top_result = results[0] if results else None
    status = 'Verified ✅' if top_result and top_result[3] > 0.75 else 'Needs Verification ⚠️'
    response = f'Status: {status}\n\nTop Matches:\n'
    for title, desc, url, score in results[:3]:
        response += f'Title: {title}\nDescription: {desc}\nURL: {url}\nSimilarity: {score:.2f}\n\n'
    return response, results

def classify_news(statement):
    statement_lower = statement.strip().lower()
    if not statement_lower:
        return '⚠️ Please enter a valid news statement.'
    if any(keyword in statement_lower for keyword in dangerous_keywords):
        return '❌ HARMFUL - This statement is potentially dangerous!'
    if statement_lower in static_facts:
        return static_facts[statement_lower]

    prediction = news_classifier(statement, classification_labels)
    predicted_label = prediction['labels'][0]
    confidence_score = prediction['scores'][0]

    if confidence_score < 0.60:
        return f'⚠️ UNCERTAIN ({confidence_score * 100:.2f}%) - Verify with trusted sources!'
    elif predicted_label.lower() == 'factual':
        return f'✅ FACTUAL ({confidence_score * 100:.2f}%)'
    elif predicted_label.lower() == 'harmful':
        return f'❌ HARMFUL ({confidence_score * 100:.2f}%)'
    else:
        return f'❌ MISLEADING ({confidence_score * 100:.2f}%)'

# Gradio Interface for Real-Time Verification
real_time_checker = gr.Interface(
    fn=verify_news,
    inputs=gr.Textbox(lines=2, placeholder='Enter a news headline or statement...', label='News Statement'),
    outputs=['text', 'json'],
    title='📰 Real-Time News Verifier',
    description='Checks the validity of news statements against real-time data fetched from NewsAPI.'
)

# Gradio Interface for Model-Based Classification
model_based_checker = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=2, placeholder='Enter a news headline or statement...', label='News Statement'),
    outputs='text',
    title='🧠 AI Model News Classifier',
    description='Classifies news as factual, misleading, or harmful using a BART-based zero-shot classifier.'
)

# Combine both interfaces into tabs
gr.TabbedInterface([real_time_checker, model_based_checker], ['Real-Time Verification', 'Model-Based Classification']).launch()
