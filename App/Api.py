import os
import re
import json
import random
import csv
import joblib
import numpy as np
import pandas as pd
import nltk
import uvicorn
import pymorphy3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ---------- Path configuration ----------
# This file is inside App/ ; Models/ and Resource/ are one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR = os.path.join(BASE_DIR, "Resource")
CSV_FILE = "lenta-ru-news.csv"
STATS_FILE = os.path.join(DATA_DIR, "stats_cache.json")
EXT = "pkl"
TFIDF_EXT = "prek"

# ---------- Text preprocessing ----------
nltk.download('stopwords', quiet=True)
RUSSIAN_STOP_WORDS = set(stopwords.words('russian'))
RUSSIAN_STOP_WORDS.update({
    'также', 'однако', 'который', 'это', 'собственный', 'сообщать', 'заявить',
    'ранее', 'свой', 'весь', 'мочь', 'стать', 'время', 'год', 'слово', 'новость',
    'отметить', 'рассказать', 'получить', 'являться', 'назвать', 'говорить',
    'частности', 'именно', 'поскольку', 'кроме', 'находиться', 'сообщается',
    'российский', 'россия', 'рф', 'москва'
})

morph = pymorphy3.MorphAnalyzer()
_cache = {}

def clean_and_lemmatize(text: str) -> str:
    if not isinstance(text, str):
        return ''
    words = re.findall(r'[а-яёa-z]+', text.lower())
    result = []
    for w in words:
        if w in RUSSIAN_STOP_WORDS or len(w) < 3:
            continue
        if w not in _cache:
            _cache[w] = morph.parse(w)[0].normal_form
        lemma = _cache[w]
        if lemma not in RUSSIAN_STOP_WORDS and len(lemma) >= 3:
            result.append(lemma)
    return ' '.join(result)

# ---------- Load models ----------
print("Loading label encoder...")
le = joblib.load(os.path.join(MODELS_DIR, f"label_encoder.{EXT}"))

print("Loading TF-IDF vectorizer...")
vec_path = os.path.join(MODELS_DIR, f"tfidf_vectorizer.{TFIDF_EXT}")
if not os.path.exists(vec_path):
    vec_path = os.path.join(MODELS_DIR, f"tfidf_vectorizer.{EXT}")
tfidf = joblib.load(vec_path)

model_files = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Naive Bayes": "model_naive_bayes.pkl",
    "Linear SVC": "model_linear_svc.pkl",
    "Final SVC (self-trained)": "final_svc_self_trained.pkl"
}

accuracy_scores = {
    "Logistic Regression": 0.8174,
    "Naive Bayes": 0.7800,
    "Linear SVC": 0.8352,
    "Final SVC (self-trained)": 0.8427,
}

models = {}
for name, fname in model_files.items():
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"Loaded {name}")

if not models:
    raise RuntimeError("No models loaded. Check MODELS_DIR and file names.")

# ---------- Statistics computation (with caching) ----------
def compute_statistics():
    """Reads CSV in chunks, computes stats, saves to JSON."""
    csv_path = os.path.join(DATA_DIR, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    total_articles = 0
    unique_topics = set()
    topic_counts = {}
    len_chars_list = []
    len_words_list = []
    date_min = None
    date_max = None

    chunk_iter = pd.read_csv(csv_path, usecols=['title', 'text', 'topic', 'date'],
                             chunksize=50000, low_memory=False)
    for chunk in chunk_iter:
        full_text = chunk['title'].fillna('') + ' ' + chunk['text'].fillna('')
        len_chars = full_text.str.len()
        len_words = full_text.str.split().str.len()
        len_chars_list.extend(len_chars.tolist())
        len_words_list.extend(len_words.tolist())

        for topic in chunk['topic'].dropna():
            unique_topics.add(topic)
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        dates = pd.to_datetime(chunk['date'], errors='coerce')
        if date_min is None:
            date_min = dates.min()
            date_max = dates.max()
        else:
            date_min = min(date_min, dates.min())
            date_max = max(date_max, dates.max())

        total_articles += len(chunk)

    avg_chars = int(np.mean(len_chars_list))
    avg_words = int(np.mean(len_words_list))
    min_chars = int(np.min(len_chars_list))
    max_chars = int(np.max(len_chars_list))
    median_chars = int(np.median(len_chars_list))
    min_words = int(np.min(len_words_list))
    max_words = int(np.max(len_words_list))
    median_words = int(np.median(len_words_list))

    stats = {
        "total_articles": total_articles,
        "unique_topics": len(unique_topics),
        "average_text_length_chars": avg_chars,
        "average_text_length_words": avg_words,
        "topic_distribution": topic_counts,
        "date_range": {
            "min": date_min.strftime('%Y-%m-%d') if date_min else "N/A",
            "max": date_max.strftime('%Y-%m-%d') if date_max else "N/A"
        },
        "min_length_chars": min_chars,
        "max_length_chars": max_chars,
        "median_length_chars": median_chars,
        "min_length_words": min_words,
        "max_length_words": max_words,
        "median_length_words": median_words
    }

    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats

if os.path.exists(STATS_FILE):
    print("Loading cached statistics...")
    with open(STATS_FILE, 'r', encoding='utf-8') as f:
        stats_cache = json.load(f)
    # Convert numeric values back to int (JSON loads them as int anyway)
    stats_cache["topic_distribution"] = {k: int(v) for k, v in stats_cache["topic_distribution"].items()}
    stats_cache["total_articles"] = int(stats_cache["total_articles"])
    stats_cache["unique_topics"] = int(stats_cache["unique_topics"])
else:
    print("Computing statistics (this may take a few minutes)...")
    stats_cache = compute_statistics()
    print("Statistics saved to JSON.")

# ---------- Random articles (robust) ----------
def get_random_articles(limit=50):
    """
    Returns a list of random articles from the CSV using csv.DictReader.
    Limits the text to 500 chars.
    """
    csv_path = os.path.join(DATA_DIR, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    # Count total lines (excluding header)
    with open(csv_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1
    if total_lines <= 0:
        return []

    limit = min(limit, total_lines)
    random_indices = set(random.sample(range(1, total_lines + 1), limit))

    articles = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Validate required columns
        required = {'title', 'text', 'topic'}
        if not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames)
            raise KeyError(f"CSV missing required columns: {missing}")

        for idx, row in enumerate(reader, start=1):
            if idx in random_indices:
                title = row['title']
                text = row['text']
                topic = row['topic']
                articles.append({
                    "title": title,
                    "text": text[:500] + "..." if len(text) > 500 else text,
                    "true_topic": topic
                })
                if len(articles) == limit:
                    break
    return articles

# ---------- FastAPI app ----------
app = FastAPI(title="News Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    model_name: str
    text: str

class PredictResponse(BaseModel):
    model_name: str
    predicted_category: str
    confidence: float
    all_probabilities: dict

@app.get("/models")
def get_models():
    return [{"name": name, "accuracy": accuracy_scores.get(name, 0.0)} for name in models.keys()]

@app.post("/predict")
def predict(request: PredictRequest):
    if request.model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = models[request.model_name]
    lemmatized = clean_and_lemmatize(request.text)
    if not lemmatized:
        raise HTTPException(status_code=400, detail="Text after preprocessing is empty")
    X = tfidf.transform([lemmatized])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        proba = np.zeros(len(le.classes_))
        proba[model.predict(X)[0]] = 1.0
    pred_idx = np.argmax(proba)
    pred_category = le.inverse_transform([pred_idx])[0]
    all_probs = {cat: float(proba[i]) for i, cat in enumerate(le.classes_)}
    return PredictResponse(
        model_name=request.model_name,
        predicted_category=pred_category,
        confidence=float(proba[pred_idx]),
        all_probabilities=all_probs
    )

@app.get("/stats/overview")
def overview():
    return stats_cache

@app.get("/sample_articles")
def sample_articles(limit: int = 50):
    try:
        articles = get_random_articles(min(limit, 100))
        return {"articles": articles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")

@app.get("/help")
def help_endpoint():
    return {
        "description": "API for Russian news topic classification",
        "endpoints": [
            "GET /models – list available models with accuracy",
            "POST /predict – classify a news article",
            "GET /stats/overview – dataset statistics",
            "GET /sample_articles – get random news samples",
            "GET /help – this help message"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("Api:app", host="0.0.0.0", port=8000, reload=True)