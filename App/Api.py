import re
import json
import joblib
import numpy as np
import pandas as pd
import uvicorn
import pymorphy3
import nltk
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from pathlib import Path
import random

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "Resource"
CSV_FILE = DATA_DIR / "lenta-ru-news.csv"
STATS_FILE = DATA_DIR / "stats_cache.json"
ARTICLES_FILE = DATA_DIR / "articles_cache.json"

models = {}
tfidf = None
le = None
stats_cache = {}
articles_cache = []  # все статьи в памяти, грузятся один раз

morph = pymorphy3.MorphAnalyzer()
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('russian'))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^а-яё ]', '', text.lower())
    return " ".join([
        morph.parse(w)[0].normal_form
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ])


def load_models():
    global tfidf, le, models
    try:
        le_path = MODELS_DIR / "label_encoder.pkl"
        if le_path.exists():
            le = joblib.load(le_path)

        v_path = MODELS_DIR / "tfidf_vectorizer.prek"
        if not v_path.exists():
            v_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if v_path.exists():
            tfidf = joblib.load(v_path)

        m_files = {
            "Logistic Regression": "model_logistic_regression.pkl",
            "Naive Bayes": "model_naive_bayes.pkl",
            "Linear SVC": "model_linear_svc.pkl",
            "Final SVC (self-trained)": "final_svc_self_trained.pkl",
        }
        for name, fname in m_files.items():
            p = MODELS_DIR / fname
            if p.exists():
                models[name] = joblib.load(p)
                print(f"✅ {name} loaded.")
    except Exception as e:
        print(f"❌ Load error: {e}")


def load_articles():
    """Грузит статьи один раз при старте и кэширует в JSON."""
    global articles_cache
    if ARTICLES_FILE.exists():
        print("📦 Loading articles from cache...")
        with open(ARTICLES_FILE, 'r', encoding='utf-8') as f:
            articles_cache = json.load(f)
        print(f"✅ {len(articles_cache)} articles loaded from cache.")
        return

    if not CSV_FILE.exists():
        print("❌ CSV not found")
        return

    print("⏳ Reading articles from CSV (first time only)...")
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False, on_bad_lines='skip')
        df = df[['title', 'text', 'topic']].dropna()
        # Берём 2000 случайных статей — этого хватит, не надо хранить всё
        sample = df.sample(n=min(2000, len(df)))
        articles_cache = sample.rename(columns={"topic": "true_topic"}).to_dict(orient="records")
        with open(ARTICLES_FILE, 'w', encoding='utf-8') as f:
            json.dump(articles_cache, f, ensure_ascii=False)
        print(f"✅ {len(articles_cache)} articles cached.")
    except Exception as e:
        print(f"❌ Articles load error: {e}")


def get_stats():
    global stats_cache
    if STATS_FILE.exists():
        print("📦 Loading stats from cache...")
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    if not CSV_FILE.exists():
        return {"error": "CSV missing"}

    print("⏳ Processing CSV for stats (first time only)...")
    df = pd.read_csv(CSV_FILE, usecols=['topic', 'date', 'text', 'title'], low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['wc'] = (df['title'].fillna('').str.len().div(5) + df['text'].fillna('').str.len().div(5)).astype(int)

    yearly = df['date'].dt.year.dropna().astype(int).value_counts().sort_index().to_dict()
    topic_dist = df['topic'].value_counts().to_dict()
    avg_len = df.groupby('topic')['wc'].mean().round(1).to_dict()
    monthly = (
        df.dropna(subset=['date'])
        .groupby(df['date'].dt.to_period('M').astype(str))
        .size()
        .to_dict()
    )

    stats_cache = {
        "total_articles": int(len(df)),
        "unique_topics": int(df['topic'].nunique()),
        "average_text_length_words": int(df['wc'].mean()),
        "topic_distribution": {str(k): int(v) for k, v in topic_dist.items()},
        "avg_length_per_topic": {str(k): float(v) for k, v in avg_len.items()},
        "yearly_distribution": {int(k): int(v) for k, v in yearly.items()},
        "monthly_distribution": {str(k): int(v) for k, v in monthly.items()},
        "word_count_raw": df['wc'].sample(n=min(2000, len(df))).tolist(),
    }

    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats_cache, f, ensure_ascii=False)

    print("✅ Stats cached.")
    return stats_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    global stats_cache
    stats_cache = get_stats()
    load_articles()  # грузим статьи в память при старте
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    model_name: str
    text: str


@app.get("/models")
def get_models():
    return [{"name": k} for k in models.keys()]


@app.get("/stats/overview")
def overview():
    return stats_cache


@app.get("/sample_articles")
def samples(limit: int = 50):
    if not articles_cache:
        return {"articles": []}
    n = min(limit, len(articles_cache))
    return {"articles": random.sample(articles_cache, n)}


@app.post("/predict")
def predict(req: PredictRequest):
    if not models:
        raise HTTPException(status_code=503, detail="Модели не загружены")
    if tfidf is None:
        raise HTTPException(status_code=503, detail="Векторайзер не загружен")
    if req.model_name not in models:
        raise HTTPException(status_code=404, detail=f"Модель '{req.model_name}' не найдена")

    cleaned = clean_text(req.text)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Текст пустой после очистки")

    vec = tfidf.transform([cleaned])
    model = models[req.model_name]
    pred_idx = model.predict(vec)[0]
    category = le.inverse_transform([pred_idx])[0] if le else str(pred_idx)

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = float(np.max(proba))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        shifted = scores - scores.min()
        total = shifted.sum()
        if total > 0:
            confidence = float(shifted.max() / total)

    all_probs = None
    if le is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        all_probs = {
            str(le.inverse_transform([i])[0]): round(float(p), 4)
            for i, p in enumerate(proba)
        }

    return {
        "category": category,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "model_used": req.model_name,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)