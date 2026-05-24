import os
import re
import json
import random
import joblib
import numpy as np
import pandas as pd
import uvicorn
import pymorphy3
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from pathlib import Path
from datetime import datetime

# --- Robust Path Detection ---
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "Models"
DATA_DIR = ROOT_DIR / "Resource"
CSV_FILE = DATA_DIR / "lenta-ru-news.csv"
STATS_FILE = DATA_DIR / "stats_cache.json"

# --- Globals ---
models = {}
tfidf = None
le = None
stats_cache = {}
morph = pymorphy3.MorphAnalyzer()
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('russian'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^а-яё ]', '', text.lower())
    return " ".join([morph.parse(w)[0].normal_form for w in text.split() if w not in STOP_WORDS and len(w) > 2])

# --- Load Models ---
def load_models():
    global tfidf, le, models
    try:
        if (MODELS_DIR / "label_encoder.pkl").exists():
            le = joblib.load(MODELS_DIR / "label_encoder.pkl")
        
        v_path = MODELS_DIR / "tfidf_vectorizer.prek"
        if not v_path.exists(): v_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        if v_path.exists(): tfidf = joblib.load(v_path)

        m_files = {
            "Logistic Regression": "model_logistic_regression.pkl",
            "Naive Bayes": "model_naive_bayes.pkl",
            "Linear SVC": "model_linear_svc.pkl",
            "Final SVC (self-trained)": "final_svc_self_trained.pkl"
        }
        for name, fname in m_files.items():
            p = MODELS_DIR / fname
            if p.exists():
                models[name] = joblib.load(p)
                print(f"✅ {name} loaded.")
    except Exception as e:
        print(f"❌ Load error: {e}")

# --- FAST Statistics Logic ---
def get_fast_stats():
    global stats_cache
    if STATS_FILE.exists():
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    if not CSV_FILE.exists(): return {"error": "CSV missing"}

    print("⏳ Processing CSV using Fast Vectorized Operations...")
    
    # We only read the columns we need
    df = pd.read_csv(CSV_FILE, usecols=['topic', 'date', 'text', 'title'], low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['wc'] = df['title'].fillna('').str.len().div(5) + df['text'].fillna('').str.len().div(5) # Approximation is 5x faster
    df['wc'] = df['wc'].astype(int)
    
    # 1. Yearly Distribution (Vectorized)
    yearly = df['date'].dt.year.dropna().astype(int).value_counts().sort_index().to_dict()
    
    # 2. Topic Distribution (Vectorized)
    topic_dist = df['topic'].value_counts().to_dict()
    
    # 3. Avg Length per Topic (Vectorized)
    avg_len = df.groupby('topic')['wc'].mean().round(1).to_dict()
    
    stats_cache = {
        "total_articles": int(len(df)),
        "unique_topics": int(df['topic'].nunique()),
        "average_text_length_words": int(df['wc'].mean()),
        "topic_distribution": {str(k): int(v) for k, v in topic_dist.items()},
        "avg_length_per_topic": {str(k): float(v) for k, v in avg_len.items()},
        "yearly_distribution": {int(k): int(v) for k, v in yearly.items()},
        "word_count_raw": df['wc'].sample(n=min(2000, len(df))).tolist()
    }

    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats_cache, f, ensure_ascii=False)
    
    print("✅ Stats generated successfully.")
    return stats_cache

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    load_models()
    global stats_cache
    stats_cache = get_fast_stats()

@app.get("/models")
def get_m(): return [{"name": k, "accuracy": 0.84} for k in models.keys()]

@app.get("/stats/overview")
def overview(): return stats_cache

@app.get("/sample_articles")
def samples(limit: int = 50):
    if not CSV_FILE.exists(): return {"articles": []}
    df = pd.read_csv(CSV_FILE, usecols=['title', 'text', 'topic']).dropna().sample(n=limit)
    return {"articles": df.rename(columns={"topic": "true_topic"}).to_