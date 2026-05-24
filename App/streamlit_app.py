import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lenta Analytics", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .card { background-color: #1e2229; padding: 15px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
    .metric { color: #58a6ff; font-size: 22px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

@st.cache_data(ttl=60)
def get_api(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=30)
        return r.json() if r.status_code == 200 else None
    except: return None

# Load Data
stats = get_api("stats/overview")
models_list = get_api("models") or []
articles_resp = get_api("sample_articles", {"limit": 50})
articles = articles_resp.get("articles", []) if articles_resp else []

st.title("📰 Lenta.ru Dashboard")

# --- Classifier ---
st.header("🔍 Классификатор")
c1, c2 = st.columns(2)

with c1:
    source = st.radio("Источник:", ["Ввод текста", "Случайная новость"])
    if source == "Ввод текста":
        input_text = st.text_area("Текст:", height=150)
    else:
        if articles:
            sel = st.selectbox("Заголовок:", range(len(articles)), format_func=lambda i: articles[i]['title'])
            input_text = articles[sel]['title'] + " " + articles[sel]['text']
            st.info(f"Тема в базе: {articles[sel]['true_topic']}")
        else: input_text = ""

with c2:
    if models_list:
        selected_m = st.selectbox("Модель:", [m['name'] for m in models_list])
        if st.button("🚀 Анализировать") and input_text:
            try:
                res = requests.post(f"{API_URL}/predict", json={"model_name": selected_m, "text": input_text}).json()
                st.success(f"**Предсказано:** {res['category']}")
                st.write(f"Уверенность: {res['confidence']:.2%}")
            except: st.error("Ошибка API")

# --- Stats Section ---
st.markdown("---")
st.header("📊 Аналитика данных")

if not stats or "total_articles" not in stats:
    st.info("⌛ Статистика загружается. Пожалуйста, обновите страницу через несколько секунд.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='card'><div class='metric'>{stats['total_articles']:,}</div>Статей</div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='card'><div class='metric'>{stats['unique_topics']}</div>Тем</div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='card'><div class='metric'>{stats['average_text_length_words']}</div>Ср. слов</div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='card'><div class='metric'>1999-2019</div>Период</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📅 Тренды", "📏 Длины", "🥧 Категории"])

    with tab1:
        df_y = pd.DataFrame(list(stats['yearly_distribution'].items()), columns=["Год", "Кол-во"])
        st.plotly_chart(px.line(df_y, x="Год", y="Кол-во", template="plotly_dark"), use_container_width=True)

    with tab2:
        df_l = pd.DataFrame(list(stats['avg_length_per_topic'].items()), columns=["Тема", "Слова"]).sort_values("Слова")
        st.plotly_chart(px.bar(df_l, x="Слова", y="Тема", orientation='h', template="plotly_dark"), use_container_width=True)

    with tab3:
        df_t = pd.DataFrame(list(stats['topic_distribution'].items()), columns=["Тема", "Всего"]).sort_values("Всего", ascending=False)
        st.plotly_chart(px.pie(df_t.head(15), values='Всего', names='Тема', template="plotly_dark"), use_container_width=True)