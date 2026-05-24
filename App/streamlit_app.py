import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ---------- Page config ----------
st.set_page_config(
    page_title="Lenta.ru News Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- DARK THEME CSS (dark background, light text, cards) ----------
st.markdown(
    """
    <style>
    /* Main dark background */
    .stApp {
        background-color: #0e1117 !important;
    }
    /* Hide sidebar completely */
    [data-testid="collapsedControl"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    /* Dark cards */
    .dashboard-card {
        background-color: #1e2229;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        border: 1px solid #2d313a;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4c9aff;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    /* Headers and text – light */
    h1, h2, h3, .stMarkdown, label, .stTextArea label, .stSelectbox label, .stRadio label {
        color: #e2e8f0 !important;
    }
    /* Input fields dark */
    .stTextArea textarea, .stSelectbox div, .stSelectbox div[data-baseweb="select"] {
        background-color: #1e2229;
        color: #e2e8f0;
        border-color: #2d313a;
        border-radius: 8px;
    }
    /* Buttons */
    .stButton button {
        background-color: #2d6a4f;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    .stButton button:hover {
        background-color: #1b4d3e;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e2229;
        color: #e2e8f0;
        border-radius: 8px;
    }
    /* Dataframe */
    .dataframe {
        background-color: #1e2229;
        color: #e2e8f0;
    }
    hr {
        margin: 1.5rem 0;
        border-color: #2d313a;
    }
    /* Plotly charts background transparent */
    .plotly-graph-div .main-svg {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_URL = "http://localhost:8000"

# ---------- API calls ----------
@st.cache_data(ttl=600)
def get_models():
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except:
        st.error("❌ FastAPI не запущен. Запустите `python api.py`")
        return []

@st.cache_data(ttl=3600)
def get_sample_articles(limit=50):
    try:
        resp = requests.get(f"{API_URL}/sample_articles", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except:
        return []

@st.cache_data(ttl=3600)
def get_statistics():
    try:
        resp = requests.get(f"{API_URL}/stats/overview", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

# ---------- Load data ----------
models_info = get_models()
if not models_info:
    st.stop()

articles = get_sample_articles(limit=100)
stats = get_statistics()

# ---------- Dashboard Header ----------
st.title("📰 Lenta.ru News Classifier")
st.markdown("---")

# ---------- Model Selection Row (cards) ----------
st.markdown("### 🤖 Выберите модели")
model_cols = st.columns(len(models_info))
selected_models = {}
for i, model in enumerate(models_info):
    with model_cols[i]:
        with st.container():
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            sel = st.checkbox(
                f"**{model['name']}**  \n{model['accuracy']:.2%} accuracy",
                value=True,
                key=model["name"]
            )
            selected_models[model["name"]] = sel
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- Input Section ----------
input_mode = st.radio(
    "📰 Источник новости:",
    ["✍️ Ввести свой текст", "📋 Выбрать из образцов"],
    horizontal=True,
)

classification_text = ""
true_topic = "—"

if input_mode == "✍️ Ввести свой текст":
    classification_text = st.text_area(
        "Введите текст новости (заголовок + содержание):",
        height=200,
        placeholder="Например: Сборная России по футболу выиграла чемпионат мира..."
    ).strip()
else:
    if articles:
        article_titles = [f"{a['title'][:70]}... (тема: {a['true_topic']})" for a in articles]
        selected_idx = st.selectbox(
            "Выберите новость из датасета:",
            range(len(articles)),
            format_func=lambda i: article_titles[i],
        )
        article = articles[selected_idx]
        classification_text = article["title"] + " " + article["text"]
        true_topic = article["true_topic"]
        with st.expander("📄 Просмотреть текст новости"):
            st.write(classification_text[:1000] + ("..." if len(classification_text) > 1000 else ""))
    else:
        st.warning("Образцы новостей не загружены (возможно, нет CSV).")
        classification_text = ""

# ---------- Classification Button ----------
col_btn, _ = st.columns([1, 3])
with col_btn:
    classify = st.button("🚀 Классифицировать", type="primary", use_container_width=True)

# ---------- Results Area ----------
if classify:
    if not classification_text:
        st.warning("Пожалуйста, введите текст или выберите образец.")
    elif not any(selected_models.values()):
        st.warning("Выберите хотя бы одну модель.")
    else:
        with st.spinner("Анализ текста..."):
            results = []
            probabilities = {}
            for model_name, selected in selected_models.items():
                if not selected:
                    continue
                payload = {"model_name": model_name, "text": classification_text}
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        results.append({
                            "Модель": model_name,
                            "Категория": data["predicted_category"],
                            "Уверенность": f"{data['confidence']:.2%}",
                        })
                        probabilities[model_name] = data["all_probabilities"]
                    else:
                        results.append({"Модель": model_name, "Категория": "Ошибка API", "Уверенность": "—"})
                except Exception:
                    results.append({"Модель": model_name, "Категория": "Соединение не удалось", "Уверенность": "—"})

        # Show results
        st.subheader("🏆 Результаты предсказания")
        if true_topic != "—":
            st.info(f"📌 Истинная категория (из датасета): **{true_topic}**")
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        # Probability distribution
        if probabilities:
            st.subheader("📊 Детальные вероятности")
            chosen = st.selectbox("Модель для детализации:", list(probabilities.keys()))
            if chosen:
                probs = probabilities[chosen]
                prob_df = pd.DataFrame(probs.items(), columns=["Тема", "Вероятность"])
                prob_df = prob_df.sort_values("Вероятность", ascending=False).head(10)
                fig = px.bar(
                    prob_df,
                    x="Вероятность",
                    y="Тема",
                    orientation='h',
                    title=f"Топ‑10 тем – {chosen}",
                    color="Вероятность",
                    color_continuous_scale="blues",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- Statistics Dashboard ----------
if stats:
    st.subheader("📈 Статистика датасета")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["total_articles"]:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Всего новостей</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["unique_topics"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Уникальных тем</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["average_text_length_words"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Средняя длина (слова)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{stats["average_text_length_chars"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Средняя длина (символы)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Topic distribution
    topics = stats["topic_distribution"]
    topic_df = pd.DataFrame(topics.items(), columns=["Тема", "Количество"])
    topic_df = topic_df.sort_values("Количество", ascending=False)
    fig2 = px.bar(
        topic_df.head(15),
        x="Количество",
        y="Тема",
        orientation='h',
        title="Топ‑15 категорий",
        color="Количество",
        color_continuous_scale="viridis",
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=500,
        yaxis={'categoryorder': 'total ascending'},
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Статистика недоступна. Убедитесь, что CSV-файл находится в папке `Resource/`.")