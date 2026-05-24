import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Lenta.ru Analytics", layout="wide")

API_URL = "http://localhost:8000"

@st.cache_data(ttl=60)
def get_api(endpoint, params=None):
    try:
        r = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

stats = get_api("stats/overview")
models_list = get_api("models") or []
articles_resp = get_api("sample_articles", {"limit": 100})
articles = articles_resp.get("articles", []) if articles_resp else []

st.title("Lenta.ru — анализ новостей")

if stats is None:
    st.error("API недоступен. Запустите python Api.py")
    st.stop()

# --- Метрики ---
col1, col2, col3 = st.columns(3)
col1.metric("Всего статей", f"{stats.get('total_articles', 0):,}")
col2.metric("Рубрик", stats.get('unique_topics', 0))
col3.metric("Средняя длина (слов)", stats.get('average_text_length_words', 0))

st.divider()

# --- Классификатор ---
st.subheader("Классификатор текста")

left, right = st.columns(2)

with left:
    source = st.radio("Источник:", ["Ввести текст", "Случайная статья"])
    input_text = ""

    if source == "Ввести текст":
        input_text = st.text_area("Текст новости:", height=150)
    else:
        if articles:
            idx = st.selectbox("Выбрать статью:", range(len(articles)),
                               format_func=lambda i: articles[i].get("title", f"Статья {i}")[:80])
            art = articles[idx]
            input_text = art.get("title", "") + " " + art.get("text", "")
            st.info(f"Рубрика в базе: {art.get('true_topic', '?')}")
        else:
            st.warning("Статьи не загружены")

with right:
    if models_list:
        selected_model = st.selectbox("Модель:", [m["name"] for m in models_list])

        if st.button("Классифицировать"):
            if not input_text.strip():
                st.warning("Введите текст")
            else:
                with st.spinner("Анализ..."):
                    try:
                        resp = requests.post(f"{API_URL}/predict",
                                             json={"model_name": selected_model, "text": input_text},
                                             timeout=20)
                        if resp.status_code == 200:
                            res = resp.json()
                            st.success(f"Рубрика: **{res.get('category', '—')}**")
                            conf = res.get("confidence")
                            if conf:
                                st.write(f"Уверенность: {conf:.1%}")
                            all_probs = res.get("all_probabilities")
                            if all_probs:
                                top5 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                                df_p = pd.DataFrame(top5, columns=["Рубрика", "Вероятность"])
                                fig = px.bar(df_p, x="Вероятность", y="Рубрика", orientation="h")
                                fig.update_layout(height=220, margin=dict(l=8,r=8,t=8,b=8))
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Ошибка API: {resp.status_code}")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
    else:
        st.warning("Модели не загружены")

st.divider()

# --- Графики ---
st.subheader("Статистика корпуса")

tab1, tab2, tab3, tab4 = st.tabs(["По годам", "Рубрики", "Длина статей", "Рубрики: статьи vs длина"])

with tab1:
    yearly = {int(k): v for k, v in stats.get("yearly_distribution", {}).items()}
    df_y = pd.DataFrame(list(yearly.items()), columns=["Год", "Статей"]).sort_values("Год")
    fig = px.line(df_y, x="Год", y="Статей", title="Количество статей по годам", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_t = pd.DataFrame(list(stats.get("topic_distribution", {}).items()),
                        columns=["Рубрика", "Кол-во"]).sort_values("Кол-во", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df_t.head(12), values="Кол-во", names="Рубрика",
                     title="Топ-12 рубрик", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(df_t.head(15).sort_values("Кол-во"), x="Кол-во", y="Рубрика",
                     orientation="h", title="Топ-15 рубрик")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    df_l = pd.DataFrame(list(stats.get("avg_length_per_topic", {}).items()),
                        columns=["Рубрика", "Слов"]).sort_values("Слов")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df_l, x="Слов", y="Рубрика", orientation="h",
                     title="Средняя длина по рубрикам")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        wc = stats.get("word_count_raw", [])
        if wc:
            fig = px.histogram(x=wc, nbins=50, title="Распределение длин статей",
                               labels={"x": "Слов", "y": "Статей"})
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    df_t2 = pd.DataFrame(list(stats.get("topic_distribution", {}).items()),
                         columns=["Рубрика", "Кол-во"])
    df_l2 = pd.DataFrame(list(stats.get("avg_length_per_topic", {}).items()),
                         columns=["Рубрика", "Слов"])
    df_scatter = df_t2.merge(df_l2, on="Рубрика")
    fig = px.scatter(df_scatter, x="Кол-во", y="Слов", text="Рубрика",
                     title="Рубрики: количество статей vs средняя длина",
                     labels={"Кол-во": "Кол-во статей", "Слов": "Средняя длина (слов)"})
    fig.update_traces(textposition="top center", marker=dict(size=8))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)