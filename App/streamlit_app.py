# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Классификатор новостей", layout="wide")
st.title("Классификатор новостей Lenta.ru")
st.markdown("Выберите новость из списка и модели для сравнения")

# ---------- Загрузка списка моделей ----------
@st.cache_data(ttl=600)
def get_models():
    try:
        resp = requests.get(f"{API_URL}/models")
        resp.raise_for_status()
        return resp.json()
    except:
        st.error("Не удалось подключиться к API. Запустите FastAPI на порту 8000.")
        return []

# ---------- Загрузка примеров новостей ----------
@st.cache_data(ttl=3600)
def get_sample_articles(limit=100):
    try:
        resp = requests.get(f"{API_URL}/sample_articles", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()["articles"]
    except:
        st.error("Не удалось загрузить образцы новостей.")
        return []

# ---------- Загрузка статистики датасета ----------
@st.cache_data(ttl=3600)
def get_statistics():
    try:
        resp = requests.get(f"{API_URL}/stats/overview")
        resp.raise_for_status()
        return resp.json()
    except:
        st.error("Не удалось загрузить статистику.")
        return None

models_info = get_models()
if not models_info:
    st.stop()

articles = get_sample_articles(limit=100)
if not articles:
    st.stop()

# ---------- Основные вкладки ----------
tab1, tab2 = st.tabs(["Сравнение моделей", "Статистика датасета"])

# ----- Вкладка 1: Сравнение моделей -----
with tab1:
    st.subheader("Выберите новость и модели для предсказания")

    selected_idx = st.selectbox(
        "Выберите новость:",
        options=range(len(articles)),
        format_func=lambda i: f"{articles[i]['title']} (истинная тема: {articles[i]['true_topic']})"
    )
    selected_article = articles[selected_idx]

    st.markdown("---")
    st.write("**Текст новости:**")
    st.write(selected_article["text"])

    st.markdown("---")
    st.subheader("Выберите модели для сравнения")
    selected_models = {}
    cols = st.columns(4)
    for i, model in enumerate(models_info):
        with cols[i % 4]:
            selected_models[model["name"]] = st.checkbox(model["name"], value=True)

    if st.button("Сравнить модели", type="primary"):
        if not any(selected_models.values()):
            st.warning("Выберите хотя бы одну модель.")
        else:
            results = []
            for model_name, selected in selected_models.items():
                if not selected:
                    continue
                full_text = selected_article["title"] + " " + selected_article["text"]
                payload = {"model_name": model_name, "text": full_text}
                resp = requests.post(f"{API_URL}/predict", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    results.append({
                        "Модель": model_name,
                        "Предсказанная тема": data["predicted_category"],
                        "Уверенность": f"{data['confidence']:.2%}"
                    })
                else:
                    results.append({"Модель": model_name, "Предсказанная тема": "Ошибка", "Уверенность": "—"})
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            st.subheader("Детальные вероятности для модели")
            chosen_model = st.selectbox(
                "Выберите модель для просмотра распределения вероятностей:",
                [m["name"] for m in models_info if selected_models.get(m["name"], False)]
            )
            if chosen_model:
                full_text = selected_article["title"] + " " + selected_article["text"]
                payload = {"model_name": chosen_model, "text": full_text}
                resp = requests.post(f"{API_URL}/predict", json=payload)
                if resp.status_code == 200:
                    probs = resp.json()["all_probabilities"]
                    prob_df = pd.DataFrame(probs.items(), columns=["Тема", "Вероятность"])
                    fig = px.bar(prob_df, x="Вероятность", y="Тема", orientation='h',
                                 title=f"Вероятности – {chosen_model}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Не удалось получить вероятности")

# ----- Вкладка 2: Статистика датасета (расширенная) -----
with tab2:
    st.subheader("Статистика датасета Lenta.ru")
    stats = get_statistics()
    if stats:
        # Общие метрики в 4 колонках
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего новостей", f"{stats['total_articles']:,}")
        col2.metric("Уникальных тем", stats['unique_topics'])
        col3.metric("Средняя длина (слова)", stats['average_text_length_words'])
        col4.metric("Средняя длина (символы)", stats['average_text_length_chars'])

        # Дополнительные статистики: минимум, максимум, медиана
        st.write("**Распределение длины текста:**")
        col5, col6, col7 = st.columns(3)
        col5.metric("Минимальная длина (символы)", stats['min_length_chars'])
        col6.metric("Максимальная длина (символы)", stats['max_length_chars'])
        col7.metric("Медианная длина (символы)", stats['median_length_chars'])

        col8, col9, col10 = st.columns(3)
        col8.metric("Минимальная длина (слова)", stats['min_length_words'])
        col9.metric("Максимальная длина (слова)", stats['max_length_words'])
        col10.metric("Медианная длина (слова)", stats['median_length_words'])

        st.write("**Период публикаций:**", f"{stats['date_range']['min']} — {stats['date_range']['max']}")

        # Полное распределение по темам
        st.write("**Полное распределение по темам:**")
        topics_dict = stats['topic_distribution']
        topic_df = pd.DataFrame(topics_dict.items(), columns=["Тема", "Количество"])
        topic_df = topic_df.sort_values("Количество", ascending=False)
        fig = px.bar(topic_df, x="Количество", y="Тема", orientation='h',
                     title="Количество статей по темам (все темы)")
        st.plotly_chart(fig, use_container_width=True)

        # Таблица с долями
        topic_df["Доля, %"] = (topic_df["Количество"] / stats['total_articles'] * 100).round(2)
        st.dataframe(topic_df, use_container_width=True)
    else:
        st.warning("Статистика недоступна. Убедитесь, что API запущен.")