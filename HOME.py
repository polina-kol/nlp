import streamlit as st

# Установи заголовок приложения
st.set_page_config(page_title="NLP App", layout="wide")

# Навигация через sidebar
st.sidebar.title("Навигация")
page = st.sidebar.selectbox("Выберите страницу", ["Классификация отзыва на фильм", "Оценка степени токсичности", "Генерация текста GPT"])

# Загрузка нужной страницы
if page == "Классификация отзыва на фильм":
    st.title("Классификация отзыва на фильм")
    st.info("Страница в разработке.")

elif page == "Оценка степени токсичности":
    from pages import toxicity_page
    toxicity_page.show()

elif page == "Генерация текста GPT":
    from pages import generation_page
    generation_page.show()