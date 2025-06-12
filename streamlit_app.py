import streamlit as st

# Настройка страницы
st.set_page_config(page_title="Мультистраничное приложение", layout="wide")

# Меню навигации
st.sidebar.title("Навигация")
page = st.sidebar.selectbox("Выберите страницу", [
    "Классификация отзыва на фильм (в разработке)",
    "Оценка степени токсичности",
    "Генерация текста GPT"
])

# Заглушка: Классификация отзыва
if page == "Классификация отзыва на фильм (в разработке)":
    st.title("Классификация отзыва на фильм")
    st.info("Страница находится в разработке.")

# Подключаем другие страницы
elif page == "Оценка степени токсичности":
    from pages import toxic_page
    toxic_page.show()

elif page == "Генерация текста GPT":
    from pages import generation_page
    generation_page.show()