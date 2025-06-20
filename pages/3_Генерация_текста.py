import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import os
import textwrap
import streamlit as st

@st.cache_resource
def load_gpt_model_and_tokenizer(model_path):
    try:
        model_file = os.path.join(model_path, "gpt.pt")
        if not os.path.exists(model_file):
            st.error(f"Файл модели не найден по пути: {model_file}")
            return None, None

        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Создаем модель по архитектуре
        model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

        # Загружаем веса (state_dict)
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        return tokenizer, model

    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None, None

MODEL_PATH = "."  # Корень, где лежит gpt.pt
tokenizer, model = load_gpt_model_and_tokenizer(MODEL_PATH)

def show():
    st.title("Генерация текста GPT-моделью")
    st.write("Введите текст и настройте параметры генерации")

    if model is None:
        st.error("❌ Модель не загружена. Проверьте:")
        st.markdown("""
        1. Файл `gpt.pt` должен находиться в корневой папке
        2. Размер файла должен быть около 400 МБ
        3. Модель должна быть сохранена через `torch.save(model.state_dict(), 'gpt.pt')`
        """)
        return

    prompt = st.text_area("Введите начальный текст (prompt):", height=150)

    col1, col2, col3 = st.columns(3)
    with col1:
        max_length = st.slider("Длина генерации", 20, 200, 100)
    with col2:
        num_return_sequences = st.slider("Число вариантов", 1, 5, 3)
    with col3:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        top_k = st.slider("Top-k", 10, 500, 500)

    if st.button("Сгенерировать текст"):
        if not prompt.strip():
            st.warning("Введите текст для генерации!")
        else:
            with st.spinner("Генерация..."):
                try:
                    device = next(model.parameters()).device
                    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)

                    outputs = model.generate(
                        input_ids=encoded_prompt,
                        max_length=max_length,
                        num_beams=10,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        no_repeat_ngram_size=3,
                        num_return_sequences=num_return_sequences,
                    )

                    st.subheader("Результаты:")
                    for i, generated in enumerate(outputs):
                        decoded = tokenizer.decode(generated, skip_special_tokens=True)
                        st.markdown(f"### Вариант {i+1}:")
                        st.markdown(f"> {textwrap.fill(decoded, 80)}")
                        st.markdown("---")

                except Exception as e:
                    st.error(f"Ошибка генерации: {str(e)}")

if __name__ == "__main__":
    show()
