import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import textwrap
import streamlit as st

@st.cache_resource
def load_gpt_model_and_tokenizer(model_path):
    if not os.path.exists(os.path.join(model_path, "gpt.pt")):
        return None, None
    
    # Загружаем токенизатор и базовую модель
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    # Загружаем веса обученной модели
    model.load_state_dict(torch.load(os.path.join(model_path, "gpt.pt")))
    model.eval()
    return tokenizer, model

# --- Путь к модели ---
MODEL_PATH = "."  # Корневая директория (где находится gpt.pt)

tokenizer, model = load_gpt_model_and_tokenizer(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model is not None:
    model.to(device).eval()

def show():
    st.title("Генерация текста GPT-моделью")

    if model is None:
        st.error("❌ Модель не найдена. Убедитесь, что файл 'gpt.pt' находится в корневой папке.")
        return

    prompt = st.text_area("Введите начальный текст (prompt):", height=150)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_length = st.slider("Длина генерации", min_value=20, max_value=200, value=100)
    with col2:
        num_return_sequences = st.slider("Число вариантов генерации", min_value=1, max_value=5, value=3)
    with col3:
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        top_k = st.slider("Top-k sampling", min_value=10, max_value=500, value=500)

    if st.button("Сгенерировать текст"):
        if not prompt.strip():
            st.warning("Введите текст для генерации!")
        else:
            with st.spinner("Генерация..."):
                encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
                out = model.generate(
                    input_ids=encoded_prompt,
                    max_length=max_length,
                    num_beams=10,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    no_repeat_ngram_size=3,
                    num_return_sequences=num_return_sequences,
                )

                st.subheader("Результаты генерации:")
                for i, generated in enumerate(out):
                    decoded = tokenizer.decode(generated, skip_special_tokens=True)
                    st.markdown(f"### Вариант {i+1}:")
                    st.markdown(f"> {textwrap.fill(decoded, 80)}")
                    st.markdown("---")