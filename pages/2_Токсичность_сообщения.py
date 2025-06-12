import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import textwrap

@st.cache_resource
def load_model():
    model_name = "cointegrated/rubert-tiny-toxicity"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

classes = ['нормальный', 'оскорбления', 'мат', 'угрозы', 'неуместный']

def analyze_text_detailed(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        logits = model(**inputs).logits
        proba = torch.sigmoid(logits).cpu().numpy()

    if isinstance(text, str):
        proba = proba[0]

    aggregate_toxicity = 1 - proba[0] * (1 - proba[-1])
    predicted_class_idx = proba.argmax()
    predicted_class = classes[predicted_class_idx]
    confidence = float(proba[predicted_class_idx])
    is_safe = proba[0] > 0.5 and proba[-1] < 0.5

    return {
        'aggregate_toxicity': float(aggregate_toxicity),
        'is_safe': is_safe,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {classes[i]: float(proba[i]) for i in range(len(classes))}
    }

def show():
    st.title("Оценка степени токсичности")
    st.write("Введите текст для проверки:")

    user_input = st.text_area("", height=200)

    if st.button("Проверка"):
        if not user_input.strip():
            st.warning("Введите текст для анализа!")
        else:
            result = analyze_text_detailed(user_input)

            st.subheader("Результат анализа:")
            toxic_percent = result['aggregate_toxicity'] * 100
            st.write(f"**Степень токсичности:** {toxic_percent:.1f}%")

            st.write(f"**Класс токсичности:** {result['predicted_class'].upper()}")
            st.write(f"**Уверенность модели:** {result['confidence'] * 100:.1f}%")

            st.write("**Вероятности по категориям:**")
            for cls, prob in result['probabilities'].items():
                st.write(f"- {cls}: {prob * 100:.1f}%")

            if toxic_percent > 70:
                st.error("⚠️ Рекомендуется удалить комментарий.")
            elif toxic_percent > 40:
                st.warning("❗ Рекомендуется модерировать комментарий.")
            else:
                st.success("✅ Комментарий можно оставить.")

if __name__ == "__main__":
    show()
