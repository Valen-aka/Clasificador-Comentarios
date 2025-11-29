import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer


# CARGAR MODELOS Y VECTORIZADOR
nb_model = joblib.load("modelos/naive_bayes_model.pkl")
lr_model = joblib.load("modelos/logistic_regression_model.pkl")
vectorizer = joblib.load("modelos/tfidf_vectorizer.pkl")
encoder = joblib.load("modelos/label_encoder.pkl")


#PREPROCESAMIENTO

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
tokenizer = ToktokTokenizer()

def normalize_and_tokenize(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def classify_with_confidence(text, model):
    clean_text = normalize_and_tokenize(text)
    vect = vectorizer.transform([clean_text])

    pred_class = model.predict(vect)[0]
    pred_label = encoder.inverse_transform([pred_class])[0]

    probs = model.predict_proba(vect)[0]

    table = pd.DataFrame({
        "Categoria": encoder.classes_,
        "Probabilidad": probs
    }).sort_values(by="Probabilidad", ascending=False)

    confidence = probs[pred_class]

    return pred_label, confidence, table

#INTERFAZ
st.title("📌 Clasificador de Reclamos")
st.write("Escribe el texto y selecciona el modelo.")

# Entrada de texto
texto = st.text_area("Escribe un comentario:", height=150)

# Botones
col1, col2 = st.columns(2)

if col1.button("🤖 Naive Bayes"):
    categoria, confianza, tabla = classify_with_confidence(texto, nb_model)
    st.subheader(f"Categoría predicha: {categoria}")
    st.write(f"Confianza del modelo: **{confianza*100:.2f}%**")
    st.dataframe(tabla)

if col2.button("📈 Regresión Logística"):
    categoria, confianza, tabla = classify_with_confidence(texto, lr_model)
    st.subheader(f"Categoría predicha: {categoria}")
    st.write(f"Confianza del modelo: **{confianza*100:.2f}%**")
    st.dataframe(tabla)
