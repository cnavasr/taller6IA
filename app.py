import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Clasificador de vinos K-NN", page_icon="🍷")

FEATURES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols",
            "proanthocyanins", "color_intensity", "hue",
            "od280/od315_of_diluted_wines", "proline"]

RANGOS = {
    "alcohol": (11.0, 14.8, 13.0, 0.01),
    "malic_acid": (0.74, 5.80, 2.30, 0.01),
    "ash": (1.36, 3.23, 2.36, 0.01),
    "alcalinity_of_ash": (7.0, 30.0, 19.5, 0.5),
    "magnesium": (70.0, 162.0, 99.0, 1.0),
    "total_phenols": (0.98, 3.88, 2.36, 0.01),
    "flavanoids": (0.34, 5.08, 2.36, 0.01),
    "nonflavanoid_phenols": (0.06, 0.66, 0.36, 0.01),
    "proanthocyanins": (0.41, 3.58, 1.59, 0.01),
    "color_intensity": (1.28, 13.00, 5.08, 0.01),
    "hue": (0.48, 1.71, 0.96, 0.01),
    "od280/od315_of_diluted_wines": (1.15, 4.00, 2.61, 0.01),
    "proline": (278.0, 1680.0, 746.0, 1.0),
}

DESCRIPCION_CLUSTER = {
    0: "Segmento 0: vinos con fenoles, flavanoides y prolina altos (cultivar clase 0 predominante).",
    1: "Segmento 1: vinos de color y matiz intermedios-bajos (clase 1 predominante).",
    2: "Segmento 2: vinos con color y matiz altos pero fenoles medios (clase 2 predominante).",
}

@st.cache_resource
def cargar_modelo():
    knn = joblib.load("knn_wine.joblib")
    scaler = joblib.load("scaler_wine.joblib")
    return knn, scaler

knn, scaler = cargar_modelo()

st.title("🍷 Clasificador de segmentos de vino (K-NN)")
st.write("Ingrese las 13 variables químicas de una muestra nueva y la app indica "
         "a qué segmento (clúster de K-Means) pertenece, con la confianza de la votación.")

st.sidebar.header("Variables de la muestra")
valores = {}
for f in FEATURES:
    mn, mx, df_, paso = RANGOS[f]
    valores[f] = st.sidebar.number_input(f, min_value=mn, max_value=mx, value=df_, step=paso)

if st.button("Predecir segmento"):
    X = pd.DataFrame([valores])[FEATURES]
    Xs = scaler.transform(X)
    pred = int(knn.predict(Xs)[0])
    proba = knn.predict_proba(Xs)[0]
    st.success(f"La muestra se asigna al **segmento {pred}**.")
    st.metric("Confianza (votación de vecinos)", f"{proba.max():.0%}")
    st.write(DESCRIPCION_CLUSTER[pred])
    st.write("Votación de los vecinos por segmento:")
    st.bar_chart(pd.DataFrame({"proporción": proba}, index=knn.classes_))
else:
    st.info("Presione el botón para clasificar la muestra.")
