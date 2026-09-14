import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Taller 6 - Panel de predicción", page_icon="🤖", layout="wide")

# ============================================================
# MENÚ LATERAL: DOS SECCIONES CON NOMBRES FÁCILES
# ============================================================
st.sidebar.title("🤖 Taller 6 - Panel de predicción")
opcion = st.sidebar.radio(
    "Selecciona la sección:",
    ("📦 ENVIOS (E-COMMERCE)", "🍷 VINOS (WINE)"),
)

@st.cache_resource
def cargar_archivo(nombre):
    try:
        return joblib.load(nombre)
    except FileNotFoundError:
        return None

# ============================================================
# SECCIÓN 1: ENVIOS (E-COMMERCE)
# ============================================================
if opcion == "📦 ENVIOS (E-COMMERCE)":
    st.title("📦 SECCIÓN ENVIOS (E-COMMERCE)")
    st.write("Modelo: **K-Means de Train.csv** (punto 4.2) · Archivos: `scaler_ecom.joblib` y `km_ecom.joblib`")
    st.write("Ingresa los datos de un nuevo envío y la app te dice a qué **segmento logístico** pertenece.")

    scaler_ecom = cargar_archivo("scaler_ecom.joblib")
    km_ecom = cargar_archivo("km_ecom.joblib")

    if scaler_ecom is None or km_ecom is None:
        st.error("Faltan `scaler_ecom.joblib` o `km_ecom.joblib` en esta carpeta. "
                 "Ejecuta en tu cuaderno: joblib.dump(scaler_ecom, 'scaler_ecom.joblib') y "
                 "joblib.dump(km_ecom, 'km_ecom.joblib'), y súbelos al repositorio.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        weight = st.number_input("Peso (gramos)", 90, 8000, 2500, 10)
        cost = st.number_input("Costo del producto", 90, 320, 200, 1)
    with c2:
        discount = st.number_input("Descuento ofrecido", 1, 65, 30, 1)
        prior = st.number_input("Compras previas", 1, 10, 3, 1)
    with c3:
        calls = st.number_input("Llamadas a servicio", 2, 7, 3, 1)
        rating = st.number_input("Calificación cliente", 1, 5, 3, 1)

    if st.button("🔍 Predecir segmento del envío", type="primary"):
        X = pd.DataFrame([{
            "Weight_in_gms": weight,
            "Cost_of_the_Product": cost,
            "Discount_offered": discount,
            "Prior_purchases": prior,
            "Customer_care_calls": calls,
            "Customer_rating": rating,
        }])
        Xs = scaler_ecom.transform(X)
        pred = int(km_ecom.predict(Xs)[0])
        dist = np.linalg.norm(Xs - km_ecom.cluster_centers_, axis=1)

        st.success(f"El envío pertenece al **SEGMENTO {pred}**.")

        # Perfil del segmento: su centroide comparado con el promedio de centroides
        media = km_ecom.cluster_centers_.mean(axis=0)
        nombres = X.columns.tolist()
        perfil = [f"{n}: {'alto' if v > m else 'bajo'}"
                  for n, v, m in zip(nombres, km_ecom.cluster_centers_[pred], media)]
        st.info("Perfil del segmento (vs. promedio): " + " · ".join(perfil))

        st.write("Distancia a cada centroide (menor = más parecido a ese segmento):")
        st.bar_chart(pd.DataFrame(
            {"distancia": dist},
            index=[f"Segmento {i}" for i in range(len(dist))]))

# ============================================================
# SECCIÓN 2: VINOS (WINE)
# ============================================================
else:
    st.title("🍷 SECCIÓN VINOS (WINE)")
    st.write("Modelo: **K-NN entrenado sobre los clústeres de K-Means de Wine** (puntos 4.3–4.4) · "
             "Archivos: `scaler_wine.joblib` y `knn_wine.joblib`")
    st.write("Ingresa las 13 variables químicas y la app te dice a qué **segmento** pertenece el vino.")

    scaler_wine = cargar_archivo("scaler_wine.joblib")
    knn_wine = cargar_archivo("knn_wine.joblib")

    if scaler_wine is None or knn_wine is None:
        st.error("Faltan `scaler_wine.joblib` o `knn_wine.joblib` en esta carpeta.")
        st.stop()

    FEATURES_WINE = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                     "total_phenols", "flavanoids", "nonflavanoid_phenols",
                     "proanthocyanins", "color_intensity", "hue",
                     "od280/od315_of_diluted_wines", "proline"]
    RANGOS = {
        "alcohol": (11.0, 14.9, 13.0, 0.01),
        "malic_acid": (0.7, 5.8, 2.3, 0.01),
        "ash": (1.3, 3.3, 2.4, 0.01),
        "alcalinity_of_ash": (7.0, 30.0, 19.5, 0.5),
        "magnesium": (70.0, 162.0, 99.0, 1.0),
        "total_phenols": (0.9, 3.9, 2.4, 0.01),
        "flavanoids": (0.3, 5.1, 2.4, 0.01),
        "nonflavanoid_phenols": (0.06, 0.66, 0.36, 0.01),
        "proanthocyanins": (0.4, 3.6, 1.6, 0.01),
        "color_intensity": (1.3, 13.0, 5.1, 0.01),
        "hue": (0.48, 1.71, 0.96, 0.01),
        "od280/od315_of_diluted_wines": (1.15, 4.0, 2.6, 0.01),
        "proline": (278.0, 1680.0, 746.0, 1.0),
    }

    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(FEATURES_WINE):
        mn, mx, df_, paso = RANGOS[f]
        with cols[i % 3]:
            vals[f] = st.number_input(f, mn, mx, df_, paso)

    if st.button("🔍 Predecir segmento del vino", type="primary"):
        X = pd.DataFrame([vals])[FEATURES_WINE]
        Xs = scaler_wine.transform(X)
        pred = int(knn_wine.predict(Xs)[0])
        proba = knn_wine.predict_proba(Xs)[0]

        st.success(f"El vino pertenece al **SEGMENTO {pred}**.")
        st.metric("Confianza (votación de vecinos)", f"{proba.max():.0%}")
        st.write("Votación de los vecinos por segmento:")
        st.bar_chart(pd.DataFrame(
            {"proporción": proba},
            index=[f"Segmento {i}" for i in knn_wine.classes_]))