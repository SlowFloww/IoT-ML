import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Smart Home AI - Personalizado", layout="wide")

# 1. Cargar el modelo (El modelo ORIGINAL con clo/met)
@st.cache_resource
def load_model():
    # Asegúrate de que este archivo exista (fue el primero que creamos)
    return joblib.load('modelo_control_hvac.pkl')

try:
    modelo = load_model()
except:
    st.error("⚠️ No se encontró 'modelo_control_hvac.pkl'. Necesitas el modelo entrenado con variables personales.")
    st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🏠 Control Térmico Personalizado (Equipo 3)")
st.markdown("""
Este sistema calcula el **Setpoint de Temperatura (Ta)** exacto para garantizar el confort de un usuario específico.
*Utiliza variables personales (`clo`, `met`) para máxima precisión.*
""")

st.divider()

# 2. Columnas para los controles (Inputs)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("👤 Variables Personales")
    clo = st.slider("👔 Ropa (clo)", 0.3, 1.5, 0.6, step=0.1, help="0.3: Shorts - 1.5: Abrigo")
    met = st.slider("🏃 Actividad (met)", 0.8, 4.0, 1.2, step=0.1, help="1.0: Sentado - 3.0: Ejercicio")

with col2:
    st.subheader("🌡️ Variables Ambientales")
    tr = st.slider("🧱 Temp. Paredes (Tr)", 15.0, 35.0, 25.0, step=0.5)
    rh = st.slider("💧 Humedad (%)", 20.0, 80.0, 50.0, step=5.0)
    vel = st.slider("💨 Viento (m/s)", 0.0, 1.0, 0.1, step=0.05)
    
    st.write("---")
    # LIMITACIÓN DE ALEX: PMV solo en rango de confort
    pmv = st.slider("🎯 Objetivo (PMV)", -0.5, 0.5, 0.0, step=0.1, help="Rango de confort estricto.")

# 3. Predicción
# El orden de entrenamiento original era: ['tr', 'rh', 'vel', 'met', 'clo', 'pmv_ce']
input_data = pd.DataFrame([[tr, rh, vel, met, clo, pmv]], 
                          columns=['tr', 'rh', 'vel', 'met', 'clo', 'pmv_ce'])

prediccion = modelo.predict(input_data)[0]

# 4. Visualización del Resultado (LÓGICA CORREGIDA)
with col3:
    st.subheader("✅ Setpoint Calculado")
    st.metric(label="Temperatura Aire Interior (Ta)", value=f"{prediccion:.1f} °C")

    # Analizamos la predicción real para dar el mensaje correcto
    if prediccion < 20:
        st.info(f"❄️ **Estrategia de Enfriamiento:** El modelo sugiere una temperatura baja ({prediccion:.1f}°C).")
        if met > 1.5:
            st.markdown("- **Causa:** Alta actividad física detectada.")
        if clo > 1.0:
            st.markdown("- **Causa:** Usuario muy abrigado.")
        if tr < 18:
            st.markdown("- **Contexto:** Las paredes están frías, el sistema se alinea con el entorno.")

    elif prediccion > 25:
        st.warning(f"🔥 **Estrategia de Calor/Ahorro:** El modelo sugiere una temperatura alta ({prediccion:.1f}°C).")
        if clo < 0.5:
            st.markdown("- **Causa:** Usuario con ropa ligera.")
        if tr > 28:
            st.markdown("- **Contexto:** Paredes muy calientes. El modelo evita un choque térmico excesivo (Confort Adaptativo).")
    
    else:
        st.success(f"🍃 **Confort Estándar:** El sistema mantiene una temperatura equilibrada ({prediccion:.1f}°C).")

st.divider()

# 5. Mapa de Calor (Actividad vs Ropa)
# Este es el gráfico más relevante para ESTE modelo
st.subheader(f"🗺️ Mapa de Decisión (Personalización)")

plot_col, text_col = st.columns([3, 1])

with plot_col:
    # Crear grilla (Eje X: Met, Eje Y: Clo)
    met_range = np.linspace(0.8, 4.0, 30)
    clo_range = np.linspace(0.3, 1.5, 30)
    met_grid, clo_grid = np.meshgrid(met_range, clo_range)

    num_puntos = met_grid.size
    
    # Matriz de simulación (mantenemos fijas las ambientales)
    sim_matrix = pd.DataFrame({
        'tr': np.full(num_puntos, tr),
        'rh': np.full(num_puntos, rh),
        'vel': np.full(num_puntos, vel),
        'met': met_grid.ravel(),
        'clo': clo_grid.ravel(),
        'pmv_ce': np.full(num_puntos, pmv)
    })
    
    sim_matrix = sim_matrix[['tr', 'rh', 'vel', 'met', 'clo', 'pmv_ce']]

    # Predecir
    temp_flat = modelo.predict(sim_matrix)
    temp_grid = temp_flat.reshape(met_grid.shape)

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 5))
    contour = ax.contourf(met_grid, clo_grid, temp_grid, levels=20, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    cbar.set_label('Setpoint Sugerido (Ta) [°C]', rotation=270, labelpad=15)

    # Punto actual
    ax.scatter(met, clo, color='yellow', s=150, edgecolors='black', label='Usuario Actual', zorder=10)
    
    ax.set_xlabel('Actividad (met)')
    ax.set_ylabel('Ropa (clo)')
    ax.set_title(f'Setpoint según Usuario (con Paredes a {tr}°C)')
    ax.legend()
    
    st.pyplot(fig)

with text_col:
    st.markdown("### ¿Por qué cambia la temperatura?")
    st.write("""
    Este gráfico demuestra la capacidad **adaptativa** del modelo ante diferentes usuarios.
    """)
    st.info("""
    - **Esquina Azul (Arriba-Der):** Mucha ropa + Ejercicio = El sistema enfría agresivamente.
    - **Esquina Roja (Abajo-Izq):** Poca ropa + Reposo = El sistema calienta para evitar frío.
    """)