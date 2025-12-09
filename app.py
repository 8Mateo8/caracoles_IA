import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Clasificador de Imágenes IA",
    page_icon="🤖",
    layout="centered"
)

# --- TITULO Y DESCRIPCIÓN ---
st.title("🤖 Reconocimiento de Imágenes con EfficientNetB7")
st.write("Sube una imagen y el modelo te dirá a qué clase pertenece.")

# --- CARGAR EL MODELO (Con Caché) ---
# Usamos @st.cache_resource para que Streamlit cargue el modelo UNA sola vez en memoria.
# Si no usamos esto, el modelo se recargaría cada vez que subes una foto (muy lento).
@st.cache_resource
def load_model():
    # Nombre del archivo local
    output_path = 'mi_modelo_b7.h5'
    
    # Si el archivo no existe, lo descargamos de Drive
    if not os.path.exists(output_path):
        # ¡¡¡PEGA AQUÍ EL ID DE TU ARCHIVO DE DRIVE!!!
        file_id = '14J3hAIrG43OSrmPu1oxH-vbaAoIt1IJU' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    
    # Cargamos el modelo
    model = tf.keras.models.load_model(output_path)
    return model

# Ejecutamos la función de carga. Aparecerá un spinner mientras carga.
with st.spinner('Cargando modelo inteligente...'):
    model = load_model()

# --- DEFINIR LAS CLASES ---
# IMPORTANTE: Deben estar en el mismo orden alfabético que tus carpetas en Drive.
# Ejemplo: Si tus carpetas eran 'Gatos', 'Perros', 'Ratones', ponlos así.
nombres_clases = ['Clase_A', 'Clase_B', 'Clase_C'] # <--- ¡CAMBIA ESTO!

# --- SUBIDA DE IMAGEN ---
file = st.file_uploader("Por favor sube una imagen (JPG o PNG)", type=["jpg", "png", "jpeg"])

# --- LÓGICA DE PREDICCIÓN ---
if file is not None:
    # 1. Mostrar la imagen subida al usuario
    image = Image.open(file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # 2. Preprocesar la imagen para que el modelo la entienda
    # El modelo espera una imagen de (600, 600) píxeles.
    size = (600, 600) 
    
    # Usamos ImageOps.fit para recortar el centro y redimensionar sin deformar mucho
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convertimos la imagen a un array de números (matriz)
    img_array = np.asarray(image)
    
    # El modelo espera un lote de imágenes (batch), no una sola.
    # Convertimos la forma de (600, 600, 3) a (1, 600, 600, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # NOTA: No dividimos por 255 aquí porque tu modelo ya tiene 
    # la capa 'layers.Lambda(preprocess_input)' adentro.
    
    # 3. Realizar la predicción
    predictions = model.predict(img_array)
    
    # 'predictions' es un array de probabilidades, ej: [0.1, 0.8, 0.1]
    # np.argmax nos dice la posición del valor más alto (en este caso, posición 1)
    score = tf.nn.softmax(predictions[0]) # Convertimos a porcentajes legibles
    class_index = np.argmax(predictions[0])
    
    class_name = nombres_clases[class_index]
    confidence = np.max(predictions[0]) * 100 # Confianza en %

    # 4. Mostrar resultados
    st.write("---")
    st.success(f"Predicción: **{class_name}**")
    st.info(f"Probabilidad de certeza: **{confidence:.2f}%**")
    
    # (Opcional) Mostrar gráfico de barras con las probabilidades de todas las clases
    st.write("Detalle de probabilidades:")
    st.bar_chart(data=predictions[0])