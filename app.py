import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
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

st.title("🤖 Reconocimiento de Imágenes con EfficientNetB7")
st.write("Sube una imagen y el modelo te dirá a qué clase pertenece.")

# --- DEFINIR EL ESQUELETO DEL MODELO MANUALMENTE ---
# Esto evita el error de "Layer expects 1 input but received 2"
# porque construimos la estructura limpia desde cero.
def crear_esqueleto_modelo():
    # 1. Definir la base igual que en el entrenamiento
    base_model = EfficientNetB7(
        weights='imagenet', 
        include_top=False, 
        input_shape=(600, 600, 3)
    )
    base_model.trainable = False # No importa para inferencia, pero buena práctica

    # 2. Reconstruir la estructura Secuencial EXACTA
    model = models.Sequential([
        layers.Input(shape=(600, 600, 3)),
        # Incluimos las capas de aumentación para que la estructura coincida 
        # con los pesos guardados (aunque no se usan al predecir)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax') # Tus 3 clases
    ])
    return model

@st.cache_resource
def load_model_weights():
    # 1. Crear el modelo vacío (el esqueleto)
    model = crear_esqueleto_modelo()
    
    # 2. Descargar el archivo de pesos (.h5)
    output_path = 'mi_modelo_b7.h5'
    if not os.path.exists(output_path):
        # --- PEGA AQUÍ TU ID DE GOOGLE DRIVE ---
        file_id = '14J3hAIrG43OSrmPu1oxH-vbaAoIt1IJU' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    
    # 3. Cargar SOLO los pesos en el esqueleto
    # Usamos 'load_weights' en lugar de 'load_model'. 
    # Esto evita el error de compilación de Keras 3.
    try:
        model.load_weights(output_path)
    except Exception as e:
        st.error(f"Hubo un error cargando los pesos: {e}")
        return None
        
    return model

# Ejecutamos la carga
with st.spinner('Cargando cerebro de la IA...'):
    model = load_model_weights()

# --- DEFINIR LAS CLASES ---
# ¡CAMBIA ESTO POR LOS NOMBRES REALES DE TUS CARPETAS!
nombres_clases = ['Burn', 'Skidmark', 'Turbo'] 

# --- INTERFAZ DE USUARIO ---
file = st.file_uploader("Sube una foto (JPG/PNG)", type=["jpg", "png", "jpeg"])

if file is not None and model is not None:
    image = Image.open(file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # Procesamiento
    size = (600, 600)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(predictions[0])
    
    class_name = nombres_clases[class_index]
    confidence = np.max(predictions[0]) * 100
    
    st.write("---")
    st.success(f"Predicción: **{class_name}**")
    st.info(f"Certeza: **{confidence:.2f}%**")
    st.bar_chart(predictions[0])

