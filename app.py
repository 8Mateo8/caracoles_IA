import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

st.set_page_config(
    page_title="Examen IA",
    layout="centered"
)

st.title("Reconocimiento de Objetos")

st.subheader('Mateo Calderón - Lisseth Guazhambo - Juan Hurtado')

st.markdown('<div style="text-align: justify;">En esta página web se presenta un modelo de reconocimiento de imágenes de 3 categorías.</div>', unsafe_allow_html=True)



def crear_esqueleto_modelo():
    base_model = EfficientNetB7(
        weights='imagenet', 
        include_top=False, 
        input_shape=(600, 600, 3)
    )
    base_model.trainable = False
    model = models.Sequential([
        layers.Input(shape=(600, 600, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),        
        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    return model

@st.cache_resource
def load_model_weights():
    model = crear_esqueleto_modelo()
    output_path = 'caracoles.h5'
    if not os.path.exists(output_path):
        file_id = '1DMe6hBonh9HWL89nzks3tXSHEyblBVAg' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    try:
        model.load_weights(output_path)
    except Exception as e:
        st.error(f"Error: {e}")
        return None     
    return model

with st.spinner('Cargando modelo...'):
    model = load_model_weights()

nombres_clases = ['Burn', 'Skidmark', 'Turbo'] 

file = st.file_uploader("Sube una foto (JPG/PNG)", type=["jpg", "png", "jpeg"])

if file is not None and model is not None:
    image = Image.open(file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    size = (600, 600)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(predictions[0])
    class_name = nombres_clases[class_index]
    confidence = np.max(predictions[0]) * 100
    
    st.success(f"Predicción: **{class_name}**")
    st.info(f"Certeza: **{confidence:.2f}%**")
    st.bar_chart(predictions[0])








