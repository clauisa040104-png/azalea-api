from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model = tf.keras.models.load_model('azalea_model.h5')

# Clases
labels = ['botrytis', 'mancha_foliar', 'no_azalea', 'oidio', 'phytophthora', 'roya_en_hoja', 'sana']

@app.get("/")
def root():
    return {"mensaje": "AzaleaCare API funcionando 🌸"}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediccion = model.predict(img_array)
    clase_index = np.argmax(prediccion)
    confianza = float(np.max(prediccion))
    
    return {
        "enfermedad": labels[clase_index],
        "confianza": round(confianza * 100, 2)
    }
