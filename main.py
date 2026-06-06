from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite
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

# Cargar modelo .tflite
interpreter = tflite.Interpreter(model_path='azalea_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['botrytis', 'mancha_foliar', 'no_azalea', 'oidio', 'phytophthora', 'roya_en_hoja', 'sana']

@app.get("/")
def root():
    return {"mensaje": "AzaleaCare API funcionando 🌸"}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    clase_index = np.argmax(output)
    confianza = float(np.max(output))

    return {
        "enfermedad": labels[clase_index],
        "confianza": round(confianza * 100, 2)
    }
