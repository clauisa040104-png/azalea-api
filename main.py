from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import io
import os
from pydantic import BaseModel
from owlready2 import get_ontology
import google.generativeai as genai

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modelo TFLite ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
interpreter = tflite.Interpreter(model_path=os.path.join(BASE_DIR, 'azalea_model.tflite'))
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
labels = ['botrytis', 'mancha_foliar', 'no_azalea', 'oidio', 'phytophthora', 'roya_en_hoja', 'sana']

# ── Ontología OWL ──────────────────────────────────────────────
OWL_PATH = os.path.join(BASE_DIR, 'azalea_ontology.owl')
onto = get_ontology(f"file://{OWL_PATH}").load()

def extraer_conocimiento_owl() -> str:
    """Lee la ontología y la convierte en texto para el contexto de Gemini."""
    lineas = []
    for ind in onto.individuals():
        tipos = [c.name for c in ind.is_a if hasattr(c, 'name')]
        props = {}
        for prop in ind.get_properties():
            valores = prop[ind]
            if valores:
                props[prop.name] = [str(v) for v in valores]
        if props or tipos:
            lineas.append(f"\n[{ind.name}] tipo={tipos}")
            for k, v in props.items():
                lineas.append(f"  {k}: {'; '.join(v)}")
    return "\n".join(lineas)

CONTEXTO_OWL = extraer_conocimiento_owl()

# ── Gemini ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = f"""Eres AzaleaBot, el asistente experto del sistema AzaleaCare.
Tu conocimiento proviene EXCLUSIVAMENTE de la siguiente ontología OWL formal sobre azaleas.
No inventes información que no esté en la ontología.

=== ONTOLOGÍA AZALEARCARE ===
{CONTEXTO_OWL}
=== FIN DE ONTOLOGÍA ===

Responde siempre en español, de forma amable, clara y concisa.
Si la pregunta no está relacionada con azaleas o su cuidado, indica amablemente que solo puedes ayudar con temas de azaleas.
Cuando menciones enfermedades, incluye siempre los síntomas y tratamientos de la ontología.
"""

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

# ── Modelos de request ─────────────────────────────────────────
class ChatRequest(BaseModel):
    mensaje: str
    historial: list[dict] = []  # [{"role": "user"/"model", "content": "..."}]

# ── Endpoints ──────────────────────────────────────────────────
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

@app.post("/chatbot")
async def chatbot(req: ChatRequest):
    # Construir historial para Gemini
    historial_gemini = []
    for msg in req.historial[-10:]:
        historial_gemini.append({
            "role": msg["role"],
            "parts": [msg["content"]]
        })

    chat = gemini_model.start_chat(history=historial_gemini)
    respuesta = chat.send_message(req.mensaje)

    return {
        "respuesta": respuesta.text
    }
