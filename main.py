from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import io
import os
from pydantic import BaseModel
from owlready2 import get_ontology
import openai

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

CONTEXTO_OWL = """
ENFERMEDADES: Oidio (polvo blanco, fungicida azufre), ManchaFoliar (manchas marrones, fungicida cobre), RoyaEnHoja (pústulas naranjas, fungicida sistémico), Phytophthora (pudrición raíz, fosetil-aluminio), Botrytis (moho gris, fungicida botrytis), PlantaSana (cuidado preventivo), NoAzalea (imagen no es azalea).
RIEGO: cada 2-3 días en verano, 5-7 días en invierno. Agua sin cloro.
ILUMINACION: 4-6 horas luz indirecta. Sin sol directo.
TEMPERATURA: óptima 15-21°C, mínima 5°C, máxima 25°C.
ABONO: fertilizante acidófilo cada 15-30 días en primavera-verano. pH 4.5-6.0.
PODA: después de floración en primavera. Nunca en otoño-invierno.
COMPATIBLES: Camelia, Hortensia, Rododendro, Helechos, Hostas.
INCOMPATIBLES: Lavanda, Geranio, Nogal (produce juglona tóxica).
"""

# ── OpenRouter ────────────────────────────────────────────────
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "")
)

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

# ── Modelos de request ─────────────────────────────────────────
class ChatRequest(BaseModel):
    mensaje: str
    historial: list[dict] = []

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
    try:
        mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in req.historial[-10:]:
            mensajes.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        mensajes.append({"role": "user", "content": req.mensaje})

        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=mensajes,
            max_tokens=500,
            temperature=0.7
        )

        return {"respuesta": response.choices[0].message.content}

    except Exception as e:
        return {"respuesta": f"Error al procesar tu consulta: {str(e)}"}
