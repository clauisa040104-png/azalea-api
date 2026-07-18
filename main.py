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

# Mapeo entre la etiqueta que devuelve el modelo CNN y el individuo de la ontología OWL
LABEL_A_INDIVIDUO_OWL = {
    'botrytis': 'Botrytis',
    'mancha_foliar': 'ManchaFoliar',
    'no_azalea': 'NoAzalea',
    'oidio': 'Oidio',
    'phytophthora': 'Phytophthora',
    'roya_en_hoja': 'RoyaEnHoja',
    'sana': 'PlantaSana',
}

# ── Ontología OWL ──────────────────────────────────────────────
OWL_PATH = os.path.join(BASE_DIR, 'azalea_ontology.owl')
onto = get_ontology(f"file://{OWL_PATH}").load()


def _valor(propiedad):
    """Devuelve el primer valor de una data property de owlready2, o cadena vacía si no existe."""
    return propiedad[0] if propiedad else ''


def consultar_enfermedad(nombre_individuo: str) -> dict:
    """
    Consulta REAL a la ontología OWL: dado el nombre de un individuo de la clase
    Enfermedad, recorre sus object properties (tieneSintoma, tieneTratamiento,
    tieneSeveridad) y data properties (nombreComun, descripcion, agenteCausal)
    para construir la respuesta del diagnóstico a partir del grafo OWL.
    """
    individuo = onto.search_one(iri=f"*{nombre_individuo}")
    if individuo is None:
        return {}

    return {
        "nombreComun": _valor(individuo.nombreComun),
        "descripcion": _valor(individuo.descripcion),
        "agenteCausal": _valor(individuo.agenteCausal),
        "sintomas": [
            _valor(s.descripcion) or s.name for s in individuo.tieneSintoma
        ],
        "tratamientos": [
            _valor(t.descripcion) or t.name for t in individuo.tieneTratamiento
        ],
        "severidadesPosibles": [s.name for s in individuo.tieneSeveridad],
    }


def construir_contexto_desde_ontologia() -> str:
    """
    Genera el contexto de conocimiento del chatbot consultando la ontología OWL
    en tiempo real mediante owlready2 (instances(), object properties y data
    properties), en lugar de un texto fijo escrito a mano.
    """
    partes = ["=== BASE DE CONOCIMIENTO (consultada desde la ontología OWL) ===\n"]

    partes.append("--- ENFERMEDADES ---")
    for enf in onto.Enfermedad.instances():
        nombre = _valor(enf.nombreComun) or enf.name
        desc = _valor(enf.descripcion)
        agente = _valor(enf.agenteCausal) or "No aplica"
        sintomas = ", ".join(_valor(s.descripcion) or s.name for s in enf.tieneSintoma)
        tratamientos = "; ".join(_valor(t.descripcion) or t.name for t in enf.tieneTratamiento)

        bloque = f"- {nombre}: {desc} Agente: {agente}."
        if sintomas:
            bloque += f" Síntomas: {sintomas}."
        if tratamientos:
            bloque += f" Tratamiento: {tratamientos}"
        partes.append(bloque)

    partes.append("\n--- CUIDADOS GENERALES ---")
    for cuidado in onto.CuidadoGeneral.instances():
        tipo = cuidado.is_a[0].name if cuidado.is_a else "Cuidado"
        desc = _valor(cuidado.descripcion)
        extra = []
        if cuidado.frecuenciaRiego:
            extra.append(f"Frecuencia: {_valor(cuidado.frecuenciaRiego)}")
        if cuidado.cantidadAgua:
            extra.append(f"Cantidad: {_valor(cuidado.cantidadAgua)}")
        if cuidado.horasLuz:
            extra.append(f"Horas de luz: {_valor(cuidado.horasLuz)}")
        if cuidado.temperaturaMinima:
            extra.append(f"Min: {_valor(cuidado.temperaturaMinima)}, Max: {_valor(cuidado.temperaturaMaxima)}")
        if cuidado.frecuenciaAbono:
            extra.append(f"Frecuencia abono: {_valor(cuidado.frecuenciaAbono)}")
        if cuidado.epocaPoda:
            extra.append(f"Época: {_valor(cuidado.epocaPoda)}")

        linea = f"- {tipo}: {desc}"
        if extra:
            linea += " " + " | ".join(extra)
        partes.append(linea)

    partes.append("\n--- COMPATIBILIDAD CON OTRAS PLANTAS ---")
    az = onto.AzaleaRhododendron
    compatibles = ", ".join(p.name for p in az.esCompatibleCon)
    incompatibles = ", ".join(p.name for p in az.noEsCompatibleCon)
    partes.append(f"Compatibles: {compatibles}")
    partes.append(f"Incompatibles: {incompatibles}")

    return "\n".join(partes)


# Contexto generado UNA VEZ al iniciar el servidor, consultando la ontología real
CONTEXTO_OWL = construir_contexto_desde_ontologia()

SYSTEM_PROMPT = f"""Eres AzaleaBot, el asistente experto del sistema AzaleaCare.
Tu conocimiento proviene EXCLUSIVAMENTE de la siguiente ontología OWL formal sobre azaleas.
No inventes información que no esté en la ontología.

=== ONTOLOGÍA AZALEACARE ===
{CONTEXTO_OWL}
=== FIN DE ONTOLOGÍA ===

Responde siempre en español, de forma amable, clara y concisa.
Si la pregunta no está relacionada con azaleas o su cuidado, indica amablemente que solo puedes ayudar con temas de azaleas.
Cuando menciones enfermedades, incluye siempre los síntomas y tratamientos de la ontología.
"""

# ── OpenRouter ────────────────────────────────────────────────
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "")
)

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
    etiqueta = labels[clase_index]

    # Consulta en tiempo real a la ontología OWL para enriquecer el diagnóstico
    # del modelo CNN con el razonamiento simbólico (síntomas, agente causal,
    # tratamientos) definido en el grafo OWL.
    individuo_owl = LABEL_A_INDIVIDUO_OWL.get(etiqueta)
    info_ontologia = consultar_enfermedad(individuo_owl) if individuo_owl else {}

    return {
        "enfermedad": etiqueta,
        "confianza": round(confianza * 100, 2),
        "infoOntologia": info_ontologia,
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
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=mensajes,
            max_tokens=500,
            temperature=0.7
        )

        return {"respuesta": response.choices[0].message.content}

    except Exception as e:
        return {"respuesta": f"Error al procesar tu consulta: {str(e)}"}
