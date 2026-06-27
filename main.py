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
=== ENFERMEDADES ===
- Oidio: polvo blanco pulverulento en hojas y tallos. Agente: Erysiphe spp. Tratamiento: fungicida azufre o bicarbonato potásico cada 7-10 días, mejorar ventilación, eliminar hojas afectadas.
- ManchaFoliar: manchas marrones/negras necróticas en hojas. Agente: Cercospora spp., Pestalotiopsis spp. Tratamiento: fungicida cúprico cada 10-14 días, eliminar hojas afectadas, mejorar drenaje.
- RoyaEnHoja: pústulas naranjas-rojizas en envés de hojas. Agente: Phragmidium spp. Tratamiento: fungicida sistémico (tebuconazol), eliminar hojas afectadas, evitar mojar follaje.
- Phytophthora: pudrición de raíces y cuello, marchitamiento, tallo negro. Agente: Phytophthora cinnamomi. Tratamiento: fosetil-aluminio o metalaxil, mejorar drenaje, replantar en sustrato nuevo si es grave.
- Botrytis: moho gris en flores, hojas y tallos. Agente: Botrytis cinerea. Tratamiento: fungicida específico (iprodiona, pirimetanil) cada 7 días, ventilación, eliminar partes afectadas.
- PlantaSana: sin signos de enfermedad. Recomendación: cuidado preventivo, revisión periódica, fungicida preventivo al inicio de época lluviosa.
- NoAzalea: imagen no corresponde a azalea. El sistema solo diagnostica azaleas.

=== SÍNTOMAS POR ENFERMEDAD ===
- Oidio: polvo blanco, hojas deformadas, amarillamiento.
- ManchaFoliar: manchas marrones/amarillas, caída prematura de hojas.
- RoyaEnHoja: pústulas naranjas, amarillamiento, caída de hojas.
- Phytophthora: raíces podridas marrones, marchitamiento, tallo negro, amarillamiento.
- Botrytis: moho gris algodonoso, flores podridas, manchas marrones.

=== RIEGO ===
- Verano: cada 2-3 días. Invierno: cada 5-7 días. Primavera/Otoño: cada 3-4 días.
- Regar abundantemente hasta que drene. Esperar que los primeros 2-3 cm de sustrato estén secos.
- Usar agua sin cloro o agua de lluvia a temperatura ambiente.
- No regar por aspersión, regar en la base. No regar por la tarde o noche.
- Son muy sensibles tanto a la sequía como al exceso de agua.

=== ILUMINACIÓN ===
- 4-6 horas de luz indirecta o filtrada por día.
- Evitar sol directo del mediodía, quema las hojas.
- Ideal: bajo árboles caducifolios o en orientación este/norte.
- En interiores: cerca de ventanas con luz filtrada.
- En climas frescos pueden tolerar más sol.

=== TEMPERATURA ===
- Óptima: 15-21°C durante el día, ligeramente más fresco por la noche.
- Mínima: 5°C. Máxima: 25°C.
- No toleran heladas prolongadas ni calor extremo.
- Las temperaturas frescas de otoño-invierno estimulan la floración.
- Evitar corrientes de aire frío directo.

=== ABONO Y NUTRIENTES ===
- Fertilizante específico para plantas acidófilas (azaleas, rododendros, camelias).
- Rico en hierro y magnesio. pH del sustrato ideal: 4.5-6.0.
- Frecuencia: cada 15-30 días en primavera y verano.
- Suspender en otoño e invierno para que la planta descanse.
- No abonar durante la floración ni cuando la planta está estresada.
- El abono genérico puede dañarlas por alterar el pH.

=== PODA ===
- Época correcta: inmediatamente después de la floración (generalmente primavera).
- NUNCA podar en otoño o invierno: se eliminan los brotes florales del próximo año.
- Tipos: mantenimiento (ramas secas/dañadas), formación (dar forma), rejuvenecimiento (corte drástico en plantas viejas).
- Usar tijeras limpias y afiladas. Desinfectar con alcohol 70% entre plantas.

=== TRASPLANTE ===
- Frecuencia: cada 2-3 años cuando las raíces salen por los agujeros de drenaje.
- Época: primavera, después de la floración.
- Maceta nueva: 2-3 cm más grande que la anterior.
- Sustrato: mezcla específica para acidófilas o turba con perlita.
- No compactar demasiado el sustrato. Regar bien después del trasplante.
- No abonar hasta 2-3 meses después del trasplante.

=== FLORACIÓN ===
- Época: principalmente primavera (marzo-mayo), algunas variedades en otoño-invierno.
- Duración: 3-6 semanas según variedad y condiciones.
- Para estimular floración: temperaturas frescas en otoño-invierno (10-15°C).
- No mover la planta cuando tiene botones florales, puede perderlos.
- Retirar flores marchitas para prolongar floración y evitar enfermedades.
- Colores: blanco, rosa, rojo, lila, naranja, amarillo según variedad.

=== PLAGAS ===
- Araña roja (Tetranychus urticae): puntitos amarillos en hojas, telarañas finas. Tratamiento: acaricida, aumentar humedad ambiental, ducha de agua a las hojas.
- Pulgones (Aphididae): hojas enrolladas, pegajosas, deformadas. Tratamiento: jabón potásico, insecticida sistémico, eliminar manualmente.
- Cochinilla algodonosa: masas blancas algodonosas en tallos y hojas. Tratamiento: alcohol con algodón, insecticida sistémico.
- Trips: hojas plateadas con puntitos negros. Tratamiento: insecticida específico, trampas pegajosas azules.
- Mosca blanca: nube de insectos blancos al mover la planta. Tratamiento: insecticida, trampas amarillas pegajosas.

=== SUELO Y SUSTRATO ===
- pH ideal: 4.5-6.0 (suelo ácido).
- Sustrato: turba con perlita o sustrato específico para acidófilas.
- Buen drenaje es fundamental, no toleran encharcamiento.
- Evitar suelos calcáreos o con pH alto, produce clorosis férrica.
- Si el suelo es alcalino: acidificar con sulfato de hierro o azufre.

=== COMPATIBILIDAD CON OTRAS PLANTAS ===
COMPATIBLES (mismas necesidades de suelo ácido y sombra parcial):
- Camelia (Camellia japonica): compatibilidad ALTA, mismas necesidades, floración en distintas épocas.
- Hortensia (Hydrangea macrophylla): compatibilidad ALTA, suelo ácido y sombra parcial.
- Rododendro (Rhododendron spp.): compatibilidad MUY ALTA, mismo género botánico, necesidades idénticas.
- Helechos: compatibilidad ALTA, mismas condiciones de sombra y humedad.
- Hostas (Hosta spp.): compatibilidad ALTA, plantas de sombra, cubren el suelo estéticamente.
- Arándanos (Vaccinium spp.): compatibilidad ALTA, mismas necesidades de suelo ácido.
- Pino enano: compatibilidad MODERADA, comparten preferencia por suelo ácido.

INCOMPATIBLES:
- Lavanda (Lavandula spp.): incompatible, prefiere suelo alcalino y sol pleno.
- Geranio (Pelargonium spp.): incompatible, prefiere sol pleno y pH neutro-alcalino.
- Nogal (Juglans spp.): INCOMPATIBLE TOTAL, produce juglona, toxina que mata las azaleas.
- Eucalipto: incompatible, libera sustancias alelopáticas que inhiben otras plantas.
- Rosas: incompatible, prefieren suelo neutro-alcalino y pleno sol.

=== PROBLEMAS COMUNES Y SOLUCIONES ===
- Hojas amarillas: exceso de agua, pH incorrecto, falta de hierro. Solución: revisar riego, acidificar suelo, quelato de hierro.
- Hojas caen sin razón: cambio brusco de temperatura, corrientes de aire, estrés hídrico.
- No florece: poda en mal momento, poca luz, temperatura demasiado cálida en invierno.
- Hojas quemadas en puntas: exceso de sol directo, sales en el agua, exceso de abono.
- Raíces podridas: exceso de riego, mal drenaje. Solución: reducir riego, mejorar drenaje.
- Planta mustia a pesar de riego: posible Phytophthora, revisar raíces.

=== INFORMACIÓN GENERAL DE LA AZALEA ===
- Nombre científico: Rhododendron spp.
- Familia: Ericaceae.
- Origen: Asia (principalmente China, Japón, Corea).
- Tipo: arbusto ornamental perenne o caducifolio según variedad.
- Altura: 0.5-3 metros según variedad.
- Toxicidad: TÓXICA para personas, perros y gatos. Todas las partes son venenosas si se ingieren.
- Longevidad: pueden vivir más de 100 años con cuidados adecuados.
- Variedades populares: Azalea japónica, Azalea indica, Azalea mollis, Azalea kurume.
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
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
            messages=mensajes,
            max_tokens=500,
            temperature=0.7
        )

        return {"respuesta": response.choices[0].message.content}

    except Exception as e:
        return {"respuesta": f"Error al procesar tu consulta: {str(e)}"}
