from fastapi.middleware.cors import CORSMiddleware
import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf

from model.ModelYOLO import ModelYOLO

app = FastAPI()
yolo = ModelYOLO()  # carrega uma vez na inicialização

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Diretório para salvar uploads
SAVE_DIR = "uploads"
os.makedirs(SAVE_DIR, exist_ok=True)

# Caminho do modelo TFLite
MODEL_PATH = "best_saved_model/best_float32.tflite"

# Carrega o modelo TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Obtém detalhes das entradas e saídas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determina o tamanho de entrada automaticamente
input_shape = input_details[0]['shape']  # ex: [1, 640, 640, 3]
target_size = (input_shape[2], input_shape[1])  # largura x altura

# Lista de classes do modelo
class_names = [
    "anadenanthera", "arrabidaea", "cecropia", "combretum", "dipteryx",
    "faramea", "mabea", "mimosa", "protium", "schinus", "serjania", "tridax",
    "arecaceae", "chromolaena", "croton", "eucalipto", "hyptis", "matayba",
    "myrcia", "qualea", "senegalia", "syagrus", "urochloa"
]


def preprocess_image(image: Image.Image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)
    array = np.array(image, dtype=np.float32) / 255.0  # float32
    array = np.expand_dims(array, axis=0)  # Adiciona batch dimension
    return array


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Lê os bytes do arquivo direto
    image_bytes = await file.read()

    # Passa pro YOLO sem salvar
    result = yolo.analyze(image_bytes)
    print(f"results: {result}\n")
    return {"results": result}

# @app.post("/analyze")
# async def analyze_image(file: UploadFile = File(...)):
#     # Lê a imagem
#     image_bytes = await file.read()

#     # Salva a imagem para teste
#     # save_path = os.path.join(SAVE_DIR, file.filename)
#     # with open(save_path, "wb") as f:
#     #     f.write(image_bytes)

#     # Converte para PIL Image
#     image = Image.open(io.BytesIO(image_bytes))

#     # Preprocessa
#     input_array = preprocess_image(image, target_size)

#     # Define o tensor de entrada
#     interpreter.set_tensor(input_details[0]['index'], input_array)

#     # Executa inferência
#     interpreter.invoke()

#     # Obtém o resultado
#     output_data = interpreter.get_tensor(output_details[0]['index'])[
#         0]  # shape: (23,)

#     # Cria dicionário com classes e suas probabilidades
#     results = {
#         class_name: float(prob)
#         for class_name, prob in zip(class_names, output_data)
#         if prob >= 0.05
#     }

#     # Ordena do maior para o menor
#     results = dict(
#         sorted(results.items(), key=lambda item: item[1], reverse=True))

#     print("Resultados da classificação:", results)

#     return JSONResponse(content=results)


@app.post("/test")
async def analyze_image_test(file: UploadFile = File(...)):
    # Lê a imagem como bytes
    image_bytes = await file.read()

    # save_path = os.path.join(SAVE_DIR, file.filename)
    # with open(save_path, "wb") as f:
    #     f.write(image_bytes)

    # Converte para PIL Image se necessário
    image = Image.open(io.BytesIO(image_bytes))

    # Aqui você roda seu modelo de IA
    # Exemplo fictício: resultado = model.predict(image)
    resultado = {"classe": "abelha", "conf": 0.95}

    # Retorna JSON com informações da análise, não os bytes
    return JSONResponse(content=resultado)


# @app.post("/classify", response_model=ClassificationResponse)
# async def classify_text(file: UploadFile = File(...)):

#     return ClassificationResponse(
#         classification=classification,
#         confidence=confidence,
#         timestamp=datetime.datetime.now().isoformat()
#     )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
