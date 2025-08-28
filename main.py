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


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Lê os bytes do arquivo direto
    image_bytes = await file.read()

    # Passa pro YOLO sem salvar
    result = yolo.analyze(image_bytes)
    print(f"results: {result}\n")
    return {"results": result}


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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
