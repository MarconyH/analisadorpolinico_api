from ultralytics import YOLO
import numpy as np
import cv2
import os


class ModelYOLO:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "best.pt")
        # Otimizações para usar menos memória
        self.model = YOLO(model_path,
                         task="classify", 
                         verbose=False)
        # Força garbage collection após carregar o modelo
        import gc
        gc.collect()

    def analyze(self, image_bytes: bytes, save: bool = False, save_path: str = None):
        """
        Analisa a imagem usando YOLO.
        - image_bytes: bytes da imagem (ex.: UploadFile.read())
        - save: se True, salva a imagem no disco
        - save_path: caminho opcional para salvar
        """

        # Converte bytes -> array numpy (OpenCV)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # (Opcional) salvar imagem
        if save:
            if save_path is None:
                save_path = "upload.jpg"
            cv2.imwrite(save_path, img)

        # Predição
        results = self.model.predict(img, verbose=False)

        response = []
        for r in results:
            probs = r.probs
            if probs is not None:
                for class_id, prob in enumerate(probs.data.tolist()):
                    if prob > 0.1:  # filtra classes pouco relevantes
                        response.append({
                            "class": self.model.names[class_id],
                            "probability": float(prob)
                        })

        response.sort(key=lambda x: x["probability"], reverse=True)
        return response
