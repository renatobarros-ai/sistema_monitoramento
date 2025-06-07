"""
Configurações do sistema de monitoramento
"""
import os

# Caminhos dos arquivos
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATHS = [
    os.path.join(BASE_DIR, "data/cnn/modelo/modelo_treinado.pth"),
    os.path.join(BASE_DIR, "modelo/detector.pth"),
    "cnn/modelo/modelo_treinado.pth"
]
IMAGES_PATH = os.path.join(BASE_DIR, "data/cnn/imagens")

# Configurações do modelo
IMAGE_SIZE = 64
DEVICE = "cpu"

# Configurações do sistema
RAIN_THRESHOLD = 50  # mm de chuva para considerar alagamento
DISPLAY_TIME = 15    # segundos para exibir resultado
SIMULATION_PRESENCE_PROB = 0.3  # Probabilidade de simular presença

# Configurações de saída
SHOW_DEBUG = True
SHOW_CONFIDENCE = True
