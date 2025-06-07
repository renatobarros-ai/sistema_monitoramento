"""
Analisador de imagens com CNN
"""
import torch
import random
from PIL import Image
import torchvision.transforms as transforms

from models.cnn_model import load_model
from config.settings import MODEL_PATHS, IMAGE_SIZE, SIMULATION_PRESENCE_PROB

class ImageAnalyzer:
    """Analisador de imagens para detecção de pessoas"""
    
    def __init__(self):
        self.model = None
        self.transform = self._setup_transforms()
        self._load_model()
    
    def _setup_transforms(self):
        """Configura transformações de imagem"""
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    def _load_model(self):
        """Carrega modelo CNN"""
        for path in MODEL_PATHS:
            try:
                self.model, checkpoint = load_model(path)
                print(f"✅ Modelo carregado: {path}")
                print(f"   - Acurácia: {checkpoint.get('accuracy', 'N/A'):.1f}%")
                return
            except Exception as e:
                continue
        
        print("⚠️  Nenhum modelo encontrado - usando simulação")
    
    def analyze_image(self, image_path):
        """Analisa imagem e retorna resultado"""
        if self.model is None:
            return self._simulate_analysis()
        
        try:
            # Processa imagem
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Inferência
            with torch.no_grad():
                output = self.model(image_tensor)
                confidence = output.item()
            
            # Interpreta resultado
            has_people = confidence <= 0.5
            result = "PRESENÇA CONFIRMADA" if has_people else "AUSENTE"
            
            print(f"🎯 CNN: {result} (confiança: {confidence:.3f})")
            return result
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return self._simulate_analysis()
    
    def _simulate_analysis(self):
        """Simulação quando modelo não disponível"""
        result = "PRESENÇA CONFIRMADA" if random.random() < SIMULATION_PRESENCE_PROB else "AUSENTE"
        print(f"🎲 Simulação: {result}")
        return result
