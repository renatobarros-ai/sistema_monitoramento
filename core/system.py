"""
Sistema principal de monitoramento
"""
import random
import os
from core.sensor import RainSensor
from core.image_analyzer import ImageAnalyzer
from core.classifier import EmergencyClassifier
from utils.file_utils import list_images
from config.settings import IMAGES_PATH

class MonitoringSystem:
    """Sistema principal de monitoramento de alagamentos"""
    
    def __init__(self):
        self.day = 1
        self.rain_sensor = RainSensor()
        self.image_analyzer = ImageAnalyzer()
        self.classifier = EmergencyClassifier()
        self.images = list_images(IMAGES_PATH)
        
        print(f"📷 {len(self.images)} imagens disponíveis")
    
    def process_day(self):
        """Processa um dia completo de monitoramento"""
        # 1. Lê sensor de chuva
        rain_level = self.rain_sensor.get_rain_level()
        
        # 2. Verifica alagamento
        has_flooding = self.rain_sensor.has_flooding(rain_level)
        
        # 3. Analisa imagem se houver alagamento
        image_used = "nenhuma"
        people_analysis = "AUSENTE"
        
        if has_flooding:
            if self.images:
                selected_image = random.choice(self.images)
                image_used = os.path.basename(selected_image)
                people_analysis = self.image_analyzer.analyze_image(selected_image)
            else:
                people_analysis = self.image_analyzer._simulate_analysis()
        
        # 4. Classifica situação
        situation = self.classifier.classify_situation(has_flooding, people_analysis)
        
        # 5. Retorna resultado
        return {
            'day': self.day,
            'rain_level': rain_level,
            'image_used': image_used,
            **situation
        }
    
    def increment_day(self):
        """Incrementa contador de dias"""
        self.day += 1
