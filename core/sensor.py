"""
Simulador do sensor de chuva
"""
import random
from config.settings import RAIN_THRESHOLD

class RainSensor:
    """Simulador de sensor de chuva"""
    
    def __init__(self):
        self.threshold = RAIN_THRESHOLD
    
    def get_rain_level(self):
        """Simula leitura do sensor (1-100mm)"""
        return random.randint(1, 100)
    
    def has_flooding(self, rain_level):
        """Verifica se há alagamento baseado no nível de chuva"""
        return rain_level >= self.threshold
