"""
Classificador de situações de emergência
"""
from enum import Enum

class EmergencyLevel(Enum):
    NORMAL = "Normal"
    ATENCAO = "Atenção" 
    PERIGO = "Perigo"

class EmergencyClassifier:
    """Classificador de níveis de emergência"""
    
    def classify_situation(self, has_flooding, people_analysis_result):
        """Classifica situação baseada no alagamento e presença de pessoas"""
        if not has_flooding:
            return {
                'classification': EmergencyLevel.NORMAL.value,
                'people_at_risk': 'Não',
                'flooding': 'Não'
            }
        
        # Há alagamento
        flooding_status = 'Sim'
        
        if people_analysis_result == "PRESENÇA CONFIRMADA":
            return {
                'classification': EmergencyLevel.PERIGO.value,
                'people_at_risk': 'Sim',
                'flooding': flooding_status
            }
        else:
            return {
                'classification': EmergencyLevel.ATENCAO.value,
                'people_at_risk': 'Não',
                'flooding': flooding_status
            }
