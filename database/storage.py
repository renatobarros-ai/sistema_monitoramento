"""
Gerenciamento de histórico e persistência de dados
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class HistoryManager:
    """Gerenciador de histórico do sistema"""
    
    def __init__(self, storage_file='data/history.json', clear_on_init=False):
        self.storage_file = storage_file
        self.ensure_storage_exists()
        if clear_on_init:
            with open(self.storage_file, 'w') as f:
                json.dump([], f)
    
    def ensure_storage_exists(self):
        """Garante que o arquivo de storage existe"""
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w') as f:
                json.dump([], f)
    
    def add_record(self, data: Dict):
        """Adiciona novo registro ao histórico"""
        try:
            # Carrega dados existentes
            with open(self.storage_file, 'r') as f:
                history = json.load(f)
            
            # Adiciona timestamp se não existir
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
            
            # Adiciona novo registro
            history.append(data)
            
            # Salva dados atualizados
            with open(self.storage_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Erro ao salvar registro: {e}")
            return False
    
    def get_latest_record(self) -> Optional[Dict]:
        """Retorna o registro mais recente"""
        try:
            with open(self.storage_file, 'r') as f:
                history = json.load(f)
            
            if history:
                return history[-1]
            return None
        except Exception as e:
            print(f"Erro ao carregar último registro: {e}")
            return None
    
    def get_all_records(self) -> List[Dict]:
        """Retorna todo o histórico"""
        try:
            with open(self.storage_file, 'r') as f:
                history = json.load(f)
            return history
        except Exception as e:
            print(f"Erro ao carregar histórico: {e}")
            return []
    
    def get_recent_records(self, limit: int = 15) -> List[Dict]:
        """Retorna os últimos N registros"""
        try:
            with open(self.storage_file, 'r') as f:
                history = json.load(f)
            
            return history[-limit:] if len(history) > limit else history
        except Exception as e:
            print(f"Erro ao carregar registros recentes: {e}")
            return []
