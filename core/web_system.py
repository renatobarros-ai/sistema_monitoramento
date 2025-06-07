"""
Extensão web do sistema de monitoramento
"""
from datetime import datetime
from core.system import MonitoringSystem  # Importa a classe original

class WebMonitoringSystem(MonitoringSystem):
    """Sistema de monitoramento integrado com dashboard web"""
    
    def __init__(self, api_url='http://localhost:5000'):
        super().__init__()
        self.api_url = api_url
        
        # Carrega componentes web opcionalmente
        try:
            from database.storage import HistoryManager
            self.history_manager = HistoryManager()
            print("✅ Persistência de dados habilitada")
        except ImportError:
            print("⚠️  Rodando sem persistência de dados")
            self.history_manager = None
    
    def process_day(self):
        """Processa dia com recursos web adicionais"""
        # Executa processamento original
        result = super().process_day()
        
        # Adiciona recursos web
        self._add_web_features(result)
        
        return result
    
    def _add_web_features(self, result):
        """Adiciona funcionalidades web aos resultados"""
        if not self.history_manager:
            return
        
        # Adiciona timestamps
        timestamp = datetime.now()
        result.update({
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime('%d/%m/%Y'),
            'time': timestamp.strftime('%H:%M:%S')
        })
        
        # Salva histórico
        self.history_manager.add_record(result)
        
        # Notifica dashboard
        self._notify_dashboard(result)
    
    def _notify_dashboard(self, data):
        """Notifica dashboard via WebSocket/API"""
        try:
            # Implementação da notificação
            pass
        except Exception as e:
            print(f"Aviso: Não foi possível notificar dashboard: {e}")
