"""
Interface de console para exibição dos resultados
"""
import time
from config.settings import DISPLAY_TIME

class ConsoleDisplay:
    """Gerenciador de exibição no console"""
    
    def show_result(self, result):
        """Exibe resultado formatado"""
        print("\n" + "="*50)
        print(f"🌧️  Dia: {result['day']}")
        print(f"📊 Classificação: {result['classification']}")
        print(f"👥 Pessoas em risco: {result['people_at_risk']}")
        print(f"🌊 Alagamento: {result['flooding']}")
        print(f"☔ Nível de Chuva: {result['rain_level']} mm")
        print(f"📷 Imagem: {result['image_used']}")
        print("="*50)
    
    def show_waiting(self, seconds=None):
        """Exibe mensagem de espera"""
        wait_time = seconds or DISPLAY_TIME
        print(f"⏱️  Aguardando {wait_time} segundos...")
        time.sleep(wait_time)
    
    def show_startup(self):
        """Exibe mensagem de inicialização"""
        print("🚀 SISTEMA DE MONITORAMENTO INICIADO")
        print("⏹️  Pressione Ctrl+C para parar\n")
    
    def show_shutdown(self):
        """Exibe mensagem de encerramento"""
        print("\n\n🛑 Sistema parado!")
