"""
Sistema principal com integração web
"""
import threading
import time
from core.web_system import WebMonitoringSystem
from display.console_display import ConsoleDisplay
from api.app import app, socketio, broadcast_update

def run_monitoring_system():
    """Executa sistema de monitoramento em thread separada"""
    system = WebMonitoringSystem()
    display = ConsoleDisplay()
    
    display.show_startup()
    
    try:
        while True:
            # Processa um dia
            result = system.process_day()
            
            # Exibe no console
            display.show_result(result)
            
            # Envia para dashboard via WebSocket
            broadcast_update(result)
            
            # Aguarda
            display.show_waiting()
            
            # Incrementa dia
            system.increment_day()
            
    except KeyboardInterrupt:
        display.show_shutdown()

def main():
    """Função principal - roda sistema e web em paralelo"""
    print("🚀 Iniciando Sistema Completo (Console + Web)")
    
    # Thread para sistema de monitoramento
    monitoring_thread = threading.Thread(target=run_monitoring_system, daemon=True)
    monitoring_thread.start()
    
    # Servidor web (thread principal)
    print("🌐 Dashboard disponível em: http://localhost:5000")
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
