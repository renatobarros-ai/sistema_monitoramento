"""
Arquivo principal do sistema de monitoramento de alagamentos
"""
from core.system import MonitoringSystem
from display.console_display import ConsoleDisplay

def main():
    """Função principal"""
    # Inicializa componentes
    system = MonitoringSystem()
    display = ConsoleDisplay()
    
    # Exibe inicialização
    display.show_startup()
    
    try:
        while True:
            # Processa um dia
            result = system.process_day()
            
            # Exibe resultado
            display.show_result(result)
            
            # Aguarda
            display.show_waiting()
            
            # Incrementa dia
            system.increment_day()
            
    except KeyboardInterrupt:
        display.show_shutdown()

if __name__ == "__main__":
    main()
