# ⚙️ Configuração e Personalização

## 📋 Visão Geral

Este documento detalha todas as opções de configuração disponíveis no Sistema de Monitoramento, desde ajustes básicos até personalizações avançadas. O sistema foi projetado para ser altamente configurável e extensível.

## 🔧 Configuração Principal

### 1. **Arquivo `config/settings.py`**

O arquivo principal de configuração contém todas as constantes e parâmetros do sistema:

```python
"""
Configurações do sistema de monitoramento
"""
import os

# === CAMINHOS DOS ARQUIVOS ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Caminhos do modelo CNN (em ordem de prioridade)
MODEL_PATHS = [
    os.path.join(BASE_DIR, "data/cnn/modelo/modelo_treinado.pth"),
    os.path.join(BASE_DIR, "modelo/detector.pth"),
    "cnn/modelo/modelo_treinado.pth"  # Caminho legado
]

# Diretório das imagens para inferência
IMAGES_PATH = os.path.join(BASE_DIR, "data/cnn/imagens")

# === CONFIGURAÇÕES DO MODELO ===
IMAGE_SIZE = 64                    # Resolução das imagens (64x64)
DEVICE = "cpu"                     # Dispositivo: "cpu" ou "cuda"

# === CONFIGURAÇÕES DO SISTEMA ===
RAIN_THRESHOLD = 50                # mm de chuva para considerar alagamento
DISPLAY_TIME = 15                  # segundos para exibir cada resultado
SIMULATION_PRESENCE_PROB = 0.3     # Probabilidade de simular presença (0.0-1.0)

# === CONFIGURAÇÕES DE DEBUG ===
SHOW_DEBUG = True                  # Mostrar informações de debug
SHOW_CONFIDENCE = True             # Mostrar valores de confiança CNN
```

### 2. **Parâmetros Configuráveis**

#### **Limiar de Alagamento**
```python
# Ajustar sensibilidade de detecção de alagamento
RAIN_THRESHOLD = 30    # Mais sensível (detecta com menos chuva)
RAIN_THRESHOLD = 50    # Padrão balanceado
RAIN_THRESHOLD = 70    # Menos sensível (apenas chuvas intensas)
```

#### **Tempo de Ciclo**
```python
# Controlar velocidade de simulação
DISPLAY_TIME = 5      # Ciclos rápidos (5 segundos)
DISPLAY_TIME = 15     # Padrão (15 segundos)
DISPLAY_TIME = 60     # Ciclos lentos (1 minuto)
```

#### **Probabilidade de Simulação**
```python
# Quando modelo CNN não disponível
SIMULATION_PRESENCE_PROB = 0.1    # Raramente detecta pessoas (10%)
SIMULATION_PRESENCE_PROB = 0.3    # Padrão balanceado (30%)
SIMULATION_PRESENCE_PROB = 0.7    # Frequentemente detecta pessoas (70%)
```

#### **Resolução do Modelo**
```python
# Deve corresponder ao treinamento
IMAGE_SIZE = 64       # Padrão (mais rápido)
IMAGE_SIZE = 128      # Maior resolução (mais lento, mais preciso)
IMAGE_SIZE = 224      # Alta resolução (requer re-treinamento)
```

## 🌐 Configuração Web

### 1. **Servidor Flask**

#### **Arquivo `api/app.py`**
```python
# Configurações do servidor
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['DEBUG'] = False  # True para desenvolvimento

# WebSocket
socketio = SocketIO(app, cors_allowed_origins="*")

# Servidor
if __name__ == '__main__':
    socketio.run(
        app, 
        debug=True,           # False para produção
        host='0.0.0.0',      # '127.0.0.1' apenas local
        port=5000            # Porta do servidor
    )
```

#### **Variáveis de Ambiente**
```bash
# Configuração via ambiente
export FLASK_ENV=development     # development | production
export FLASK_DEBUG=1            # 0 | 1
export SECRET_KEY=sua-chave-secreta
export PORT=5000                # Porta do servidor
export HOST=0.0.0.0            # Interface de bind
```

### 2. **Configuração de CORS**
```python
# Para permitir acesso de domínios específicos
socketio = SocketIO(app, cors_allowed_origins=[
    "http://localhost:3000",
    "https://seu-dominio.com"
])

# Para desenvolvimento local
socketio = SocketIO(app, cors_allowed_origins="*")
```

### 3. **Configuração de Paths**
```python
# Customizar caminhos web
template_folder = '/caminho/personalizado/templates'
static_folder = '/caminho/personalizado/static' 

app = Flask(__name__, 
           template_folder=template_folder,
           static_folder=static_folder)
```

## 💾 Configuração de Persistência

### 1. **Histórico JSON**

#### **Arquivo `database/storage.py`**
```python
class HistoryManager:
    def __init__(self, storage_file='data/history.json'):
        self.storage_file = storage_file
        
        # Configurações
        self.max_records = 10000      # Máximo de registros
        self.backup_interval = 100    # Backup a cada N registros
        self.auto_cleanup = True      # Limpeza automática
```

#### **Configurações Avançadas**
```python
# Rotação de arquivos
class AdvancedHistoryManager(HistoryManager):
    def __init__(self, storage_file='data/history.json'):
        super().__init__(storage_file)
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.backup_count = 5  # Manter 5 backups
        
    def rotate_if_needed(self):
        if os.path.getsize(self.storage_file) > self.max_file_size:
            # Implementar rotação
            pass
```

### 2. **Configuração de Backup**
```python
# Backup automático
import shutil
from datetime import datetime

def backup_history():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'data/history_backup_{timestamp}.json'
    shutil.copy('data/history.json', backup_file)
    
# Agendar backup diário
import schedule
schedule.every().day.at("03:00").do(backup_history)
```

## 🎨 Customização da Interface

### 1. **Dashboard Web**

#### **CSS Customizado (`web/static/css/dashboard.css`)**
```css
/* Personalizar cores do tema */
:root {
    --primary-color: #0066CC;        /* Azul padrão */
    --success-color: #78BF43;        /* Verde */
    --warning-color: #FFA500;        /* Laranja */
    --danger-color: #FF4444;         /* Vermelho */
    --background-color: #f8f9fa;     /* Fundo */
}

/* Tema escuro */
body.dark-theme {
    --background-color: #1a1a1a;
    --text-color: #ffffff;
}

/* Personalizar cards */
.card {
    border-radius: 12px;            /* Bordas mais arredondadas */
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);  /* Sombra suave */
}
```

#### **JavaScript Customizado (`web/static/js/dashboard.js`)**
```javascript
// Configurações do dashboard
const DASHBOARD_CONFIG = {
    updateInterval: 30000,          // 30 segundos
    chartAnimationDuration: 1000,   // 1 segundo
    maxHistoryPoints: 50,           // Máximo de pontos no gráfico
    enableNotifications: true,      // Notificações do browser
    autoRefresh: true              // Auto-refresh quando desconectado
};

// Personalizar gráficos
const chartConfig = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        title: {
            display: true,
            text: 'Histórico de Chuva - Personalizado'
        }
    }
};
```

### 2. **Mapas Dinâmicos**

#### **Customizar Imagens de Mapa**
```bash
# Substituir mapas padrão
cp sua_imagem_normal.jpg web/static/images/maps/normal.jpeg
cp sua_imagem_atencao.jpg web/static/images/maps/atencao.jpeg  
cp sua_imagem_perigo.jpg web/static/images/maps/perigo.jpeg
```

#### **Configuração de Mapas Interativos**
```javascript
// Adicionar mapas interativos (Leaflet)
const mapConfig = {
    center: [-30.0346, -51.2177],   // Porto Alegre
    zoom: 10,
    layers: {
        normal: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        satellite: 'https://server.arcgisonline.com/...'
    }
};
```

## 🧠 Configuração do Modelo CNN

### 1. **Caminhos de Modelo**

#### **Adicionar Novos Caminhos**
```python
# Em config/settings.py
MODEL_PATHS = [
    os.path.join(BASE_DIR, "data/cnn/modelo/modelo_treinado.pth"),
    os.path.join(BASE_DIR, "models/production/best_model.pth"),
    os.path.join(BASE_DIR, "models/backup/model_v2.pth"),
    "/shared/models/flood_detector.pth",  # Caminho compartilhado
    "~/models/personal_model.pth"         # Caminho do usuário
]
```

#### **Configuração de Device**
```python
import torch

# Detecção automática
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Forçar CPU
DEVICE = "cpu"

# GPU específica
DEVICE = "cuda:0"  # Primeira GPU
DEVICE = "cuda:1"  # Segunda GPU

# Apple Silicon
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

### 2. **Configuração de Inferência**

#### **Otimizações de Performance**
```python
# Em models/cnn_model.py
class OptimizedCNN(CNN):
    def __init__(self):
        super().__init__()
        # Configurações específicas
        torch.set_num_threads(4)  # Threads CPU
        
    def forward(self, x):
        # Inferência otimizada
        with torch.inference_mode():  # Mais rápido que no_grad
            return super().forward(x)
```

#### **Configuração de Batch Processing**
```python
class BatchImageAnalyzer(ImageAnalyzer):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
    
    def analyze_batch(self, image_paths):
        # Processar múltiplas imagens simultaneamente
        pass
```

## 📊 Configuração de Monitoramento

### 1. **Métricas Customizadas**

#### **Sistema de Métricas**
```python
# Criar metrics.py
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'total_predictions': 0,
            'flooding_events': 0,
            'people_detected': 0,
            'avg_confidence': 0.0,
            'processing_time': []
        }
    
    def record_prediction(self, result, processing_time):
        self.metrics['total_predictions'] += 1
        self.metrics['processing_time'].append(processing_time)
        
        if result['flooding'] == 'Sim':
            self.metrics['flooding_events'] += 1
            
        if result['people_at_risk'] == 'Sim':
            self.metrics['people_detected'] += 1
```

### 2. **Alertas e Notificações**

#### **Sistema de Alertas**
```python
# Em config/settings.py
ALERT_CONFIG = {
    'enable_email': False,
    'enable_sms': False,
    'enable_webhook': True,
    'webhook_url': 'https://hooks.slack.com/...',
    'alert_threshold': 'Perigo',  # Normal | Atenção | Perigo
    'cooldown_minutes': 15        # Evitar spam
}

# Sistema de notificação
class AlertManager:
    def __init__(self, config):
        self.config = config
        self.last_alert = None
    
    def should_alert(self, classification):
        # Lógica de alertas
        pass
    
    def send_alert(self, data):
        # Implementar envio
        pass
```

## 🔐 Configuração de Segurança

### 1. **Autenticação Básica**

#### **Implementar Login Simples**
```python
# Em api/app.py
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash

auth = HTTPBasicAuth()

users = {
    "admin": "senha_hash_aqui",
    "viewer": "senha_hash_aqui"
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/api/protected')
@auth.login_required
def protected():
    return jsonify({'user': auth.current_user()})
```

### 2. **Rate Limiting**

#### **Controle de Taxa de Requisições**
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/current-status')
@limiter.limit("60 per minute")
def get_current_status():
    pass
```

### 3. **HTTPS e SSL**

#### **Configuração SSL**
```python
# Para desenvolvimento
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('cert.pem', 'key.pem')

socketio.run(app, host='0.0.0.0', port=5000, ssl_context=context)

# Para produção (usar proxy reverso)
# Nginx, Apache, etc.
```

## 🔧 Configuração de Desenvolvimento

### 1. **Ambiente de Desenvolvimento**

#### **Configuração para Debug**
```python
# development_config.py
DEBUG_CONFIG = {
    'FLASK_ENV': 'development',
    'FLASK_DEBUG': True,
    'LOG_LEVEL': 'DEBUG',
    'AUTO_RELOAD': True,
    'SIMULATE_ONLY': True,  # Usar apenas simulação
    'FAST_CYCLE': True      # Ciclos mais rápidos
}

# Aplicar configurações
if os.getenv('ENVIRONMENT') == 'development':
    DISPLAY_TIME = 3        # Ciclos de 3 segundos
    SHOW_DEBUG = True       # Mostrar debug
    SIMULATION_PRESENCE_PROB = 0.5  # 50% para testes
```

### 2. **Hot Reload**

#### **Configuração de Auto-reload**
```python
# Em main_web.py
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print("🔄 Código alterado, reiniciando...")
            # Implementar restart

if __name__ == '__main__':
    if os.getenv('FLASK_ENV') == 'development':
        # Configurar file watcher
        observer = Observer()
        observer.start()
    
    socketio.run(app, debug=True)
```

## 📈 Configuração de Performance

### 1. **Otimizações de Produção**

#### **Cache de Dados**
```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutos
})

@app.route('/api/history')
@cache.cached(timeout=300)
def get_history():
    # Dados ficam em cache por 5 minutos
    pass
```

#### **Compressão de Respostas**
```python
from flask_compress import Compress

Compress(app)  # Comprime automaticamente responses
```

#### **Configuração de Workers**
```python
# gunicorn_config.py
import multiprocessing

bind = "0.0.0.0:5000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "eventlet"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 30
keepalive = 2
```

### 2. **Monitoramento de Recursos**

#### **Métricas do Sistema**
```python
import psutil
import time

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def get_stats(self):
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'uptime': time.time() - self.start_time,
            'active_connections': len(socketio.server.manager.rooms)
        }
    
    @app.route('/api/system-stats')
    def system_stats():
        return jsonify(monitor.get_stats())
```

## 🎯 Exemplos Práticos

### 1. **Configuração para Escola**
```python
# config/school_config.py
RAIN_THRESHOLD = 30      # Mais sensível para proteção de crianças
DISPLAY_TIME = 10        # Atualizações mais frequentes
SIMULATION_PRESENCE_PROB = 0.4  # Maior chance de detecção
ALERT_CONFIG = {
    'enable_email': True,
    'recipients': ['diretor@escola.com', 'seguranca@escola.com'],
    'alert_threshold': 'Atenção'  # Alertar mesmo sem pessoas
}
```

### 2. **Configuração para Cidade**
```python
# config/city_config.py
RAIN_THRESHOLD = 60      # Apenas emergências reais
DISPLAY_TIME = 60        # Ciclos de 1 minuto
SIMULATION_PRESENCE_PROB = 0.2  # Detecção conservadora
ALERT_CONFIG = {
    'enable_webhook': True,
    'webhook_url': 'https://api.defesacivil.gov.br/alerts',
    'alert_threshold': 'Perigo'  # Apenas situações críticas
}
```

### 3. **Configuração para Desenvolvimento**
```python
# config/dev_config.py
DISPLAY_TIME = 2         # Ciclos super rápidos
SHOW_DEBUG = True        # Todos os logs
SIMULATION_PRESENCE_PROB = 0.5  # 50/50 para testes
MODEL_PATHS = [
    "tests/mock_model.pth",  # Modelo fake para testes
    *MODEL_PATHS            # Fallback para modelos reais
]
```

---

Esta documentação de configuração permite adaptar o sistema para diferentes cenários de uso, desde desenvolvimento até produção em larga escala, mantendo flexibilidade e robustez.