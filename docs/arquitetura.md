# Arquitetura do Sistema

## Visão Geral

O Sistema de Monitoramento de Alagamentos segue uma arquitetura modular baseada em componentes independentes e desacoplados, permitindo fácil manutenção e extensibilidade.

## Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND WEB                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    Dashboard    │  │    Chart.js     │  │   Socket.IO     │ │
│  │   (HTML/CSS/JS) │  │   (Gráficos)    │  │  (Tempo Real)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                          ┌─────────────┐
                          │   API REST  │
                          │   Flask +   │
                          │  WebSocket  │
                          └─────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND CORE                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ MonitoringSystem│  │  ImageAnalyzer  │  │   Classifier    │ │
│  │  (Orquestrador) │  │   (CNN Model)   │  │  (Emergência)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   RainSensor    │  │ HistoryManager  │  │   FileUtils     │ │
│  │  (Simulação)    │  │   (Storage)     │  │  (Utilitários)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    CAMADA DE DADOS                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  history.json   │  │ Modelo CNN.pth  │  │ Imagens Dataset │ │
│  │   (Histórico)   │  │  (Pesos IA)     │  │   (Inferência)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Componentes Detalhados

### 1. Sistema Principal (`core/system.py`)

**Responsabilidade**: Orquestração do ciclo de monitoramento

```python
class MonitoringSystem:
    - Coordena sensor, análise e classificação
    - Gerencia ciclo de dias
    - Integra componentes independentes
```

**Fluxo de Execução:**
1. Leitura do sensor de chuva
2. Verificação de threshold de alagamento  
3. Seleção e análise de imagem (se necessário)
4. Classificação de emergência
5. Retorno de resultado estruturado

### 2. Extensão Web (`core/web_system.py`)

**Responsabilidade**: Adiciona funcionalidades web ao sistema base

```python
class WebMonitoringSystem(MonitoringSystem):
    - Herda comportamento básico
    - Adiciona persistência de dados
    - Integra timestamps
    - Suporte a limpeza de histórico
```

**Funcionalidades Adicionais:**
- Timestamps automáticos
- Persistência em JSON
- Notificação via WebSocket
- Parâmetro `clear_history`

### 3. Modelo de IA (`models/cnn_model.py`)

**Responsabilidade**: Arquitetura da rede neural convolucional

```python
class CNNModel(nn.Module):
    Camadas:
    ├── Conv2d(3, 32) + BatchNorm2d + ReLU + MaxPool2d + Dropout2d(0.25)
    ├── Conv2d(32, 64) + BatchNorm2d + ReLU + MaxPool2d + Dropout2d(0.25)  
    ├── Flatten
    ├── Linear(16384, 128) + ReLU
    └── Linear(128, 1) + Sigmoid
```

**Características:**
- Entrada: RGB 64x64 pixels
- Saída: Score probabilístico [0,1]
- Threshold: ≤0.5 = pessoas detectadas
- Regularização: BatchNorm + Dropout

### 4. Analisador de Imagens (`core/image_analyzer.py`)

**Responsabilidade**: Interface entre sistema e modelo CNN

```python
class ImageAnalyzer:
    - Carregamento do modelo treinado
    - Pré-processamento de imagens
    - Inferência e pós-processamento
    - Sistema de fallback
```

**Pipeline de Processamento:**
1. Carregamento da imagem (PIL)
2. Transformações (resize, tensor, normalização)
3. Inferência no modelo
4. Interpretação do resultado
5. Retorno estruturado

### 5. Sensor de Chuva (`core/sensor.py`)

**Responsabilidade**: Simulação de dados meteorológicos

```python
class RainSensor:
    - Geração aleatória de níveis (1-100mm)
    - Verificação de threshold configurável
    - Interface extensível para sensores reais
```

### 6. Classificador de Emergência (`core/classifier.py`)

**Responsabilidade**: Lógica de classificação de risco

```python
class EmergencyClassifier:
    Regras:
    - Sem alagamento → Normal
    - Alagamento + sem pessoas → Atenção  
    - Alagamento + pessoas → Perigo
```

### 7. API Web (`api/app.py`)

**Responsabilidade**: Interface REST e WebSocket

```python
Flask App:
├── Rotas REST (/api/*)
├── Servir arquivos estáticos
├── WebSocket para tempo real
└── CORS para desenvolvimento
```

**Endpoints:**
- `/api/current-status`: Status atual
- `/api/history`: Histórico completo
- `/api/recent-records`: Últimos registros
- `/images/inference/<file>`: Servir imagens

### 8. Persistência (`database/storage.py`)

**Responsabilidade**: Gerenciamento de dados históricos

```python
class HistoryManager:
    - Operações CRUD em JSON
    - Thread-safety
    - Consultas otimizadas
    - Parâmetro clear_on_init
```

**Estrutura de Dados:**
```json
[
  {
    "day": int,
    "rain_level": int,
    "image_used": string,
    "classification": "Normal|Atenção|Perigo",
    "people_at_risk": "Sim|Não", 
    "flooding": "Sim|Não",
    "timestamp": "ISO-8601",
    "date": "DD/MM/YYYY",
    "time": "HH:MM:SS"
  }
]
```

## Design Patterns Utilizados

### 1. **Strategy Pattern**
- `RainSensor`: Interface para diferentes tipos de sensores
- `ImageAnalyzer`: Estratégias de análise (modelo vs simulação)

### 2. **Template Method Pattern**  
- `MonitoringSystem.process_day()`: Define fluxo padrão
- `WebMonitoringSystem`: Estende com funcionalidades web

### 3. **Facade Pattern**
- `MonitoringSystem`: Fachada para componentes internos
- `HistoryManager`: Fachada para operações de persistência

### 4. **Observer Pattern**
- WebSocket: Notifica dashboard sobre mudanças de estado

## Fluxo de Dados

### 1. Modo Console (`main.py`)
```
Usuario → main() → MonitoringSystem → ConsoleDisplay → Loop
```

### 2. Modo Web (`main_web.py`)
```
Usuario → main() → Thread(MonitoringSystem) + Flask(API) → WebSocket → Dashboard
```

### 3. Processamento de um Ciclo
```
RainSensor → Threshold Check → [ImageAnalyzer] → Classifier → Storage → Display/API
```

## Configuração e Extensibilidade

### Pontos de Extensão

1. **Novos Sensores**: Implementar interface `RainSensor`
2. **Novos Modelos**: Substituir `CNNModel` 
3. **Novos Displays**: Implementar interface `Display`
4. **Novos Storages**: Implementar interface similar a `HistoryManager`

### Configurações (`config/settings.py`)

```python
# Thresholds
RAIN_THRESHOLD = 50  # mm

# Temporização  
DISPLAY_TIME = 15    # segundos

# Modelo
IMAGE_SIZE = 64      # pixels
DEVICE = "cpu"       # pytorch device

# Simulação
SIMULATION_PRESENCE_PROB = 0.3  # probabilidade
```

## Considerações de Performance

### Otimizações Implementadas

1. **Carregamento Único**: Modelo CNN carregado uma vez na inicialização
2. **Threading**: Sistema de monitoramento em thread separada
3. **Cache de Transformações**: Pipeline de pré-processamento fixo
4. **JSON Atômico**: Operações de escrita thread-safe

### Limitações de Escala

1. **Processamento Sequencial**: Uma imagem por vez
2. **Storage JSON**: Não otimizado para grandes volumes
3. **CPU-bound**: Inferência em CPU

### Possíveis Melhorias

1. **GPU Support**: Transferir inferência para GPU
2. **Batch Processing**: Múltiplas imagens simultaneamente  
3. **Database**: Migrar para PostgreSQL/MongoDB
4. **Microservices**: Separar componentes em serviços independentes

## Segurança

### Medidas Implementadas

1. **Validação de Paths**: Verificação de existência de arquivos
2. **Sanitização**: Inputs validados antes do processamento
3. **CORS Configurado**: Para desenvolvimento local
4. **Error Handling**: Try/catch em operações críticas

### Recomendações para Produção

1. **HTTPS**: Certificados SSL/TLS
2. **Authentication**: Sistema de autenticação/autorização
3. **Rate Limiting**: Limitação de requests por IP
4. **Input Validation**: Validação rigorosa de entradas
5. **Logging**: Sistema de logs estruturado

## Deployment

### Desenvolvimento Local
```bash
python main_web.py  # Modo desenvolvimento
```

### Produção (Recomendações)
```bash
# Usando Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api.app:app

# Usando Docker
docker build -t monitoring-system .
docker run -p 5000:5000 monitoring-system
```

### Dependências de Sistema
- Python 3.8+
- PyTorch com CPU support
- 4GB RAM (mínimo)
- 1GB storage (dados + modelo)