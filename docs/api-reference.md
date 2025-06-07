# 🌐 API Reference

## 📋 Visão Geral

A API do Sistema de Monitoramento oferece endpoints REST para acesso a dados e WebSockets para atualizações em tempo real. Construída com Flask e SocketIO, fornece uma interface robusta para integração externa e alimentação do dashboard web.

## 🚀 Configuração Base

### URL Base
```
http://localhost:5000
```

### Headers Padrão
```http
Content-Type: application/json
Accept: application/json
```

### Status Codes
- `200` - Sucesso
- `404` - Recurso não encontrado
- `500` - Erro interno do servidor

## 📍 Endpoints REST

### 1. **Dashboard Principal**

#### `GET /`
Retorna a página principal do dashboard web.

**Resposta:**
- HTML da interface do dashboard

**Exemplo:**
```bash
curl http://localhost:5000/
```

---

### 2. **Status Atual do Sistema**

#### `GET /api/current-status`
Retorna o status mais recente do sistema de monitoramento.

**Resposta:**
```json
{
  "success": true,
  "data": {
    "day": 23,
    "rain_level": 67,
    "image_used": "imagem_0298.jpeg",
    "classification": "Perigo",
    "people_at_risk": "Sim",
    "flooding": "Sim",
    "timestamp": "2025-06-07T08:21:01.111014",
    "date": "07/06/2025",
    "time": "08:21:01"
  }
}
```

**Campos do Response:**
- `day` (int): Dia de simulação atual
- `rain_level` (int): Nível de chuva em mm (1-100)
- `image_used` (string): Nome da imagem analisada ou "nenhuma"
- `classification` (string): "Normal", "Atenção" ou "Perigo"
- `people_at_risk` (string): "Sim" ou "Não"
- `flooding` (string): "Sim" ou "Não"
- `timestamp` (string): ISO timestamp do registro
- `date` (string): Data formatada (DD/MM/YYYY)
- `time` (string): Hora formatada (HH:MM:SS)

**Exemplo:**
```bash
curl http://localhost:5000/api/current-status
```

**Resposta de Erro:**
```json
{
  "success": false,
  "message": "HistoryManager não disponível"
}
```

---

### 3. **Histórico Completo**

#### `GET /api/history`
Retorna todo o histórico de registros para geração de gráficos.

**Resposta:**
```json
{
  "success": true,
  "data": [
    {
      "day": 1,
      "rain_level": 66,
      "image_used": "imagem_0216.jpg",
      "classification": "Perigo",
      "people_at_risk": "Sim",
      "flooding": "Sim",
      "timestamp": "2025-06-07T08:05:10.104552",
      "date": "07/06/2025",
      "time": "08:05:10"
    },
    // ... mais registros
  ]
}
```

**Uso Típico:**
- Geração de gráficos históricos
- Análise de tendências
- Relatórios estatísticos

**Exemplo:**
```bash
curl http://localhost:5000/api/history
```

---

### 4. **Registros Recentes**

#### `GET /api/recent-records`
Retorna os últimos 15 registros para exibição em tabela.

**Resposta:**
```json
{
  "success": true,
  "data": [
    // Últimos 15 registros em ordem cronológica
  ]
}
```

**Características:**
- Limitado aos 15 registros mais recentes
- Otimizado para tabelas de histórico
- Dados ordenados cronologicamente

**Exemplo:**
```bash
curl http://localhost:5000/api/recent-records
```

---

### 5. **Imagens de Inferência**

#### `GET /images/inference/<filename>`
Serve imagens que foram analisadas pela CNN.

**Parâmetros:**
- `filename` (string): Nome do arquivo de imagem

**Resposta:**
- Arquivo de imagem (JPG/JPEG)
- Placeholder em caso de erro

**Exemplo:**
```bash
curl http://localhost:5000/images/inference/imagem_0298.jpeg
```

**Fallback:**
Se a imagem não for encontrada, retorna automaticamente o placeholder:
```
/static/images/placeholder.jpeg
```

---

### 6. **Arquivos Estáticos**

#### `GET /static/<path:filename>`
Serve arquivos estáticos (CSS, JS, imagens).

**Parâmetros:**
- `filename` (string): Caminho relativo do arquivo

**Exemplos:**
```bash
# CSS do dashboard
curl http://localhost:5000/static/css/dashboard.css

# JavaScript do dashboard
curl http://localhost:5000/static/js/dashboard.js

# Mapas dinâmicos
curl http://localhost:5000/static/images/maps/normal.jpeg
curl http://localhost:5000/static/images/maps/atencao.jpeg
curl http://localhost:5000/static/images/maps/perigo.jpeg
```

---

## 🔍 Endpoints de Debug

### 1. **Debug de Caminhos**

#### `GET /debug/paths`
Retorna informações detalhadas sobre caminhos e estrutura de arquivos.

**Resposta:**
```json
{
  "current_dir": "/path/to/api",
  "project_root": "/path/to/sistema_monitoramento",
  "template_folder": "/path/to/web/templates",
  "static_folder": "/path/to/web/static",
  "data_folder": "/path/to/data/cnn/imagens",
  "template_exists": true,
  "static_exists": true,
  "data_exists": true,
  "maps_files": ["normal.jpeg", "atencao.jpeg", "perigo.jpeg"],
  "data_files_count": 320
}
```

---

### 2. **Teste de Imagens**

#### `GET /test/images`
Verifica se todas as imagens necessárias estão disponíveis.

**Resposta:**
```json
{
  "static_folder": "/path/to/static",
  "images": {
    "images/maps/normal.jpeg": {
      "path": "/full/path/to/normal.jpeg",
      "exists": true,
      "url": "/static/images/maps/normal.jpeg"
    },
    "images/maps/atencao.jpeg": {
      "path": "/full/path/to/atencao.jpeg", 
      "exists": true,
      "url": "/static/images/maps/atencao.jpeg"
    },
    "images/maps/perigo.jpeg": {
      "path": "/full/path/to/perigo.jpeg",
      "exists": true,
      "url": "/static/images/maps/perigo.jpeg"
    },
    "images/placeholder.jpeg": {
      "path": "/full/path/to/placeholder.jpeg",
      "exists": true,
      "url": "/static/images/placeholder.jpeg"
    }
  }
}
```

---

## 🔌 WebSocket API

### Conexão
```javascript
const socket = io('http://localhost:5000');
```

### 1. **Eventos do Cliente**

#### `connect`
Evento automático ao conectar com o servidor.

**Response do Servidor:**
```json
{
  "status": "Conectado ao sistema de monitoramento"
}
```

#### `disconnect`
Evento automático ao desconectar do servidor.

---

### 2. **Eventos do Servidor**

#### `connected`
Confirmação de conexão estabelecida.

**Payload:**
```json
{
  "status": "Conectado ao sistema de monitoramento"
}
```

#### `status_update`
Atualização em tempo real dos dados do sistema.

**Payload:**
```json
{
  "day": 23,
  "rain_level": 67,
  "image_used": "imagem_0298.jpeg",
  "classification": "Perigo",
  "people_at_risk": "Sim",
  "flooding": "Sim",
  "timestamp": "2025-06-07T08:21:01.111014",
  "date": "07/06/2025",
  "time": "08:21:01"
}
```

**Frequência:**
- A cada 15 segundos (conforme ciclo do sistema)
- Apenas quando há novos dados

---

## 📝 Exemplos de Integração

### 1. **JavaScript (Frontend)**

```javascript
// Conexão WebSocket
const socket = io();

// Escutar atualizações
socket.on('status_update', (data) => {
    console.log('Novo status:', data);
    updateDashboard(data);
});

// Buscar histórico
async function loadHistory() {
    const response = await fetch('/api/history');
    const data = await response.json();
    
    if (data.success) {
        generateChart(data.data);
    }
}

// Buscar status atual
async function getCurrentStatus() {
    const response = await fetch('/api/current-status');
    const data = await response.json();
    
    if (data.success) {
        updateCurrentInfo(data.data);
    }
}
```

### 2. **Python (Cliente)**

```python
import requests
import socketio

# Cliente REST
def get_current_status():
    response = requests.get('http://localhost:5000/api/current-status')
    return response.json()

def get_history():
    response = requests.get('http://localhost:5000/api/history')
    return response.json()

# Cliente WebSocket
sio = socketio.Client()

@sio.on('status_update')
def on_status_update(data):
    print(f"Novo status: {data}")

@sio.on('connect')
def on_connect():
    print("Conectado ao sistema")

sio.connect('http://localhost:5000')
```

### 3. **curl (CLI)**

```bash
#!/bin/bash

# Status atual
echo "=== Status Atual ==="
curl -s http://localhost:5000/api/current-status | jq

# Últimos registros
echo -e "\n=== Últimos Registros ==="
curl -s http://localhost:5000/api/recent-records | jq '.data[-5:]'

# Teste de conectividade
echo -e "\n=== Teste de Paths ==="
curl -s http://localhost:5000/debug/paths | jq '.data_exists, .static_exists'
```

---

## ⚠️ Limitações e Considerações

### 1. **Performance**
- Histórico completo pode ser grande para muitos registros
- WebSocket limitado por recursos do servidor
- Imagens servidas diretamente (sem CDN)

### 2. **Segurança**
- Nenhuma autenticação implementada
- CORS habilitado para todas origens
- Endpoints de debug expostos

### 3. **Persistência**
- Dados armazenados em JSON (não transacional)
- Sem backup automático
- Histórico cresce indefinidamente

### 4. **Disponibilidade**
- Dependente do HistoryManager para dados
- Falhas em carregamento retornam erro genérico
- Sem sistema de retry

---

## 🔮 Melhorias Futuras

### 1. **Autenticação e Autorização**
```python
# JWT tokens
@app.route('/api/protected')
@jwt_required()
def protected():
    pass
```

### 2. **Paginação**
```python
@app.route('/api/history')
def get_history():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    # Implementar paginação
```

### 3. **Filtros e Busca**
```python
@app.route('/api/history/search')
def search_history():
    classification = request.args.get('classification')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    # Implementar filtros
```

### 4. **Cache**
```python
from flask_caching import Cache

cache = Cache(app)

@app.route('/api/history')
@cache.cached(timeout=300)  # 5 minutos
def get_history():
    pass
```

### 5. **Rate Limiting**
```python
from flask_limiter import Limiter

limiter = Limiter(app)

@app.route('/api/current-status')
@limiter.limit("60 per minute")
def get_current_status():
    pass
```

---

Esta API foi projetada para ser simples, eficiente e facilmente integrável, fornecendo todos os dados necessários para monitoramento e análise do sistema de alagamentos.