# Sistema de Monitoramento Inteligente de Alagamentos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![Status](https://img.shields.io/badge/Status-Produção-brightgreen.svg)]()

## Visão Geral

Sistema inteligente para monitoramento em tempo real de alagamentos urbanos usando **Redes Neurais Convolucionais (CNN)** para detectar pessoas em áreas inundadas. Desenvolvido para otimizar respostas de emergência e salvar vidas em situações de risco.

### Funcionalidades Principais

- **Detecção Automática de Alagamentos**: Sensor de chuva com threshold configurável
- **Análise de Imagens com IA**: CNN especializada para detectar pessoas em enchentes
- **Classificação Inteligente**: Sistema de 3 níveis de alerta (Normal/Atenção/Perigo)
- **Dashboard Web em Tempo Real**: Interface moderna com WebSockets
- **API REST Completa**: Endpoints para integração externa
- **Persistência de Dados**: Histórico completo em JSON com timestamps
- **Modo Console**: Interface textual para desenvolvimento/debug

## Instalação Rápida

```bash
# Clone o repositório
git clone <repository-url>
cd sistema_monitoramento

# Instale dependências
pip install -r requirements.txt

# Execute o sistema
python main_web.py  # Modo web (recomendado)
# ou
python main.py      # Modo console
```

**Dashboard Web**: http://localhost:5000

## Arquitetura do Sistema

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Sensor Chuva  │───▶│  Detector    │───▶│ Analisador  │
│   (1-100mm)     │    │ Alagamento   │    │ CNN         │
└─────────────────┘    │  (≥50mm)     │    │ (64x64)     │
                       └──────────────┘    └─────────────┘
                                                   │
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│  Dashboard Web  │◀───│  Storage     │◀───│ Classificador │
│  (Tempo Real)   │    │  JSON        │    │ Emergência  │
└─────────────────┘    └──────────────┘    └─────────────┘
```

### Componentes Principais

| Componente | Descrição | Localização |
|------------|-----------|-------------|
| **Sistema Principal** | Lógica core de monitoramento | `core/system.py` |
| **Modelo CNN** | Rede neural para detecção | `models/cnn_model.py` |
| **API Web** | Servidor Flask + WebSocket | `api/app.py` |
| **Dashboard** | Interface web responsiva | `web/templates/` |
| **Storage** | Persistência em JSON | `database/storage.py` |
| **Configurações** | Parâmetros do sistema | `config/settings.py` |

## Modelo de Inteligência Artificial

### Arquitetura CNN
```python
Entrada: RGB 64x64 pixels
├── Conv2D(3→32) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
├── Conv2D(32→64) + BatchNorm + ReLU + MaxPool + Dropout(0.25)  
├── Flatten + Linear(16384→128) + ReLU
└── Linear(128→1) + Sigmoid
Saída: Score (≤0.5 = pessoas detectadas)
```

### Performance
- **Acurácia**: 91.2% em dados de teste
- **Dataset**: 400 imagens (alagamentos com/sem pessoas)
- **Tempo de Inferência**: ~50ms por imagem
- **Fallback**: Sistema de simulação quando modelo indisponível

## Sistema de Classificação

| Nível | Condição | Ação Recomendada |
|-------|----------|------------------|
| 🟢 **Normal** | Sem alagamento | Monitoramento regular |
| 🟡 **Atenção** | Alagamento sem pessoas | Preparar equipes |
| 🔴 **Perigo** | Alagamento + pessoas detectadas | **Resposta imediata** |

## Modos de Execução

### 1. Modo Web (Recomendado)
```bash
python main_web.py
```
- Dashboard interativo em tempo real
- WebSockets para atualizações instantâneas
- API REST para integrações
- Histórico gráfico e tabelas
- Mapas dinâmicos por status

### 2. Modo Console
```bash
python main.py
```
- Interface textual simples
- Ideal para desenvolvimento/debug
- Saída estruturada no terminal
- Sem persistência web

## API REST

### Endpoints Principais

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/api/current-status` | Status atual do sistema |
| `GET` | `/api/history` | Histórico completo |
| `GET` | `/api/recent-records` | Últimos 15 registros |
| `GET` | `/images/inference/<file>` | Imagens analisadas |

### Exemplo de Resposta
```json
{
  "day": 1,
  "rain_level": 73,
  "image_used": "imagem_0156.jpg",
  "classification": "Perigo",
  "people_at_risk": "Sim",
  "flooding": "Sim",
  "timestamp": "2025-06-07T13:47:34.301779"
}
```

## Configuração

### Parâmetros Principais (`config/settings.py`)

```python
RAIN_THRESHOLD = 50        # mm de chuva para alagamento
DISPLAY_TIME = 15          # segundos entre processamentos
IMAGE_SIZE = 64            # resolução para CNN
SIMULATION_PRESENCE_PROB = 0.3  # probabilidade de simulação
```

### Gerenciamento de Histórico

**Limpeza Automática** (novo na v2.0):
```python
# Sistema padrão - mantém histórico
system = WebMonitoringSystem()

# Limpa histórico a cada inicialização
system = WebMonitoringSystem(clear_history=True)
```

## Estrutura de Dados

### Registro de Monitoramento
```json
{
  "day": 1,
  "rain_level": 73,
  "image_used": "imagem_0156.jpg", 
  "classification": "Perigo",
  "people_at_risk": "Sim",
  "flooding": "Sim",
  "timestamp": "2025-06-07T13:47:34.301779",
  "date": "07/06/2025",
  "time": "13:47:34"
}
```

### Persistência
- **Arquivo**: `data/history.json`
- **Formato**: Array JSON com timestamps
- **Operações**: Create, Read, Query (últimos N registros)
- **Thread-Safe**: Operações atômicas

## Dependências

```txt
torch>=1.9.0        # Framework de deep learning
torchvision>=0.10.0 # Transformações de imagem
Pillow>=8.0.0       # Processamento de imagem
flask>=2.0.0        # Framework web
flask-socketio>=5.0 # WebSocket support
```

## Documentação Técnica

Para informações detalhadas, consulte:

- **[Arquitetura Detalhada](docs/arquitetura.md)** - Componentes e design patterns
- **[Modelo CNN](docs/modelo-cnn.md)** - Treinamento e performance da IA
- **[API Reference](docs/api-reference.md)** - Endpoints e schemas completos
- **[Configuração Avançada](docs/configuracao.md)** - Parâmetros e customizações
- **[Guia de Instalação](docs/instalacao.md)** - Setup detalhado e troubleshooting

## Desenvolvimento

### Estrutura do Projeto
```
sistema_monitoramento/
├── core/           # Lógica principal
├── models/         # Modelos de IA
├── api/           # Servidor web
├── web/           # Frontend
├── database/      # Persistência
├── config/        # Configurações
├── data/          # Dados e modelos
├── docs/          # Documentação
└── tests/         # Testes unitários
```

### Contribuição

1. Fork o projeto
2. Crie branch: `git checkout -b feature/nova-funcionalidade`
3. Commit: `git commit -m 'Adiciona nova funcionalidade'`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abra Pull Request

## Casos de Uso

### Aplicações Reais
- **Gestão Municipal**: Monitoramento urbano automatizado
- **Defesa Civil**: Sistema de alerta precoce
- **IoT/Smart Cities**: Integração com sensores urbanos
- **Pesquisa**: Análise de padrões de enchentes

### Integrações Possíveis
- APIs meteorológicas
- Sistemas de câmeras urbanas
- Drones de monitoramento
- Plataformas de alertas (SMS, email)
- Sistemas GIS

## Performance e Limitações

### Pontos Fortes
- ✅ Arquitetura modular e extensível
- ✅ IA especializada com boa acurácia
- ✅ Interface web moderna e responsiva
- ✅ Sistema robusto com fallbacks
- ✅ API REST completa

### Limitações Conhecidas
- ⚠️ Dataset limitado (400 imagens)
- ⚠️ Resolução 64x64 pode perder detalhes
- ⚠️ Sensor simulado (não integrado a hardware real)
- ⚠️ Processamento sequencial (uma imagem por vez)

## Roadmap

### Próximas Versões
- [ ] Integração com APIs meteorológicas reais
- [ ] Sistema de notificações (email, SMS)
- [ ] Banco de dados relacional
- [ ] Processamento paralelo de múltiplas câmeras
- [ ] App móvel para equipes de campo
- [ ] Machine Learning Pipeline automatizado

## Suporte

Para dúvidas, bugs ou sugestões:
- **Issues**: Use o sistema de issues do GitHub
- **Documentação**: Consulte a pasta `docs/`
- **Exemplos**: Veja os arquivos de teste

---

**Desenvolvido com foco em salvar vidas em situações de emergência** 🌧️💙