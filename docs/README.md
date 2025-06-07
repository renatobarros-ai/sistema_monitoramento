# 🌧️ Sistema de Monitoramento Inteligente de Alagamentos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Ativo-brightgreen.svg)]()

## 📋 Sobre o Projeto

Sistema inteligente de monitoramento de alagamentos que utiliza **Redes Neurais Convolucionais (CNN)** para detectar automaticamente a presença de pessoas em áreas inundadas. Desenvolvido como resposta às recentes enchentes no Rio Grande do Sul, visa otimizar o tempo de resposta de equipes de emergência e potencialmente salvar vidas.

### 🎯 Objetivos

- **Principal**: Detectar automaticamente pessoas em risco em áreas alagadas
- **Secundário**: Classificar situações em três níveis de emergência
- **Terciário**: Fornecer interface web para monitoramento em tempo real

### 🚨 Níveis de Classificação

- 🟢 **Normal**: Sem alagamento detectado
- 🟡 **Atenção**: Alagamento presente, mas sem pessoas identificadas
- 🔴 **Perigo**: Alagamento com pessoas em risco confirmadas

## 🚀 Funcionalidades Principais

### ✅ Implementadas

- **Simulação de sensor de chuva** (1-100mm)
- **Detecção automática de alagamentos** (≥50mm)
- **Análise de imagens com CNN** (91.2% de acurácia)
- **Classificação inteligente de risco** (3 níveis)
- **Interface de console** em tempo real
- **Dashboard web** com atualizações via WebSocket
- **Sistema de persistência** de dados em JSON
- **API REST** para integração externa
- **Gráficos históricos** interativos
- **Sistema de fallback** quando modelo indisponível

### 🔮 Roadmap Futuro

- [ ] Integração com APIs meteorológicas reais
- [ ] Sistema de alertas (email, SMS, WhatsApp)
- [ ] Banco de dados relacional
- [ ] App móvel para equipes de campo
- [ ] Integração com imagens de satélite
- [ ] IA explicável com visualização de atenção

## 📦 Instalação Rápida

```bash
# 1. Clone o repositório
git clone <url-repositorio>
cd sistema_monitoramento

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o sistema
python main.py              # Console apenas
python main_web.py          # Console + Web Dashboard
```

## 🎮 Como Usar

### Modo Console
```bash
python main.py
```

### Modo Web (Recomendado)
```bash
python main_web.py
```
Acesse: `http://localhost:5000`

### Exemplo de Saída Console
```
🚀 SISTEMA DE MONITORAMENTO INICIADO
⏹️  Pressione Ctrl+C para parar

✅ Modelo carregado: data/cnn/modelo/modelo_treinado.pth
   - Acurácia: 91.2%
📷 320 imagens encontradas

==================================================
🌧️  Dia: 1
📊 Classificação: Perigo
👥 Pessoas em risco: Sim
🌊 Alagamento: Sim
☔ Nível de Chuva: 73 mm
📷 Imagem: imagem_0156.jpg
==================================================
⏱️  Aguardando 15 segundos...
```

## 📚 Documentação Completa

- **[🏗️ Arquitetura](arquitetura.md)** - Estrutura e componentes do sistema
- **[🧠 Modelo CNN](modelo-cnn.md)** - Detalhes da rede neural e treinamento
- **[🌐 API Reference](api-reference.md)** - Documentação completa da API
- **[⚙️ Configuração](configuracao.md)** - Guia de configuração e personalização
- **[📖 Guia de Instalação](instalacao.md)** - Instalação detalhada e troubleshooting

## 🏗️ Estrutura do Projeto

```
sistema_monitoramento/
├── 📄 main.py                    # Entrada console
├── 📄 main_web.py               # Entrada web
├── 📄 sistema.py                # Versão legada
├── 📄 requirements.txt          # Dependências
├── 📁 core/                     # Lógica principal
│   ├── system.py               # Sistema principal
│   ├── web_system.py           # Extensão web
│   ├── sensor.py               # Sensor de chuva
│   ├── image_analyzer.py       # Análise CNN
│   └── classifier.py           # Classificador emergência
├── 📁 models/                   # Modelos de IA
│   └── cnn_model.py            # Arquitetura CNN
├── 📁 api/                      # API Web
│   └── app.py                  # Servidor Flask
├── 📁 web/                      # Interface Web
│   ├── templates/              # Templates HTML
│   └── static/                 # CSS, JS, imagens
├── 📁 database/                 # Persistência
│   └── storage.py              # Gerenciador histórico
├── 📁 data/                     # Dados
│   ├── cnn/modelo/             # Modelos treinados
│   ├── cnn/imagens/            # Dataset inferência
│   └── history.json            # Histórico sistema
├── 📁 config/                   # Configurações
│   └── settings.py             # Constantes
├── 📁 utils/                    # Utilitários
│   └── file_utils.py           # Manipulação arquivos
├── 📁 display/                  # Interface
│   └── console_display.py      # Display console
├── 📁 docs/                     # Documentação
└── 📁 tests/                    # Testes unitários
```

## 🧠 Modelo de IA - Resumo

### Arquitetura CNN
- **2 Camadas Convolucionais** (32 → 64 filtros)
- **Batch Normalization** para estabilidade
- **Dropout 2D (25%)** para prevenção de overfitting
- **2 Camadas Densas** (128 → 1 neurônio)
- **Ativação Sigmoid** para classificação binária

### Performance Atual
- 📊 **Acurácia de Teste**: 91.2%
- 📊 **Resolução**: 64x64 pixels
- ⚡ **Tempo de Inferência**: ~50ms por imagem
- 🖼️ **Dataset**: 400 imagens (320 treino + 80 teste)

## 🌐 Dashboard Web

### Funcionalidades
- **Monitoramento em tempo real** via WebSockets
- **Gráficos interativos** de histórico de chuva
- **Mapas dinâmicos** que mudam conforme classificação
- **Tabela de registros** com últimas detecções
- **Visualização de imagens** analisadas pela CNN

### Endpoints API
- `GET /` - Dashboard principal
- `GET /api/current-status` - Status atual do sistema
- `GET /api/history` - Histórico completo
- `GET /api/recent-records` - Últimos 15 registros
- `GET /images/inference/<filename>` - Imagens analisadas

## ⚙️ Configuração Básica

### Arquivo `config/settings.py`
```python
# Limiar de chuva para alagamento (mm)
RAIN_THRESHOLD = 50

# Tempo de exibição de cada resultado (segundos)
DISPLAY_TIME = 15

# Resolução das imagens para o modelo
IMAGE_SIZE = 64

# Probabilidade de simular presença
SIMULATION_PRESENCE_PROB = 0.3
```

## 🧪 Lógica de Funcionamento

### Fluxo Principal
1. **Sensor de Chuva**: Gera valor aleatório 1-100mm
2. **Verificação de Alagamento**: Se ≥50mm → há alagamento
3. **Análise de Imagem**: CNN analisa imagem aleatória do dataset
4. **Classificação**:
   - Sem alagamento → **Normal**
   - Alagamento + sem pessoas → **Atenção** 
   - Alagamento + com pessoas → **Perigo**
5. **Persistência**: Salva em histórico JSON
6. **Exibição**: Console + Web Dashboard
7. **Loop**: Aguarda 15s e recomeça

### Interpretação CNN
```python
# Saída do modelo (sigmoid): 0.0 - 1.0
if confidence <= 0.5:
    resultado = "PRESENÇA CONFIRMADA"  # Classe 0: com_pessoas
else:
    resultado = "AUSENTE"              # Classe 1: sem_pessoas
```

## 🤝 Contribuição

1. **Fork** o projeto
2. **Crie** branch para feature (`git checkout -b feature/nova-funcionalidade`)
3. **Commit** mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. **Push** para branch (`git push origin feature/nova-funcionalidade`)
5. **Abra** Pull Request

### Padrões de Código
- **PEP 8**: Convenções Python
- **Docstrings**: Documentar funções
- **Type Hints**: Anotações de tipo
- **Testes**: Adicionar para novas funcionalidades

## 📊 Estatísticas Atuais

Com base no histórico de execução:
- **Taxa de Alagamento**: ~55% dos registros
- **Detecção de Pessoas**: ~45% quando há alagamento
- **Distribuição de Classificações**: Normal (45%), Atenção (30%), Perigo (25%)
- **Tempo de Processamento**: ~15s por ciclo

## 🐛 Problemas Conhecidos

- ⚠️ **Dataset pequeno**: 400 imagens podem não cobrir cenários diversos
- ⚠️ **Resolução limitada**: 64x64 pode perder detalhes importantes
- ⚠️ **Sensor simulado**: Não integrado com sensores meteorológicos reais
- ⚠️ **Modelo único**: Falta sistema de versionamento de modelos

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](../LICENSE) para detalhes.

## 👥 Autores

- **Desenvolvedor Principal** - *Desenvolvimento e arquitetura*

## 🙏 Agradecimentos

- **FIAP** - Instituição de ensino
- **PyTorch Community** - Framework de deep learning
- **Comunidade Open Source** - Inspiração e ferramentas

## 📞 Suporte

Para dúvidas, problemas ou sugestões:
- Abra uma **Issue** no repositório
- Consulte a [documentação completa](arquitetura.md)
- Verifique os [problemas conhecidos](#-problemas-conhecidos)

---

<div align="center">

**🌧️ Desenvolvido com ❤️ para salvar vidas em situações de emergência 🌧️**

*Se este projeto foi útil, considere dar uma ⭐ no repositório!*

</div>