# 🌧️ Sistema de Monitoramento Inteligente de Alagamentos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Ativo-brightgreen.svg)]()

## 📋 Sobre o Projeto

Sistema inteligente de monitoramento de alagamentos que utiliza **Redes Neurais Convolucionais (CNN)** para detectar automaticamente a presença de pessoas em áreas inundadas. O sistema foi desenvolvido como resposta às recentes enchentes no Rio Grande do Sul, visando otimizar o tempo de resposta de equipes de emergência e potencialmente salvar vidas.

### 🎯 Objetivo Principal

Detectar automaticamente pessoas em risco em áreas alagadas através da análise de imagens capturadas por drones ou câmeras urbanas, classificando as situações em três níveis:

- 🟢 **Normal**: Sem alagamento
- 🟡 **Atenção**: Alagamento sem pessoas em risco
- 🔴 **Perigo**: Alagamento com pessoas em risco confirmadas

## 🚀 Funcionalidades

- ✅ **Simulação de sensor de chuva** (1-100mm)
- ✅ **Detecção automática de alagamentos** (≥50mm)
- ✅ **Análise de imagens com CNN treinada** (91.2% de acurácia)
- ✅ **Classificação inteligente de risco**
- ✅ **Interface de console em tempo real**
- ✅ **Sistema de fallback** (simulação quando modelo não disponível)
- ✅ **Arquitetura modular e extensível**

## 🧠 Modelo de IA

### Arquitetura CNN
- **2 Camadas Convolucionais** (32 → 64 filtros)
- **Batch Normalization** para estabilidade
- **Dropout 2D (25%)** para prevenção de overfitting
- **2 Camadas Densas** (128 → 1 neurônio)
- **Ativação Sigmoid** para classificação binária

### Performance
- 📊 **Acurácia de Teste**: 91.2%
- 📊 **Acurácia de Treino**: 87.8%
- 📊 **Overfitting**: Apenas 3.4%
- 🖼️ **Resolução**: 64x64 pixels
- ⚡ **Inferência**: ~50ms por imagem

## 📦 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/sistema-monitoramento-alagamentos.git
cd sistema-monitoramento-alagamentos
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Estrutura de dados
```bash
# Crie a estrutura de pastas (se não existir)
mkdir -p data/cnn/modelo
mkdir -p data/cnn/imagens

# Coloque seu modelo treinado em:
# data/cnn/modelo/modelo_treinado.pth

# Coloque imagens para análise em:
# data/cnn/imagens/
```

## 🎮 Como Usar

### Execução Básica
```bash
python main.py
```

### Exemplo de Saída
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

## 📁 Estrutura do Projeto

```
sistema_monitoramento/
│
├── 📄 main.py                    # Arquivo principal
├── 📄 requirements.txt           # Dependências
├── 📄 README.md                 # Este arquivo
│
├── 📁 config/                   # Configurações
│   ├── __init__.py
│   └── settings.py              # Constantes e parâmetros
│
├── 📁 models/                   # Modelos de IA
│   ├── __init__.py
│   └── cnn_model.py             # Arquitetura CNN
│
├── 📁 core/                     # Lógica principal
│   ├── __init__.py
│   ├── sensor.py                # Sensor de chuva
│   ├── image_analyzer.py        # Análise de imagens
│   ├── classifier.py            # Classificador de emergência
│   └── system.py                # Sistema principal
│
├── 📁 utils/                    # Utilitários
│   ├── __init__.py
│   ├── file_utils.py            # Manipulação de arquivos
│   └── image_utils.py           # Processamento de imagens
│
├── 📁 display/                  # Interface
│   ├── __init__.py
│   └── console_display.py       # Display do console
│
├── 📁 data/                     # Dados
│   └── cnn/
│       ├── modelo/              # Modelos treinados (.pth)
│       └── imagens/             # Imagens para inferência
│
└── 📁 tests/                    # Testes unitários
    ├── __init__.py
    ├── test_sensor.py
    ├── test_classifier.py
    └── test_image_analyzer.py
```

## ⚙️ Configuração

### Arquivo `config/settings.py`

```python
# Limiar de chuva para alagamento (mm)
RAIN_THRESHOLD = 50

# Tempo de exibição de cada resultado (segundos)
DISPLAY_TIME = 15

# Resolução das imagens para o modelo
IMAGE_SIZE = 64

# Probabilidade de simular presença (quando usando simulação)
SIMULATION_PRESENCE_PROB = 0.3
```

## 🧪 Lógica de Funcionamento

### Fluxo Principal
1. **Sensor de Chuva**: Sorteia valor entre 1-100mm
2. **Verificação de Alagamento**: Se ≥50mm → há alagamento
3. **Análise de Imagem**: CNN analisa imagem aleatória
4. **Classificação**:
   - Sem alagamento → **Normal**
   - Alagamento + sem pessoas → **Atenção** 
   - Alagamento + com pessoas → **Perigo**
5. **Exibição**: Mostra resultado por 15 segundos
6. **Loop**: Incrementa dia e recomeça

### Interpretação da CNN
```python
# Saída do modelo (sigmoid): 0.0 - 1.0
if confidence <= 0.5:
    resultado = "PRESENÇA CONFIRMADA"  # Classe 0: com_pessoas
else:
    resultado = "AUSENTE"              # Classe 1: sem_pessoas
```

## 🔧 Personalização

### Adicionando Novo Sensor
```python
# Em core/sensor.py
class WeatherAPISensor(RainSensor):
    def get_rain_level(self):
        # Implementar integração com API meteorológica
        pass
```

### Mudando Interface
```python
# Em display/
class WebDisplay(ConsoleDisplay):
    def show_result(self, result):
        # Implementar interface web
        pass
```

### Configurando Novo Modelo
```python
# Em models/cnn_model.py
class ResNetModel(nn.Module):
    # Implementar arquitetura ResNet
    pass
```

## 📊 Dados de Treinamento

### Dataset Utilizado
- **Total**: 400 imagens (320 treino + 80 teste)
- **Classes**: 
  - `com_pessoas`: 160 imagens (pessoas visíveis em alagamentos)
  - `sem_pessoas`: 160 imagens (alagamentos sem pessoas)
- **Resolução**: 64x64 pixels
- **Formato**: JPG/JPEG

### Data Augmentation Aplicado
- **RandomResizedCrop**: Recortes aleatórios
- **RandomHorizontalFlip**: Espelhamento (30%)
- **RandomRotation**: Rotações (±10°)
- **ColorJitter**: Variações de brilho/contraste

## 🧪 Testes

### Executar Testes Unitários
```bash
# Instalar pytest
pip install pytest

# Executar todos os testes
python -m pytest tests/

# Executar teste específico
python -m pytest tests/test_sensor.py -v
```

### Exemplo de Teste
```python
# tests/test_sensor.py
def test_rain_sensor():
    sensor = RainSensor()
    rain_level = sensor.get_rain_level()
    assert 1 <= rain_level <= 100
    
    assert sensor.has_flooding(60) == True
    assert sensor.has_flooding(30) == False
```

## 🤝 Contribuição

### Como Contribuir
1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. **Commit** suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. **Push** para a branch (`git push origin feature/nova-funcionalidade`)
5. **Abra** um Pull Request

### Padrões de Código
- **PEP 8**: Seguir convenções Python
- **Docstrings**: Documentar todas as funções
- **Type Hints**: Usar anotações de tipo quando possível
- **Testes**: Adicionar testes para novas funcionalidades

## 🐛 Problemas Conhecidos

- ⚠️ **Dataset pequeno**: 400 imagens podem não cobrir todos os cenários
- ⚠️ **Resolução limitada**: 64x64 pode perder detalhes importantes
- ⚠️ **Simulação de sensor**: Não integrado com sensores reais

## 🔮 Roadmap

### Versão 2.0
- [ ] 🌐 **Interface Web** com dashboard em tempo real
- [ ] 📡 **Integração com APIs meteorológicas** reais
- [ ] 📧 **Sistema de alertas** (email, SMS, WhatsApp)
- [ ] 🗄️ **Banco de dados** para histórico
- [ ] 🔗 **API REST** para integração externa

### Versão 3.0
- [ ] 🤖 **Transfer Learning** com modelos pré-treinados
- [ ] 📱 **App móvel** para equipes de campo
- [ ] 🛰️ **Integração com imagens de satélite**
- [ ] 🧠 **IA explicável** (visualização de atenção)
- [ ] ⚡ **Otimização para edge computing**

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- **Seu Nome** - *Desenvolvimento inicial* - [@seu-github](https://github.com/seu-usuario)

## 🙏 Agradecimentos

- **FIAP** - Instituição de ensino
- **PyTorch Community** - Framework de deep learning
- **Comunidade Open Source** - Inspiração e ferramentas

## 📞 Contato

- **Email**: seu.email@exemplo.com
- **LinkedIn**: [seu-perfil](https://linkedin.com/in/seu-perfil)
- **GitHub**: [@seu-usuario](https://github.com/seu-usuario)

---

<div align="center">

**🌧️ Desenvolvido com ❤️ para salvar vidas em situações de emergência 🌧️**

*Se este projeto foi útil, considere dar uma ⭐ no repositório!*

</div>
