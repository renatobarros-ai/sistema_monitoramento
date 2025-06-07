# 📖 Guia de Instalação e Uso

## 📋 Visão Geral

Este guia completo fornece instruções detalhadas para instalação, configuração e uso do Sistema de Monitoramento Inteligente de Alagamentos em diferentes ambientes e sistemas operacionais.

## 🔧 Pré-requisitos

### 1. **Sistema Operacional**
- **Linux**: Ubuntu 18.04+ (recomendado), CentOS 7+, Debian 10+
- **Windows**: Windows 10+ com WSL2 (recomendado) ou nativo
- **macOS**: macOS 10.14+

### 2. **Python**
```bash
# Verificar versão (necessário 3.8+)
python --version
# ou
python3 --version
```

**Instalação do Python:**

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### CentOS/RHEL
```bash
sudo yum install python3 python3-pip
# ou no CentOS 8+
sudo dnf install python3 python3-pip
```

#### Windows
```bash
# Baixar de https://python.org
# Ou usar Chocolatey
choco install python
```

#### macOS
```bash
# Usar Homebrew
brew install python
```

### 3. **Git**
```bash
# Verificar instalação
git --version

# Instalar se necessário
# Ubuntu/Debian: sudo apt install git
# CentOS/RHEL: sudo yum install git
# Windows: baixar de https://git-scm.com
# macOS: brew install git
```

## 🚀 Instalação Rápida

### 1. **Clone do Repositório**
```bash
# Clone o projeto
git clone <url-do-repositorio>
cd sistema_monitoramento

# Verificar estrutura
ls -la
```

### 2. **Ambiente Virtual (Recomendado)**
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Verificar ativação (deve mostrar (venv) no prompt)
which python
```

### 3. **Instalação de Dependências**
```bash
# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt

# Verificar instalação
pip list
```

### 4. **Verificação da Instalação**
```bash
# Teste básico
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import flask; print('Flask:', flask.__version__)"
python -c "import PIL; print('Pillow: OK')"
```

## 📁 Preparação de Dados

### 1. **Estrutura de Diretórios**
```bash
# Criar diretórios necessários (se não existirem)
mkdir -p data/cnn/modelo
mkdir -p data/cnn/imagens
mkdir -p web/static/images/maps

# Verificar estrutura
tree data/ -I "__pycache__"
```

### 2. **Modelo CNN**
```bash
# Colocar modelo treinado em:
# data/cnn/modelo/modelo_treinado.pth

# Verificar se existe
ls -la data/cnn/modelo/

# Tamanho esperado: ~8-10MB
du -h data/cnn/modelo/modelo_treinado.pth
```

### 3. **Dataset de Imagens**
```bash
# Verificar imagens disponíveis
ls data/cnn/imagens/ | wc -l

# Formatos suportados
ls data/cnn/imagens/*.{jpg,jpeg,png} 2>/dev/null | head -5
```

### 4. **Imagens Web (Mapas)**
```bash
# Verificar mapas para dashboard
ls web/static/images/maps/
# Deve conter: normal.jpeg, atencao.jpeg, perigo.jpeg

# Placeholder para imagens não encontradas
ls web/static/images/placeholder.jpeg
```

## 🎮 Executando o Sistema

### 1. **Modo Console Básico**
```bash
# Execução simples
python main.py

# Com logs detalhados
python main.py 2>&1 | tee logs/console.log

# Em background
nohup python main.py > logs/console.log 2>&1 &
```

**Saída Esperada:**
```
🚀 SISTEMA DE MONITORAMENTO INICIADO
⏹️  Pressione Ctrl+C para parar

🤖 CARREGANDO MODELO CNN:
✅ Modelo encontrado: data/cnn/modelo/modelo_treinado.pth
✅ Checkpoint válido encontrado
   - Acurácia: 91.2%
   - Épocas: 100
✅ Modelo carregado com sucesso!
📷 320 imagens encontradas
```

### 2. **Modo Web Completo (Recomendado)**
```bash
# Execução com dashboard web
python main_web.py

# Com configuração específica
FLASK_ENV=development python main_web.py

# Em produção
FLASK_ENV=production python main_web.py
```

**Saída Esperada:**
```
🚀 Iniciando Sistema Completo (Console + Web)
🌐 Dashboard disponível em: http://localhost:5000

📁 Pasta atual: /path/to/api
📁 Raiz do projeto: /path/to/sistema_monitoramento
✅ Templates existe: True
✅ Static existe: True
✅ Data CNN existe: True
✅ HistoryManager carregado com sucesso

 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://[IP_LOCAL]:5000
```

### 3. **Acessando o Dashboard**
```bash
# Abrir no navegador
# Linux
xdg-open http://localhost:5000

# macOS  
open http://localhost:5000

# Windows
start http://localhost:5000

# Ou acessar manualmente: http://localhost:5000
```

## 🛠️ Configuração Personalizada

### 1. **Arquivo de Configuração**
```python
# Editar config/settings.py

# Limiar de chuva para alagamento (mm)
RAIN_THRESHOLD = 50  # Padrão: 50mm

# Tempo de exibição de cada resultado (segundos)  
DISPLAY_TIME = 15   # Padrão: 15s

# Resolução das imagens para o modelo
IMAGE_SIZE = 64     # Padrão: 64x64

# Probabilidade de simular presença
SIMULATION_PRESENCE_PROB = 0.3  # Padrão: 30%
```

### 2. **Variáveis de Ambiente**
```bash
# Configurar variáveis opcionais
export FLASK_ENV=development     # development | production
export FLASK_DEBUG=1            # 0 | 1
export FLASK_PORT=5000          # Porta do servidor
export MODEL_PATH=/path/to/model.pth  # Caminho alternativo do modelo

# Executar com configurações
python main_web.py
```

### 3. **Configuração de Porta**
```python
# Editar main_web.py ou api/app.py
socketio.run(app, debug=False, host='0.0.0.0', port=8080)  # Mudar para 8080
```

## 🐳 Instalação com Docker (Alternativa)

### 1. **Dockerfile**
```dockerfile
# Criar Dockerfile na raiz do projeto
FROM python:3.9-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/cnn/modelo data/cnn/imagens logs

# Porta
EXPOSE 5000

# Comando padrão
CMD ["python", "main_web.py"]
```

### 2. **Build e Execução**
```bash
# Build da imagem
docker build -t sistema-monitoramento .

# Executar container
docker run -p 5000:5000 -v $(pwd)/data:/app/data sistema-monitoramento

# Com docker-compose
cat > docker-compose.yml << EOF
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
EOF

docker-compose up
```

## 🧪 Teste da Instalação

### 1. **Teste Básico de Componentes**
```bash
# Criar script de teste
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3

import sys
import os

def test_imports():
    """Testa imports principais"""
    try:
        import torch
        import torchvision
        import PIL
        import flask
        import flask_socketio
        print("✅ Todos os imports OK")
        return True
    except ImportError as e:
        print(f"❌ Erro no import: {e}")
        return False

def test_directories():
    """Testa estrutura de diretórios"""
    dirs = [
        'core', 'models', 'api', 'web', 'config', 
        'data/cnn/modelo', 'data/cnn/imagens'
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ Diretório {dir_path} existe")
        else:
            print(f"❌ Diretório {dir_path} não encontrado")
            return False
    return True

def test_model_loading():
    """Testa carregamento do modelo"""
    try:
        from models.cnn_model import CNN
        model = CNN()
        print("✅ Arquitetura CNN OK")
        return True
    except Exception as e:
        print(f"❌ Erro no modelo: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testando instalação...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Diretórios", test_directories), 
        ("Modelo CNN", test_model_loading)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        if test_func():
            passed += 1
    
    print(f"\n📊 Resultado: {passed}/{len(tests)} testes passaram")
    
    if passed == len(tests):
        print("🎉 Instalação OK - Sistema pronto para uso!")
        sys.exit(0)
    else:
        print("⚠️ Problemas encontrados - verificar configuração")
        sys.exit(1)
EOF

# Executar teste
python test_installation.py
```

### 2. **Teste de Integração Web**
```bash
# Testar API endpoints
curl -s http://localhost:5000/debug/paths | python -m json.tool
curl -s http://localhost:5000/test/images | python -m json.tool
curl -s http://localhost:5000/api/current-status | python -m json.tool
```

### 3. **Teste de Performance**
```bash
# Criar script de benchmark
cat > benchmark.py << 'EOF'
import time
import sys
import os

# Adicionar path
sys.path.append('.')

def benchmark_model():
    """Benchmark do modelo CNN"""
    try:
        from core.image_analyzer import ImageAnalyzer
        from utils.file_utils import list_images
        
        analyzer = ImageAnalyzer()
        images = list_images("data/cnn/imagens")
        
        if not images:
            print("❌ Nenhuma imagem encontrada")
            return
        
        # Teste com 5 imagens
        test_images = images[:5]
        start_time = time.time()
        
        for img in test_images:
            result = analyzer.analyze_image(img)
            print(f"📷 {os.path.basename(img)}: {result}")
        
        elapsed = time.time() - start_time
        avg_time = elapsed / len(test_images)
        
        print(f"\n⏱️ Tempo total: {elapsed:.2f}s")
        print(f"⏱️ Tempo médio: {avg_time:.2f}s por imagem")
        
        if avg_time < 0.1:
            print("🚀 Performance excelente!")
        elif avg_time < 0.5:
            print("✅ Performance boa")
        else:
            print("⚠️ Performance pode ser melhorada")
            
    except Exception as e:
        print(f"❌ Erro no benchmark: {e}")

if __name__ == "__main__":
    benchmark_model()
EOF

python benchmark.py
```

## 🚨 Solução de Problemas (Troubleshooting)

### 1. **Problemas Comuns**

#### Modelo não carregado
```bash
# Verificar existência
ls -la data/cnn/modelo/

# Verificar permissões
chmod 644 data/cnn/modelo/*.pth

# Verificar integridade
python -c "import torch; torch.load('data/cnn/modelo/modelo_treinado.pth')"
```

#### Porta em uso
```bash
# Verificar processo usando porta 5000
lsof -i :5000
netstat -tulpn | grep 5000

# Matar processo se necessário
kill -9 <PID>

# Ou usar porta alternativa
python main_web.py --port 8080
```

#### Permissões de arquivo
```bash
# Corrigir permissões
chmod -R 755 .
chmod -R 644 *.py
chmod +x main.py main_web.py
```

#### Dependências faltando
```bash
# Reinstalar dependências
pip install --force-reinstall -r requirements.txt

# Verificar versões
pip check
pip list --outdated
```

### 2. **Logs e Debug**

#### Habilitar logs detalhados
```python
# Adicionar no início do main.py ou main_web.py
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug.log'),
        logging.StreamHandler()
    ]
)
```

#### Verificar logs do sistema
```bash
# Criar diretório de logs
mkdir -p logs

# Executar com log
python main_web.py 2>&1 | tee logs/system.log

# Monitorar logs em tempo real
tail -f logs/system.log
```

### 3. **Problemas de Rede**

#### Firewall
```bash
# Ubuntu/Debian
sudo ufw allow 5000/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload

# Verificar conectividade local
curl -I http://localhost:5000
```

#### Binding de IP
```python
# Para acesso remoto, editar main_web.py
socketio.run(app, debug=False, host='0.0.0.0', port=5000)
#                               ^^^^^^^^^^^^
#                               Permite acesso externo
```

## 📊 Configuração de Produção

### 1. **Servidor WSGI**
```bash
# Instalar gunicorn
pip install gunicorn

# Executar com gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 api.app:app

# Com configuração
cat > gunicorn.conf.py << 'EOF'
bind = "0.0.0.0:5000"
workers = 1
worker_class = "eventlet"
timeout = 120
keepalive = 2
max_requests = 1000
preload_app = True
EOF

gunicorn -c gunicorn.conf.py api.app:app
```

### 2. **Proxy Reverso (Nginx)**
```nginx
# /etc/nginx/sites-available/sistema-monitoramento
server {
    listen 80;
    server_name seu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /socket.io/ {
        proxy_pass http://127.0.0.1:5000/socket.io/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. **Systemd Service**
```ini
# /etc/systemd/system/sistema-monitoramento.service
[Unit]
Description=Sistema de Monitoramento de Alagamentos
After=network.target

[Service]
User=usuario
Group=grupo
WorkingDirectory=/path/to/sistema_monitoramento
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python main_web.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Ativar e iniciar serviço
sudo systemctl daemon-reload
sudo systemctl enable sistema-monitoramento
sudo systemctl start sistema-monitoramento
sudo systemctl status sistema-monitoramento
```

## 🔄 Atualizações

### 1. **Atualizar Código**
```bash
# Backup atual
cp -r . ../sistema_monitoramento_backup

# Atualizar via git
git pull origin main

# Instalar novas dependências
pip install -r requirements.txt

# Reiniciar serviço
sudo systemctl restart sistema-monitoramento
```

### 2. **Migração de Dados**
```bash
# Backup do histórico
cp data/history.json data/history.json.bak

# Verificar integridade
python -c "import json; json.load(open('data/history.json'))"
```

---

Este guia fornece todas as informações necessárias para uma instalação bem-sucedida do sistema em diferentes ambientes, desde desenvolvimento até produção.