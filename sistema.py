import random
import time
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """Arquitetura EXATA do modelo treinado"""
    def __init__(self):
        super().__init__()
        # Bloco Convolucional 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Bloco Convolucional 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Pooling e regularização
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        # Camadas densas
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class SensorChuva:
    def __init__(self):
        self.dia = 1
        print("🔍 CARREGANDO SISTEMA...")
        self.modelo = self.carregar_modelo_correto()
        self.imagens = self.listar_imagens()
    
    def carregar_modelo_correto(self):
        """Carrega modelo com arquitetura correta"""
        print("\n🤖 CARREGANDO MODELO CNN:")
        
        caminhos = [
            "modelo/detector.pth",
            "cnn/modelo/modelo_treinado.pth", 
            "cnn/modelo/detector.pth"
        ]
        
        checkpoint = None
        for caminho in caminhos:
            if os.path.exists(caminho):
                try:
                    checkpoint = torch.load(caminho, map_location='cpu')
                    print(f"✅ Modelo encontrado: {caminho}")
                    break
                except Exception as e:
                    print(f"❌ Erro em {caminho}: {e}")
                    continue
        
        if checkpoint is None:
            print("❌ Modelo não encontrado!")
            return None
        
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("✅ Checkpoint válido encontrado")
                print(f"   - Acurácia: {checkpoint.get('accuracy', 'N/A'):.1f}%")
                print(f"   - Épocas: {checkpoint.get('epochs', 'N/A')}")
                
                modelo = CNN()
                modelo.load_state_dict(checkpoint['model_state_dict'])
                modelo.eval()
                
                print("✅ Modelo carregado com sucesso!")
                return modelo
            else:
                print("❌ Formato de checkpoint inválido")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            return None
    
    def listar_imagens(self):
        """Lista imagens disponíveis"""
        os.makedirs("cnn/imagens", exist_ok=True)
        
        extensoes = ["*.jpg", "*.jpeg", "*.png"]
        imagens = []
        
        for ext in extensoes:
            encontradas = glob.glob(f"cnn/imagens/{ext}")
            imagens.extend(encontradas)
        
        print(f"📷 {len(imagens)} imagens encontradas")
        return imagens
    
    def analisar_imagem(self, caminho_imagem):
        """Analisa imagem com modelo real - INTERPRETAÇÃO CORRIGIDA"""
        if self.modelo is None:
            return self.simular_cnn()
        
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Transformações do treinamento
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
            
            # Processa imagem
            imagem = Image.open(caminho_imagem).convert('RGB')
            imagem_tensor = transform(imagem).unsqueeze(0)
            
            # Inferência
            with torch.no_grad():
                output = self.modelo(imagem_tensor)
                confianca = output.item()
            
            # ✅ INTERPRETAÇÃO CORRETA:
            # 0 = com_pessoas = PRESENÇA CONFIRMADA
            # 1 = sem_pessoas = AUSENTE
            if confianca <= 0.5:
                resultado = "PRESENÇA CONFIRMADA"
                print(f"🎯 CNN Real: {resultado} (confiança: {confianca:.3f})")
            else:
                resultado = "AUSENTE"
                print(f"🎯 CNN Real: {resultado} (confiança: {confianca:.3f})")
            
            return resultado
        
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return self.simular_cnn()
    
    def simular_cnn(self):
        """Simulação quando modelo não disponível"""
        resultado = "PRESENÇA CONFIRMADA" if random.random() < 0.3 else "AUSENTE"
        print(f"🎲 Simulação: {resultado}")
        return resultado
    
    def simular_sensor(self):
        return random.randint(1, 100)
    
    def processar_dia(self):
        """Processa um dia completo"""
        nivel_chuva = self.simular_sensor()
        
        alagamento = "Não"
        pessoas_risco = "Não"
        classificacao = "Normal"
        imagem_usada = "nenhuma"
        
        if nivel_chuva >= 50:
            alagamento = "Sim"
            
            if self.imagens:
                imagem_selecionada = random.choice(self.imagens)
                imagem_usada = os.path.basename(imagem_selecionada)
                resultado_cnn = self.analisar_imagem(imagem_selecionada)
            else:
                resultado_cnn = self.simular_cnn()
            
            if resultado_cnn == "PRESENÇA CONFIRMADA":
                pessoas_risco = "Sim"
                classificacao = "Perigo"
            else:
                classificacao = "Atenção"
        
        self.exibir_resultado(nivel_chuva, classificacao, pessoas_risco, 
                            alagamento, imagem_usada)
    
    def exibir_resultado(self, nivel_chuva, classificacao, pessoas_risco, 
                        alagamento, imagem_usada):
        """Exibe resultado na tela"""
        print("\n" + "="*50)
        print(f"🌧️  Dia: {self.dia}")
        print(f"📊 Classificação: {classificacao}")
        print(f"👥 Pessoas em risco: {pessoas_risco}")
        print(f"🌊 Alagamento: {alagamento}")
        print(f"☔ Nível de Chuva: {nivel_chuva} mm")
        print(f"📷 Imagem: {imagem_usada}")
        print("="*50)
        
        print("⏱️  Aguardando 15 segundos...")
        time.sleep(15)
        
        self.dia += 1
    
    def executar_loop(self):
        """Loop principal"""
        print("\n🚀 SISTEMA DE MONITORAMENTO INICIADO")
        print("⏹️  Pressione Ctrl+C para parar\n")
        
        try:
            while True:
                self.processar_dia()
        except KeyboardInterrupt:
            print("\n\n🛑 Sistema parado!")

# Execução
if __name__ == "__main__":
    sensor = SensorChuva()
    sensor.executar_loop()
