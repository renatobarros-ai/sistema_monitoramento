"""
Arquitetura da CNN para detecção de pessoas em alagamentos
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """Arquitetura CNN original do treinamento"""
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

def load_model(model_path, device="cpu"):
    """Carrega modelo treinado"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = CNN()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, checkpoint
        else:
            raise ValueError("Formato de checkpoint inválido")
            
    except Exception as e:
        raise Exception(f"Erro ao carregar modelo: {e}")
