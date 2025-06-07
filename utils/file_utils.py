"""
Utilitários para manipulação de arquivos
"""
import os
import glob

def list_images(directory, extensions=None):
    """Lista todas as imagens em um diretório"""
    if extensions is None:
        extensions = ["*.jpg", "*.jpeg", "*.png"]
    
    # Cria diretório se não existir
    os.makedirs(directory, exist_ok=True)
    
    images = []
    for ext in extensions:
        found = glob.glob(os.path.join(directory, ext))
        images.extend(found)
    
    return images

def ensure_directory_exists(path):
    """Garante que um diretório existe"""
    os.makedirs(path, exist_ok=True)
