import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição das transformações (mesmo pipeline usado no treinamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Definição do modelo de reconhecimento facial (usado com ArcFace no treinamento)
class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=None):
        super(FaceRecognitionModel, self).__init__()
        # Usamos ResNet18 pré-treinado como backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        # Substituímos a última camada por uma camada que gera embeddings
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        embeddings = self.backbone(x)
        # Normaliza os embeddings para que tenham norma 1
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

# Carrega o modelo treinado
model_path = "face_recognition_model_arcface2.pth"  # ajuste conforme necessário
model = FaceRecognitionModel(embedding_dim=128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define o diretório das imagens (estrutura: ./post-processed/<celebridade>/imagem.jpg)
data_dir = './post-processed/'
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

results = []
with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Gerando embeddings"):
        images = images.to(device)
        embeddings = model(images)
        embeddings = embeddings.cpu().numpy()
        # Para cada imagem no batch, recupera o caminho e o nome da celebridade
        for i in range(images.size(0)):
            idx = len(results)
            img_path, label = dataset.imgs[idx]
            celeb_name = dataset.classes[label]
            emb_str = ','.join([f"{x:.4f}" for x in embeddings[i]])
            results.append({'imagem': img_path, 'celebridade': celeb_name, 'embedding': emb_str})

# Salva os resultados em um arquivo CSV
df = pd.DataFrame(results)
csv_filename = "celebridades_embeddings_arcface2.csv"
df.to_csv(csv_filename, index=False)
print(f"CSV salvo com sucesso em: {csv_filename}")
