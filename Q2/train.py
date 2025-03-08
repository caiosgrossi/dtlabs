import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm  # Importa o tqdm para exibir barras de progresso

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição das transformações (pré-processamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset customizado para gerar triplets
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Utiliza ImageFolder para organizar as imagens por classe (nome da celebridade)
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        self.transform = transform
        self.classes = self.dataset.classes
        self.imgs = self.dataset.imgs

        # Cria um dicionário mapeando cada classe para os índices das imagens correspondentes
        self.class_indices = {}
        for idx, (path, label) in enumerate(self.imgs):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        # Imagem âncora
        anchor_img, anchor_label = self.dataset[index]
        
        # Seleciona uma imagem positiva (mesma classe, diferente da âncora)
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.class_indices[anchor_label])
        positive_img, _ = self.dataset[positive_index]
        
        # Seleciona uma imagem negativa (classe diferente)
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_indices.keys()))
        negative_index = random.choice(self.class_indices[negative_label])
        negative_img, _ = self.dataset[negative_index]
        
        # Retorna também o caminho da imagem âncora e o label, que podem ser úteis posteriormente
        return anchor_img, positive_img, negative_img, self.imgs[index][0], anchor_label

# Diretório onde estão as imagens (estrutura: /post-processed/<celebridade>/imagem.jpg)
data_dir = './post-processed/'

# Criação do dataset e do DataLoader para treinamento
dataset = TripletFaceDataset(root_dir=data_dir, transform=transform)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Definição do modelo de extração de embeddings
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingModel, self).__init__()
        # Utiliza ResNet18 pré-treinado como backbone
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        # Substitui a última camada fully-connected para gerar o embedding desejado
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        # Obtém o embedding e normaliza-o (opcional, mas útil para métricas de distância)
        embedding = self.backbone(x)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

model = FaceEmbeddingModel(embedding_dim=128).to(device)

# Função de perda: Triplet Margin Loss
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# Otimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop de treinamento com barra de progresso
num_epochs = 10  # ajuste conforme a complexidade do dataset e necessidade
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    # Cria uma barra de progresso para cada epoch
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        anchor, positive, negative, _, _ = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        emb_anchor = model(anchor)
        emb_positive = model(positive)
        emb_negative = model(negative)
        
        loss = criterion(emb_anchor, emb_positive, emb_negative)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Após o treinamento, gera os embeddings de todas as imagens e cria um CSV
# Para isso, usamos ImageFolder (sem gerar triplets) para percorrer todas as imagens
full_dataset = ImageFolder(root=data_dir, transform=transform)
full_dataloader = DataLoader(full_dataset, batch_size=32, shuffle=False)

model.eval()
results = []
with torch.no_grad():
    # Barra de progresso para a extração dos embeddings
    for images, labels in tqdm(full_dataloader, desc="Extraindo embeddings"):
        images = images.to(device)
        embeddings = model(images)
        embeddings = embeddings.cpu().numpy()
        # Itera sobre cada imagem do batch
        for i in range(images.size(0)):
            idx = len(results)
            img_path, label = full_dataset.imgs[idx]
            celeb_name = full_dataset.classes[label]
            # Converte o vetor de embedding em uma string separada por vírgulas
            emb_str = ','.join([f"{x:.4f}" for x in embeddings[i]])
            results.append({'imagem': img_path, 'celebridade': celeb_name, 'embedding': emb_str})

# Salva os resultados em um arquivo CSV
df = pd.DataFrame(results)
df.to_csv('celebridades_embeddings.csv', index=False)
print("CSV salvo com sucesso!")
