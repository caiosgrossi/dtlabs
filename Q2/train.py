import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet18, ResNet18_Weights  # Utilizando a nova API
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm  
import matplotlib.pyplot as plt  # Para gerar o plot

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição das transformações (pré-processamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset customizado para gerar triplets com filtragem de classes com menos de 2 imagens
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Utiliza ImageFolder para organizar as imagens por classe (nome da celebridade)
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        self.transform = transform

        # Conta as imagens por classe e filtra classes com pelo menos 2 imagens
        class_counts = {}
        for _, label in self.dataset.imgs:
            class_counts[label] = class_counts.get(label, 0) + 1
        self.imgs = [item for item in self.dataset.imgs if class_counts[item[1]] >= 2]

        # As classes permanecem as mesmas, mas agora self.imgs contém apenas imagens de classes válidas
        self.classes = self.dataset.classes

        # Cria um dicionário mapeando cada classe para os índices das imagens filtradas correspondentes
        self.class_indices = {}
        for idx, (path, label) in enumerate(self.imgs):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        # Carrega a imagem âncora
        anchor_path, anchor_label = self.imgs[index]
        anchor_img = self.transform(Image.open(anchor_path).convert("RGB"))
        
        # Seleciona uma imagem positiva (mesma classe, diferente da âncora)
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.class_indices[anchor_label])
        positive_path, _ = self.imgs[positive_index]
        positive_img = self.transform(Image.open(positive_path).convert("RGB"))
        
        # Seleciona uma imagem negativa (classe diferente)
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_indices.keys()))
        negative_index = random.choice(self.class_indices[negative_label])
        negative_path, _ = self.imgs[negative_index]
        negative_img = self.transform(Image.open(negative_path).convert("RGB"))
        
        # Retorna também o caminho da imagem âncora e o label, úteis para verificação
        return anchor_img, positive_img, negative_img, anchor_path, anchor_label

# Diretório onde estão as imagens (estrutura: ./post-processed/<celebridade>/imagem.jpg)
data_dir = './post-processed/'

# Criação do dataset filtrado para treinamento com triplets
dataset = TripletFaceDataset(root_dir=data_dir, transform=transform)

# Divisão do dataset em treino (80%) e validação (20%)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Definição do modelo de extração de embeddings
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingModel, self).__init__()
        # Utiliza ResNet18 pré-treinado como backbone utilizando o novo parâmetro "weights"
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        # Substitui a última camada fully-connected para gerar o embedding desejado
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        embedding = self.backbone(x)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

model = FaceEmbeddingModel(embedding_dim=128).to(device)

# Função de perda: Triplet Margin Loss
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# Otimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Listas para armazenar os valores de loss por época
train_losses = []
val_losses = []

# Loop de treinamento com validação e barras de progresso
num_epochs = 15  # ajuste conforme necessário
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Treinamento Epoch {epoch+1}/{num_epochs}"):
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
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Validação
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validação Epoch {epoch+1}/{num_epochs}"):
            anchor, positive, negative, _, _ = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Treino Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Salva o modelo treinado
torch.save(model.state_dict(), "face_embedding_model2.pth")
print("Modelo salvo como face_embedding_model.pth")

# Gera os embeddings de todas as imagens e cria um CSV.
# Para essa etapa, usamos o ImageFolder original (sem filtragem) para obter todos os dados
full_dataset = ImageFolder(root=data_dir, transform=transform)
full_dataloader = DataLoader(full_dataset, batch_size=32, shuffle=False)

model.eval()
results = []
with torch.no_grad():
    for images, labels in tqdm(full_dataloader, desc="Extraindo embeddings"):
        images = images.to(device)
        embeddings = model(images)
        embeddings = embeddings.cpu().numpy()
        for i in range(images.size(0)):
            idx = len(results)
            img_path, label = full_dataset.imgs[idx]
            celeb_name = full_dataset.classes[label]
            emb_str = ','.join([f"{x:.4f}" for x in embeddings[i]])
            results.append({'imagem': img_path, 'celebridade': celeb_name, 'embedding': emb_str})

df = pd.DataFrame(results)
df.to_csv('celebridades_embeddings.csv', index=False)
print("CSV salvo com sucesso!")

# Gera o gráfico de perda e salva como imagem
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss de Treino e Validação por Epoch")
plt.legend()
plt.grid(True)
plt.savefig(f"loss_plot_epochs_{num_epochs}_batch_{batch_size}.png")
plt.close()
print("Gráfico de loss salvo como loss_plot.png")
