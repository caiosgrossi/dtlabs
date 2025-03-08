import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights  # Utilizando a nova API
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição das transformações (mesmo pipeline usado no treinamento original)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Diretório onde estão as imagens organizadas por classe (cada subpasta é uma celebridade)
data_dir = './post-processed/'

# Criação do dataset usando ImageFolder para classificação
dataset = ImageFolder(root=data_dir, transform=transform)
num_classes = len(dataset.classes)

# Divisão do dataset em treino (80%) e validação (20%)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Implementação do ArcFace Loss
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        # Parâmetro de pesos para as classes
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normaliza os embeddings e os pesos
        normalized_weights = nn.functional.normalize(self.weight, dim=1)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        # Calcula a similaridade (cosine) entre os embeddings e os pesos
        cosine = torch.matmul(embeddings, normalized_weights.t())
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        # Calcula theta e adiciona a margem ao ângulo
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        # Cria one-hot encoding para os rótulos
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        # Substitui as logits dos rótulos verdadeiros pelas ajustadas (com margem)
        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.scale
        loss = nn.functional.cross_entropy(output, labels)
        return loss

# Modelo de reconhecimento facial com ArcFace (para treinamento)
class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        # A camada final é substituída para gerar o embedding
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        embeddings = self.backbone(x)
        # Normaliza os embeddings para que tenham norma 1
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

model = FaceRecognitionModel(embedding_dim=128, num_classes=num_classes).to(device)
arcface_loss = ArcFaceLoss(embedding_dim=128, num_classes=num_classes, margin=0.5, scale=64.0).to(device)

# Otimizador (ajusta os parâmetros tanto do modelo quanto do ArcFace)
optimizer = optim.Adam(list(model.parameters()) + list(arcface_loss.parameters()), lr=0.001)

# Listas para armazenar os valores de loss por época
train_losses = []
val_losses = []

# Loop de treinamento com validação e tqdm para acompanhar o progresso
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Treino Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        embeddings = model(images)
        loss = arcface_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validação Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            loss = arcface_loss(embeddings, labels)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Treino Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Salva o modelo treinado
torch.save(model.state_dict(), "face_recognition_model_arcface3.pth")
print("Modelo salvo como face_recognition_model_arcface3.pth")

# Gera o gráfico de loss e salva como imagem
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ArcFace Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"arcface_loss_plot_epochs_{num_epochs}_batch_{batch_size}.png")
plt.close()
print("Gráfico de loss salvo.")