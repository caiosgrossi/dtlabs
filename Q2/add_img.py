import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import argparse

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição da transformação (mesma usada no treinamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Definição do modelo para extração de embeddings
class FaceEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingModel, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        embedding = self.backbone(x)
        # Normaliza o embedding para que tenha norma 1
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

def load_model(model_path, embedding_dim=128):
    model = FaceEmbeddingModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_embedding(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Adiciona dimensão de batch
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy().flatten()

def append_embedding(csv_file, image_path, name, embedding):
    emb_str = ','.join([f"{x:.4f}" for x in embedding])
    new_row = {'imagem': image_path, 'celebridade': name, 'embedding': emb_str}
    
    # Se o CSV já existir, carrega e adiciona a nova linha; caso contrário, cria um novo DataFrame
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(csv_file, index=False)
    print(f"Novo registro adicionado ao CSV: {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adiciona uma nova pessoa à base de dados de embeddings."
    )
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem da nova pessoa")
    parser.add_argument("--name", type=str, required=True, help="Nome da nova pessoa")
    parser.add_argument("--model", type=str, default="face_embedding_model2", help="Caminho para o modelo treinado")
    parser.add_argument("--csv", type=str, required=True, help="Arquivo CSV para salvar os embeddings")
    args = parser.parse_args()
    
    # Carrega o modelo treinado
    model = load_model(args.model, embedding_dim=128)
    
    # Obtém o embedding da imagem fornecida
    embedding = get_embedding(model, args.image)
    
    # Adiciona (ou atualiza) o embedding da nova pessoa no CSV
    append_embedding(args.csv, args.image, args.name, embedding)
