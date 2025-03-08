import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição da transformação (mesmo pipeline usado no treinamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Definição do modelo (mesmo utilizado no treinamento)
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingModel, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        embedding = self.backbone(x)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

def load_image(image_path):
    """Carrega uma imagem e a converte para RGB."""
    return Image.open(image_path).convert("RGB")

def get_embedding(model, image, device):
    """Gera o embedding para uma imagem usando o modelo."""
    model.eval()
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0).to(device)  # Adiciona dimensão de batch
        embedding = model(img_tensor)
    return embedding.cpu().numpy().flatten()

def load_database(csv_path):
    """
    Carrega o CSV com o banco de dados e converte a coluna 'embedding'
    (armazenada como string) em um array NumPy.
    """
    df = pd.read_csv(csv_path)
    embeddings = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processando embeddings do CSV"):
        emb_str = row['embedding']
        emb_arr = np.array([float(x) for x in emb_str.split(",")])
        embeddings.append(emb_arr)
    embeddings = np.vstack(embeddings)  # shape: (n_samples, embedding_dim)
    return df, embeddings

def compute_mean_embeddings(df):
    """
    Calcula a média dos embeddings para cada celebridade.
    Retorna um dicionário: {celebridade: mean_embedding}
    """
    mean_embeddings = {}
    for celeb in tqdm(df['celebridade'].unique(), desc="Calculando médias dos embeddings"):
        rows = df[df['celebridade'] == celeb]
        embeddings = []
        for idx, row in rows.iterrows():
            emb_arr = np.array([float(x) for x in row['embedding'].split(",")])
            embeddings.append(emb_arr)
        mean_embeddings[celeb] = np.mean(embeddings, axis=0)
    return mean_embeddings

def find_matches_mean(query_embedding, mean_embeddings, top_k=3):
    """
    Compara o embedding da query com os embeddings médios de cada celebridade,
    calculando a distância Euclidiana. Retorna os top_k matches como uma lista
    de tuplas (celebridade, score).
    """
    distances = {}
    for celeb, mean_emb in mean_embeddings.items():
         dist = np.linalg.norm(query_embedding - mean_emb)
         distances[celeb] = dist
    sorted_celebs = sorted(distances.items(), key=lambda x: x[1])
    return sorted_celebs[:top_k]

def score_for_specific_celebrity(query_embedding, mean_embeddings, celebrity_name):
    """
    Calcula o score (distância Euclidiana) entre o embedding da query e
    o embedding médio da celebridade especificada.
    """
    if celebrity_name not in mean_embeddings:
        print(f"Celebridade '{celebrity_name}' não encontrada no banco de dados.")
        return None
    score = np.linalg.norm(query_embedding - mean_embeddings[celebrity_name])
    print(f"Score para a celebridade '{celebrity_name}': {score:.4f}")
    return score

def display_results(query_image, df, matches):
    """
    Exibe a imagem de consulta e, para cada celebridade encontrada, exibe uma imagem
    representativa (a primeira encontrada no CSV) com o score (distância).
    Salva o resultado em 'match_results.png'.
    """
    top_k = len(matches)
    plt.figure(figsize=(15, 5))
    
    # Exibe a imagem de consulta
    plt.subplot(1, top_k+1, 1)
    plt.imshow(query_image)
    plt.title("Imagem Consulta")
    plt.axis("off")
    
    # Para cada celebridade, exibe uma imagem representativa e o score
    for i, (celeb, score) in enumerate(matches):
        # Seleciona a primeira imagem da celebridade no CSV como representativa
        row = df[df['celebridade'] == celeb].iloc[0]
        match_path = row['imagem']
        match_img = load_image(match_path)
        plt.subplot(1, top_k+1, i+2)
        plt.imshow(match_img)
        plt.title(f"{celeb}\nScore: {score:.4f}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("match_results.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encontra as correspondências comparando o embedding da imagem de consulta com a média dos embeddings de cada celebridade e, opcionalmente, mede o score para uma celebridade específica."
    )
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem de consulta (.jpg)")
    parser.add_argument("--model", type=str, default="face_embedding_model.pth", help="Caminho para o modelo treinado")
    parser.add_argument("--csv", type=str, default="celebridades_embeddings.csv", help="Caminho para o CSV do banco de dados")
    parser.add_argument("--topk", type=int, default=3, help="Número de correspondências a retornar")
    parser.add_argument("--celebrity", type=str, default=None, help="Nome da celebridade para medir o score")
    args = parser.parse_args()
    
    # Carrega a imagem de consulta
    query_image = load_image(args.image)
    
    # Carrega o modelo treinado
    model = FaceEmbeddingModel(embedding_dim=128).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Gera o embedding da imagem de consulta
    query_embedding = get_embedding(model, query_image, device)
    
    # Carrega o banco de dados de embeddings
    df, _ = load_database(args.csv)
    
    # Calcula a média dos embeddings para cada celebridade
    mean_embeddings = compute_mean_embeddings(df)
    
    # Encontra as top_k correspondências comparando com os embeddings médios
    matches = find_matches_mean(query_embedding, mean_embeddings, top_k=args.topk)
    
    # Exibe os resultados
    print("Top correspondências (baseadas na média dos embeddings):")
    for rank, (celeb, score) in enumerate(matches):
        rep_img = df[df['celebridade'] == celeb].iloc[0]['imagem']
        print(f"{rank+1}: {celeb} - Score (distância): {score:.4f} - Imagem representativa: {rep_img}")
    
    display_results(query_image, df, matches)
    
    # Se o argumento --celebrity for fornecido, mede e imprime o score para essa celebridade
    if args.celebrity is not None:
        score_for_specific_celebrity(query_embedding, mean_embeddings, args.celebrity)
