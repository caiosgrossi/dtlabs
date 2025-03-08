import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis  # Biblioteca com ArcFace pré-treinado

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializa o modelo ArcFace pré-treinado
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)  # Configura para usar GPU se disponível

# Diretório das imagens organizadas por classe (cada pasta é uma celebridade)
data_dir = './post-processed/'

# Listar todas as imagens no dataset
image_paths = []
celebrity_names = []
for celeb_name in os.listdir(data_dir):
    celeb_folder = os.path.join(data_dir, celeb_name)
    if os.path.isdir(celeb_folder):
        for img_name in os.listdir(celeb_folder):
            img_path = os.path.join(celeb_folder, img_name)
            image_paths.append(img_path)
            celebrity_names.append(celeb_name)

# Função para processar imagem e obter embeddings do ArcFace pré-treinado
def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    faces = app.get(img)  # Detecta o rosto e extrai o embedding
    if len(faces) > 0:
        return faces[0].embedding  # Pega o primeiro rosto detectado
    else:
        return None  # Se não detectar rosto, retorna None

# Geração dos embeddings
results = []
for img_path, celeb_name in tqdm(zip(image_paths, celebrity_names), total=len(image_paths), desc="Gerando embeddings"):
    embedding = get_embedding(img_path)
    if embedding is not None:
        emb_str = ','.join([f"{x:.4f}" for x in embedding])
        results.append({'imagem': img_path, 'celebridade': celeb_name, 'embedding': emb_str})

# Salva os resultados em um arquivo CSV
df = pd.DataFrame(results)
csv_filename = "celebridades_embeddings_arcface_pretrained.csv"
df.to_csv(csv_filename, index=False)
print(f"CSV salvo com sucesso em: {csv_filename}")
