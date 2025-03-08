import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Definir dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def remap_axes_custom(points, order=[2, 0, 1], signs=[1, 1, -1]):
    """
    Reordena as colunas de 'points' de acordo com 'order' e multiplica cada coluna
    pelo respectivo valor em 'signs'.
    
    Por exemplo, com order = [0, 2, 1] e signs = [-1, -1, 1], 
    o ponto (x, y, z) se transforma em (-x, -z, y).
    
    Parameters:
      points: array de shape (N, 3) com as coordenadas dos pontos.
      order: lista de 3 inteiros indicando a nova ordem dos eixos.
      signs: lista de 3 inteiros (1 ou -1) para aplicar inversões de sinal.
    
    Returns:
      points_remapped: array de shape (N, 3) com os eixos remapeados.
    """
    points = np.array(points)
    remapped = points[:, order]
    remapped = remapped * np.array(signs)
    return remapped

def load_point_cloud(file_path):
    """
    Carrega a nuvem de pontos usando Trimesh, remapeia os eixos e retorna os vértices
    como tensor no dispositivo configurado (GPU ou CPU).
    
    Parameters:
      file_path: caminho para o arquivo de nuvem de pontos (.obj).
    
    Returns:
      vertices_remapped: tensor contendo as coordenadas dos pontos remapeados.
    """
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    vertices_remapped = remap_axes_custom(vertices)
    return torch.tensor(vertices_remapped, dtype=torch.float32, device=device)

def compute_min_indices(src, target, batch_size=25000):
    """
    Calcula os índices dos pontos mais próximos de 'src' em relação a 'target' utilizando
    a distância euclidiana.
    
    Parameters:
      src: tensor contendo os pontos da nuvem de origem.
      target: tensor contendo os pontos da nuvem de destino.
      batch_size: tamanho do batch para processamento em paralelo.
    
    Returns:
      indices: tensor com os índices dos pontos mais próximos.
    """
    indices_list = []
    for i in tqdm(range(0, src.shape[0], batch_size), desc="Processando batches no ICP", leave=False):
        src_batch = src[i:i+batch_size]
        d = torch.cdist(src_batch, target)
        indices_batch = torch.argmin(d, dim=1)
        indices_list.append(indices_batch)
        del src_batch, d, indices_batch
        torch.cuda.empty_cache()
    return torch.cat(indices_list, dim=0)

def icp(source, target, max_iter=10, tol=1e-5):
    """
    Algoritmo ICP (Iterative Closest Point) para registrar duas nuvens de pontos.
    
    Parameters:
      source: tensor contendo os pontos da nuvem de origem.
      target: tensor contendo os pontos da nuvem de destino.
      max_iter: número máximo de iterações do ICP.
      tol: tolerância para o critério de convergência.
    
    Returns:
      T_total: matriz de transformação acumulada no formato 4x4.
    """
    T_total = torch.eye(4, device=device)  # Inicializa matriz de transformação como identidade
    src = source.clone()
    
    for iter_num in range(max_iter):
        indices = compute_min_indices(src, target)
        target_corr = target[indices]
        
        centroid_src = src.mean(dim=0)
        centroid_target = target_corr.mean(dim=0)
        
        src_centered = src - centroid_src
        target_centered = target_corr - centroid_target
        
        H = src_centered.t() @ target_centered
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.t() @ U.t()
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.t() @ U.t()
            
        t = centroid_target - R @ centroid_src
        
        # Matriz de transformação 4x4 homogênea
        T = torch.eye(4, device=device)
        T[:3, :3] = R
        T[:3, 3] = t
        
        ones = torch.ones(src.shape[0], 1, device=device)
        src_h = torch.cat([src, ones], dim=1)
        src_h = (T @ src_h.t()).t()
        src = src_h[:, :3]
        
        T_total = T @ T_total
        
        if S.mean() < tol:
            break
        
        del indices, target_corr, centroid_src, centroid_target, src_centered, target_centered, H, U, S, Vt, R, t, ones, src_h
        torch.cuda.empty_cache()
    
    print(T.shape)
    return T_total

def translational_error(T_est, T_gt):
    """
    Calcula o erro translacional entre duas matrizes de transformação.
    
    Parameters:
      T_est: matriz de transformação estimada.
      T_gt: matriz de transformação de ground truth.
    
    Returns:
      erro: erro translacional calculado como a norma da diferença de posições.
    """
    pos_est = T_est[:3, 3]
    pos_gt = T_gt[:3, 3]
    return np.linalg.norm(pos_est - pos_gt)

def rotational_error(T_est, T_gt):
    """
    Calcula o erro rotacional entre duas matrizes de transformação em graus.
    
    Parameters:
      T_est: matriz de transformação estimada.
      T_gt: matriz de transformação de ground truth.
    
    Returns:
      erro: erro rotacional em graus.
    """
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    R_err = R_est.T @ R_gt
    angle = np.arccos((np.trace(R_err) - 1) / 2)
    return np.degrees(angle)

def plot_trajectory(trajectory, ground_truth=None, output_file="meu_grafico.png"):
    """
    Plota a trajetória estimada e, opcionalmente, a ground truth.
    
    Parameters:
      trajectory: lista de matrizes de transformação (trajetória estimada).
      ground_truth: matriz de ground truth, se disponível.
      output_file: nome do arquivo de saída da imagem do gráfico.
    """
    trajectory_cpu = [T.cpu().numpy() for T in trajectory]
    positions_est = np.array([T[:3, 3] for T in trajectory_cpu])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot da trajetória estimada
    ax.plot(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2],
            marker='o', linestyle='-', label='Estimado')

    if ground_truth is not None:
        positions_gt = np.array([T[:3, 3] for T in ground_truth])
        ax.plot(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2],
                marker='^', linestyle='--', label='Ground Truth')

    ax.set_title("Trajetória Estimada vs Ground Truth")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Salvar a figura com resolução de 300 dpi
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

# Definindo o diretório base dos arquivos
point_cloud_dir = "objetos/KITTI-Sequence"
ground_truth_file = "./ground_truth.npy"

num_scans = 30
trajectory = [torch.eye(4, device=device)]  # Inicializa com a matriz identidade

# Carregar ground truth se disponível
if os.path.exists(ground_truth_file):
    ground_truth = np.load(ground_truth_file)
else:
    ground_truth = None
    print("Arquivo de ground truth não encontrado.")

# Loop principal com tqdm e impressão dos erros a cada scan
for i in tqdm(range(1, num_scans), desc="Processando scans"):
    source_file = os.path.join(point_cloud_dir, f"{i-1:06d}", f"{i-1:06d}_points.obj")
    target_file = os.path.join(point_cloud_dir, f"{i:06d}", f"{i:06d}_points.obj")
    
    if not os.path.exists(source_file) or not os.path.exists(target_file):
        tqdm.write(f"Arquivo não encontrado: {source_file} ou {target_file}")
        continue

    source = load_point_cloud(source_file)
    target = load_point_cloud(target_file)

    T = icp(source, target)
    T_acc = T @ trajectory[-1]
    trajectory.append(T_acc)
    
    # Se ground truth estiver disponível, calcular e imprimir os erros
    if ground_truth is not None and i < ground_truth.shape[0]:
        T_gt = ground_truth[i]
        T_est = T_acc.cpu().numpy()
        trans_error = translational_error(T_est, T_gt)
        rot_error = rotational_error(T_est, T_gt)
        tqdm.write(f"Scan {i}: Erro translacional = {trans_error:.3f}, Erro rotacional = {rot_error:.3f}°")
    
    del source, target, T
    torch.cuda.empty_cache()

# Plot da trajetória
plot_trajectory(trajectory, ground_truth, "meu_grafico3.png")
