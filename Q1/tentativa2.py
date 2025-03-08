import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # para barras de progresso

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_point_cloud(file_path):
    """Carrega a nuvem de pontos usando Trimesh e retorna os vértices como tensor no dispositivo configurado."""
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    return torch.tensor(vertices, dtype=torch.float32, device=device)

def compute_min_indices(src, target, batch_size=262144):
    indices_list = []
    # Barra de progresso para o loop dos batches
    for i in tqdm(range(0, src.shape[0], batch_size), desc="Processando batches no ICP", leave=False):
        src_batch = src[i:i+batch_size]
        # Calcular as distâncias apenas para o batch atual
        d = torch.cdist(src_batch, target)
        # Obter os índices dos mínimos para o batch
        indices_batch = torch.argmin(d, dim=1)
        indices_list.append(indices_batch)
        # Libera memória do batch
        del src_batch, d, indices_batch
        torch.cuda.empty_cache()
    return torch.cat(indices_list, dim=0)

def icp(source, target, max_iter=50, tol=1e-5):
    T_total = torch.eye(4, device=device)
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
    
    return T_total

# Defina o diretório base dos arquivos
point_cloud_dir = "objetos/KITTI-Sequence"  # Diretório base onde estão as pastas dos scans
ground_truth_file = "./ground_truth.npy"  # Mantenha ou ajuste se necessário

num_scans = 30
trajectory = [torch.eye(4, device=device)]

# Carregar ground truth se disponível
if os.path.exists(ground_truth_file):
    ground_truth = np.load(ground_truth_file)  # Espera-se um array de tamanho (num_scans, 4, 4)
else:
    ground_truth = None
    print("Arquivo de ground truth não encontrado.")

# Função para computar o erro de translação
def translational_error(T_est, T_gt):
    pos_est = T_est[:3, 3]
    pos_gt = T_gt[:3, 3]
    return np.linalg.norm(pos_est - pos_gt)

# Função para computar o erro rotacional (em graus)
def rotational_error(T_est, T_gt):
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    R_err = R_est.T @ R_gt
    # Calcular o ângulo a partir do traço da matriz de erro
    angle = np.arccos((np.trace(R_err) - 1) / 2)
    return np.degrees(angle)

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

# Converter a trajetória para CPU e extrair posições para plotagem
trajectory_cpu = [T.cpu().numpy() for T in trajectory]
positions_est = np.array([T[:3, 3] for T in trajectory_cpu])

import numpy as np
import matplotlib.pyplot as plt

def set_axes_equal(ax):
    """
    Ajusta os limites do gráfico 3D para ter a mesma escala em X, Y e Z.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    
    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

def plot_trajectory(trajectory, ground_truth=None, output_file="meu_grafico.png"):
    # Converter a trajetória para CPU e extrair posições para plotagem
    trajectory_cpu = [T.cpu().numpy() for T in trajectory]
    positions_est = np.array([T[:3, 3] for T in trajectory_cpu])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot da trajetória estimada
    ax.plot(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2],
            marker='o', linestyle='-', label='Estimado')

    # Se a ground truth estiver disponível, extraia as posições e plote
    if ground_truth is not None:
        positions_gt = np.array([T[:3, 3] for T in ground_truth])
        ax.plot(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2],
                marker='^', linestyle='--', label='Ground Truth')

    ax.set_title("Trajetória Estimada vs Ground Truth")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Ajusta o gráfico para ter a mesma escala em X, Y e Z
    set_axes_equal(ax)

    # Salvar a figura com resolução de 300 dpi
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

plot_trajectory(trajectory, ground_truth, "meu_grafico.png")

# Calcular e imprimir o erro médio de translação
if ground_truth is not None:
    errors = []
    for T_est, T_gt in zip(trajectory_cpu, ground_truth):
        errors.append(translational_error(T_est, T_gt))
    print("Erro médio de translação:", np.mean(errors))
else:
    print("Arquivo de ground truth não encontrado.")



