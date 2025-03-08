import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import trimesh

def remap_axes_custom(points, order=[2, 0, 1], signs=[1, 1, -1]):
    """
    Reordena as colunas de 'points' de acordo com 'order' e multiplica cada coluna pelo respectivo
    valor em 'signs'.
    
    Por exemplo, com order = [0, 2, 1] e signs = [-1, -1, 1], 
    o ponto (x, y, z) se transforma em (-x, -z, y).
    
    Parameters:
      points: array de shape (N, 3)
      order: lista de 3 inteiros indicando a nova ordem dos eixos.
      signs: lista de 3 inteiros (1 ou -1) para aplicar inversões de sinal.
    
    Returns:
      points_remapped: array de shape (N, 3) com os eixos remapeados.
    """
    points = np.array(points)
    remapped = points[:, order]
    remapped = remapped * np.array(signs)
    return remapped

def load_point_cloud_o3d(file_path, order=[2, 1, 0], signs=[1, -1, 1]):
    """
    Carrega o arquivo OBJ usando Trimesh, extrai os vértices,
    remapeia os eixos conforme os parâmetros e retorna um objeto PointCloud do Open3D.
    """
    mesh = trimesh.load(file_path, process=False)
    vertices = mesh.vertices
    if vertices is None or len(vertices) == 0:
        raise ValueError(f"A malha carregada de {file_path} não possui vértices.")
    # Remapeia os eixos conforme desejado
    vertices_remapped = remap_axes_custom(vertices)
    # Cria e retorna um objeto PointCloud do Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_remapped)
    return pcd

def translational_error(T_est, T_gt):
    pos_est = T_est[:3, 3]
    pos_gt = T_gt[:3, 3]
    return np.linalg.norm(pos_est - pos_gt)

# Configurações e caminhos
point_cloud_dir = "objetos/KITTI-Sequence"
ground_truth_file = "./ground_truth.npy"
num_scans = 30
threshold = 1.0
voxel_size = 0.1  # ajuste conforme necessário

if os.path.exists(ground_truth_file):
    ground_truth = np.load(ground_truth_file)
else:
    ground_truth = None
    print("Ground truth não encontrada.")

trajectory = [np.eye(4)]

for i in tqdm(range(1, num_scans), desc="Processando scans"):
    source_file = os.path.join(point_cloud_dir, f"{i-1:06d}", f"{i-1:06d}_points.obj")
    target_file = os.path.join(point_cloud_dir, f"{i:06d}", f"{i:06d}_points.obj")
    
    if not os.path.exists(source_file) or not os.path.exists(target_file):
        tqdm.write(f"Arquivo não encontrado: {source_file} ou {target_file}")
        continue

    # Carregar, remapear e downsample os point clouds
    source_pcd = load_point_cloud_o3d(source_file, order=[2,1,0], signs=[1,-1,1]).voxel_down_sample(voxel_size)
    target_pcd = load_point_cloud_o3d(target_file, order=[2,1,0], signs=[1,-1,1]).voxel_down_sample(voxel_size)
    
    init_transformation = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T_rel = reg_p2p.transformation
    T_acc = T_rel @ trajectory[-1]
    trajectory.append(T_acc)
    
    if ground_truth is not None and i < ground_truth.shape[0]:
        T_gt = ground_truth[i]
        trans_error = translational_error(T_acc, T_gt)
        tqdm.write(f"Scan {i}: Erro translacional = {trans_error:.3f} metros")

# Plotar a trajetória
positions_est = np.array([T[:3, 3] for T in trajectory])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
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

fig.savefig("trajetoria_completa.png", dpi=300)
plt.show()
