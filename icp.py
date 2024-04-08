
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ponto_mais_proximo(ponto, pontos, k=1):
    distancias = np.sum((pontos - ponto) ** 2, axis=1)
    indice = np.argsort(distancias)[:k]
    return pontos[indice]

def amostragem_aleatoria(pontos, n_amostras):
    indices_amostras = np.random.choice(len(pontos), size=n_amostras, replace=False)
    return pontos[indices_amostras]

def remover_outliers(pontos, desvio_padrao_limite=2):
    media = np.mean(pontos, axis=0)
    desvio_padrao = np.std(pontos, axis=0)
    pontos_filtrados = pontos[np.all(np.abs(pontos - media) < desvio_padrao_limite * desvio_padrao, axis=1)]
    return pontos_filtrados

def icp(pontos_alvo, pontos_referencia, max_iter=50, threshold=0.0001, n_amostras=100, k_vizinhos=1):
    T = np.eye(4)

    for _ in range(max_iter):
        pontos_amostrados_referencia = amostragem_aleatoria(pontos_referencia, n_amostras)
        pontos_amostrados_alvo = amostragem_aleatoria(pontos_alvo, n_amostras)

        pontos_transformados = np.dot(T, np.vstack([pontos_amostrados_referencia.T, np.ones(n_amostras)]))
        pontos_transformados = pontos_transformados[:3].T
        
        pares_pontos = []
        for i, ponto in enumerate(pontos_transformados):
            pontos_proximos = ponto_mais_proximo(ponto, pontos_amostrados_alvo, k=k_vizinhos)
            for ponto_proximo in pontos_proximos:
                pares_pontos.append([pontos_amostrados_referencia[i], ponto_proximo])

        pares_pontos = np.array(pares_pontos)
        pontos_referencia_corrigidos = pares_pontos[:, 0]
        pontos_alvo_corrigidos = pares_pontos[:, 1]

        T_novo = np.linalg.inv(np.dot(pontos_referencia_corrigidos.T, pontos_alvo_corrigidos))
        T = np.dot(T_novo, T)

        if np.linalg.norm(T_novo - np.eye(4)) < threshold:
            break

    return T

def carregar_ground_truth(caminho):
    return np.load(caminho)

def carregar_scans(caminhos):
    scans = []
    for caminho in caminhos:
        ponto = trimesh.load_mesh(caminho + ".obj").vertices
        scans.append(ponto)
    return scans

# Caminho para a ground-truth
caminho_ground_truth = "ground_truth.npy"
# Caminhos para os scans da nuvem de pontos
caminhos_scans = ["000000_points.obj", "000001_points.obj", ..., "000030_points.obj"]

ground_truth = carregar_ground_truth(caminho_ground_truth)
scans = carregar_scans(caminhos_scans)

# Remover outliers dos pontos alvo e de referência
scans = [remover_outliers(scan) for scan in scans]

# Executar o algoritmo ICP para cada par de scans consecutivos
trajetoria = [np.eye(4)]  # Inicializar a trajetória com a identidade (posição inicial)
for i in range(len(scans) - 1):
    T = icp(scans[i], scans[i+1])
    trajetoria.append(np.dot(T, trajetoria[-1]))

# Comparar com a ground-truth
# Calcula o erro entre duas transformações
def calcular_erro(transformacao_estimada, transformacao_real):
    erro_translacao = np.linalg.norm(transformacao_estimada[:3, 3] - transformacao_real[:3, 3])
    erro_rotacao = np.arccos((np.trace(np.dot(transformacao_estimada[:3, :3], transformacao_real[:3, :3].T)) - 1) / 2)
    return erro_translacao, np.degrees(erro_rotacao)

# Calcula o erro médio entre todas as estimativas e a ground-truth
erros_translacao = []
erros_rotacao = []
for i in range(len(trajetoria)):
    erro_translacao, erro_rotacao = calcular_erro(trajetoria[i], ground_truth[i])
    erros_translacao.append(erro_translacao)
    erros_rotacao.append(erro_rotacao)

erro_medio_translacao = np.mean(erros_translacao)
erro_medio_rotacao = np.mean(erros_rotacao)

print(f'Erro médio de translacão: {erro_medio_translacao} metros')
print(f'Erro médio de rotação: {erro_medio_rotacao} graus')

# Plotar os resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(trajetoria)):
    ax.scatter(trajetoria[i][0, 3], trajetoria[i][1, 3], trajetoria[i][2, 3], c='b', marker='o', label=f'Posição {i}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
