import torch
import heapq
from utils.cutils import KDTree

def mst_dfs_seriation(xyz: torch.Tensor,
                      k: int = 16,
                      start_idx: int = 0) -> torch.Tensor:
    """
    Erzeugt eine 1D-Seriation der Punktwolke via MST + DFS.

    Args:
        xyz (torch.Tensor): Punktwolke der Form (N, 3).
        k (int): Anzahl der k-NN-Kanten im Graphen (default: 16).
        start_idx (int): Startpunkt für MST & DFS (default: 0).

    Returns:
        torch.Tensor: Länge-N LongTensor, der die Reihenfolge der Punkte angibt.
    """
    N = xyz.shape[0]

    # 1) KNN-Graph (einmalig)
    kdt = KDTree(xyz)
    dists, neighs = kdt.knn(xyz, k, False)    # dists: (N,k), neighs: (N,k)
    neighs = neighs.long()

    # 2) Adjazenzliste mit Kantengewichten aufbauen
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j_idx, j in enumerate(neighs[i]):
            j = int(j)
            w = float(dists[i, j_idx])
            adj[i].append((j, w))
            adj[j].append((i, w))

    # 3) Prim's Algorithmus für MST
    visited = [False] * N
    visited[start_idx] = True
    heap = []
    # initial alle Kanten vom Startpunkt rein
    for (nbr, w) in adj[start_idx]:
        heapq.heappush(heap, (w, start_idx, nbr))

    mst_adj = [[] for _ in range(N)]
    count = 1
    while heap and count < N:
        w, u, v = heapq.heappop(heap)
        if visited[v]:
            continue
        # Kante (u–v) in den MST aufnehmen
        visited[v] = True
        mst_adj[u].append(v)
        mst_adj[v].append(u)
        count += 1
        # alle ausgehenden Kanten von v hinzufügen
        for (nv, nw) in adj[v]:
            if not visited[nv]:
                heapq.heappush(heap, (nw, v, nv))

    # 4) DFS auf dem MST für die Seriation
    order = []
    visited_dfs = [False] * N
    stack = [start_idx]
    while stack:
        u = stack.pop()
        if visited_dfs[u]:
            continue
        visited_dfs[u] = True
        order.append(u)
        # Nachbarn in umgekehrter Reihenfolge pushen, damit die ursprüngliche
        # Reihenfolge erhalten bleibt
        for v in reversed(mst_adj[u]):
            if not visited_dfs[v]:
                stack.append(v)

    return torch.tensor(order, dtype=torch.long)
