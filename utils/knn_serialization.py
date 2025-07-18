from utils.cutils import KDTree
import torch

def greedy_nn_tour_fast(xyz: torch.Tensor, K_max: int = 32, start_idx: int = 0):
    N = xyz.shape[0]
    visited = torch.zeros(N, dtype=torch.bool)
    tour    = torch.empty(N, dtype=torch.long)

    # 1) KD-Tree bauen und einmalig K_max Nachbarn pro Punkt holen
    kdt = KDTree(xyz)
    _, neighs = kdt.knn(xyz, K_max, False)   # neighs: (N, K_max)
    neighs = neighs.long()

    # 2) Greedy-Tour
    current = start_idx
    visited[current] = True
    tour[0] = current

    for i in range(1, N):
        # Durch die vorgefertigten Nachbarn scannen, bis wir einen unbesuchten finden
        neigh_list = neighs[current]
        for idx in neigh_list:
            if not visited[idx]:
                next_idx = int(idx)
                break
        else:
            # Fallback: zufälliger unbesuchter Punkt
            candidates = torch.nonzero(~visited, as_tuple=False).view(-1)
            next_idx = int(candidates[torch.randint(len(candidates), (1,))])

        tour[i] = next_idx
        visited[next_idx] = True
        current = next_idx

    return tour
