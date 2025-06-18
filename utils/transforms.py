import torch
from .serialization import Point


def serialization(pos, feat, x_res=None, order="z", layers_outputs=[], grid_size=0.02):
    bs, n_p, _ = pos.size()  # (batch_size, num_points, 3)
    if not isinstance(order, list):
        order = [order]

    # Voxilize Points and normalize coordinates
    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)

    point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)

    order = point_dict.serialized_order
    inverse_order = point_dict.serialized_inverse

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    return pos, feat, x_res

def deserialization(pos, feat, x_res=None, layers_outputs=None, inverse_order=None):
    """
    Revertiert die Permutation, die in `serialization` durch order erzeugt wurde.
    Args:
      pos, feat, x_res: Tensor [B, N, C] im serialisierten (permute-)Zustand
      layers_outputs: Liste von Tensoren gleicher Form (optional)
      inverse_order: LongTensor [B*N] mit Rückpermute-Indizes
    Returns:
      pos, feat, x_res, layers_outputs jeweils in Original-Reihenfolge
    """
    bs, n_p, _ = pos.size()
    # flach machen, permuten, dann neu formen
    def unpermute(tensor):
        flat = tensor.flatten(0, 1)        # [B*N, C]
        unflat = flat[inverse_order]       # zurück in Original-Index
        return unflat.view(bs, n_p, -1).contiguous()

    pos     = unpermute(pos)
    feat    = unpermute(feat)
    x_res   = unpermute(x_res)   if x_res   is not None else None

    if layers_outputs is not None:
        for i, lo in enumerate(layers_outputs):
            layers_outputs[i] = unpermute(lo)

    return pos, feat, x_res, layers_outputs
