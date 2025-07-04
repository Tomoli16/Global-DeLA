import torch
from .serialization import Point
from itertools import accumulate
from torch import Tensor
from typing import List, Optional, Tuple




def serialization(xyz, feat, x_res=None, order="z", pts=None, layers_outputs=[], grid_size=0.02):
    if not isinstance(order, list):
        order = [order]

    offset = torch.tensor([0] + list(accumulate(pts)), dtype=torch.long, device=xyz.device)

    # Baue Pointcept Point Objekt
    # Ausreichende Attribute damit Pointcept den Rest berechnet
    point_dict = {'offset': offset, 'coord': xyz, 'grid_size': grid_size}
    point_dict = Point(**point_dict)

    # print("coord:", xyz.shape)           # sollte [P, 3] sein
    # print("offset:", offset.shape)       # sollte [B+1] sein
    # point = Point(coord=xyz, offset=offset, grid_size=0.02)
    # print("batch:", point["batch"].shape)         # sollte [P]
    # print("grid_coord:", point["grid_coord"].shape)  # sollte [P, 3]

    point_dict.serialization(order=order)

    order = point_dict.serialized_order.squeeze(0)       # (P,)
    inverse_order = point_dict.serialized_inverse.squeeze(0)   # (P,)

    # Permutiere alle Flach-Tensoren direkt:
    xyz = xyz[order]
    feat = feat[order]

    if x_res is not None:
        x_res = x_res[order]

    for i in range(len(layers_outputs)):
        layers_outputs[i] = layers_outputs[i][order]
    return xyz, feat, x_res, inverse_order

def deserialization(
    xyz_ser: Tensor,
    feat_ser: Tensor,
    x_res_ser: Optional[Tensor],
    inverse_order: Tensor,
    layers_outputs_ser: Optional[List[Tensor]] = None
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[List[Tensor]]]:
    """
    Revertiere eine Flat-Serialization (P, C) mithilfe von inverse_order.
    
    Args:
        xyz_ser:            Tensor [P, C] (serialisiert)
        feat_ser:           Tensor [P, C] (serialisiert)
        x_res_ser:          Optional[Tensor [P, C]] (serialisiert)
        inverse_order:      LongTensor [P] – Index-Mapping zurück zur Originalreihenfolge
        layers_outputs_ser: Optional[List[Tensor [P, C]]] – serialisierte Layer-Outputs

    Returns:
        xyz:           Tensor [P, C] in Originalreihenfolge
        feat:          Tensor [P, C] in Originalreihenfolge
        x_res:         Optional[Tensor [P, C]] in Originalreihenfolge
        layers_outputs: Optional[List[Tensor [P, C]]] in Originalreihenfolge
    """
    # 1) Gather zurück in Originalreihenfolge
    xyz    = xyz_ser[inverse_order]
    feat   = feat_ser[inverse_order]
    x_res  = x_res_ser[inverse_order] if x_res_ser is not None else None

    # 2) Falls vorhanden, jedes Layer-Output ebenfalls zurückpermuten
    if layers_outputs_ser is not None:
        for i, lo in enumerate(layers_outputs_ser):
            layers_outputs_ser[i] = lo[inverse_order]

    return xyz, feat, x_res, layers_outputs_ser