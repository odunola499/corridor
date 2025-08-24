from typing import List


def compute_stochastic_depth_drop_probs(
    num_layers: int,
    stochastic_depth_drop_prob: float = 0.0,
    stochastic_depth_mode: str = "linear",
    stochastic_depth_start_layer: int = 1,
) -> List[float]:
    """Computes drop probabilities for stochastic depth regularization technique.
    The first layer is never dropped and the starting layer needs to be greater
    or equal to 1.

    Args:
        num_layers (int): number of layers in the network.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
    Returns:
        List[float]: list of drop probabilities for all layers
    """
    if not (0 <= stochastic_depth_drop_prob < 1.0):
        raise ValueError("stochastic_depth_drop_prob has to be in [0, 1).")
    if not (1 <= stochastic_depth_start_layer <= num_layers):
        raise ValueError("stochastic_depth_start_layer has to be in [1, num layers].")
    layer_drop_probs = [0.0] * stochastic_depth_start_layer

    if (L := num_layers - stochastic_depth_start_layer) > 0:
        if stochastic_depth_mode == "linear":
            layer_drop_probs += [l / L * stochastic_depth_drop_prob for l in range(1, L + 1)]
        elif stochastic_depth_mode == "uniform":
            layer_drop_probs += [stochastic_depth_drop_prob] * L
        else:
            raise ValueError(
                f'stochastic_depth_mode has to be one of ["linear", "uniform"]. Current value: {stochastic_depth_mode}'
            )
    return layer_drop_probs