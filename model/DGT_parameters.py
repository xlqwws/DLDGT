
from typing import List
class DGTParameters:
    """
     Configuration parameters for the overall DGT architecture
    :param window_size: Number of flows used in each window
    :param mlp_layer_sizes: Number of nodes in each layer of the external classification MLP
    :param mlp_dropout: Dropout rate applied between layers in the external classification MLP
    """
    def __init__(self, window_size:int, mlp_layer_sizes:List[int], mlp_dropout:float=0.1):
        self.window_size:int = window_size
        self.mlp_layer_sizes = mlp_layer_sizes
        self.mlp_dropout = mlp_dropout

        self._train_ensure_flows_are_ordered_within_windows = True

        self._train_draw_sequential_windows = False
