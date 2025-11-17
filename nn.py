import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class NeuralNet:

    def __init__(self,
                 layers: List[int],
                 epochs: int = 100,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 activation: str = 'sigmoid',
                 validation_split: float = 0.2):
