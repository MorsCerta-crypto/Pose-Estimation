
import numpy as np
from typing import Union,List


def dist(p1: Union[np.ndarray,List[float]], p2: Union[np.ndarray,List[float]]) -> float:
    """ calculate the distance of movement for each landmark"""
    if type(p1) ==List[float]:
        p1 = np.ndarray(p1)
    if type(p2) == List[float]:
        p2 = np.ndarray(p2)
        
    dist = np.linalg.norm(p1 - p2)

    return dist