import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from app.lib.utils.jsonl import jsonl_to_df


class Ontology(object):
    def __init__(self, matrix):
        # Transform to type array
        if type(matrix) == np.ndarray:
            self.matrix = matrix
        else:
            self.matrix = matrix.toarray()

        
        
        