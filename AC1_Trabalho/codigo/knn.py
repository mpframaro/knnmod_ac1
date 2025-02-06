from collections import Counter

import numpy as np
from scipy.spatial.distance import euclidean                                        #pode ser usado cityblock (manhattan), braycurtis, ...

from .base import BaseEstimator







class KNNBase(BaseEstimator):       

    def __init__(self, k=5, distance_func=euclidean):                               #define os parametros da classe
        self.k = None if k == 0 else k  # l[:None] returns the whole list
        self.distance_func = distance_func


    def aggregate(self, neighbors_targets):                                         #coloca uma excepção (VER!!!!!) usado p override
        raise NotImplementedError()


    def _predict(self, X=None):                                                     #percorre todos os X, chamando _predict_x para cada um, e cria um array c as predictions
        predictions = [self._predict_x(x) for x in X]
        return np.array(predictions)


    def _predict_x(self, x):
        # compute distances between x and all examples in the training set.
        distances = (self.distance_func(x, example) for example in self.X)

        # Sort all examples by their distance to x and keep their target value.
        neighbors = sorted(((dist, target) for (dist, target) in zip(distances, self.y)), key=lambda x: x[0])

        # Get targets of the k-nn and aggregate them (most common one or
        # average).
        neighbors_targets = [target for (_, target) in neighbors[: self.k]]

        return self.aggregate(neighbors_targets)        #usa aggregate do KNNClassifier





class KNNClassifier(KNNBase):

    def aggregate(self, neighbors_targets):
        most_common_label = Counter(neighbors_targets).most_common(1)[0][0]
        return most_common_label