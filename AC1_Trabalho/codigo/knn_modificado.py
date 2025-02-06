from collections import Counter
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, mahalanobis
from sklearn.preprocessing import StandardScaler
from .base import BaseEstimator





#-----





class KNNBase(BaseEstimator):

    def __init__(self, k=5, distance_func=euclidean, outlier_threshold=2.0):        #ALTERACAO (novo parametro "outlier_threshold")
        
        #INICIO ALTERACOES (validacao dos parametros de input)
        if not isinstance(k, int) or k < 0:
            raise ValueError("k tem de ser integer positivo!")
        if not callable(distance_func):
            raise ValueError("distance_func tem de ser uma função de scipy.spatial.distance!")
        if not isinstance(outlier_threshold, int) or outlier_threshold < 0:
            raise ValueError("outlier_threshold tem de ser float positivo")
        #FIM ALTERACOES!
            

        self.k = None if k == 0 else k
        self.distance_func = distance_func
        self.outlier_threshold = outlier_threshold          #ALTERACAO (novo parametro "outlier_threshold")



    #INICIO ALTERACOES (definicao da funcao fit, com normalizacao dos dados)
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
        self.scaler = StandardScaler().fit(self.X)
        self.X = self.scaler.transform(self.X)

        self.outliers = self._identify_outliers()
    #FIM ALTERACOES



    #INICIO ALTERACOES (definicao da funcao _identify_outliers para identificar outliers)
    def _identify_outliers(self):
        distances = np.array([np.linalg.norm(self.X - x, axis=1) for x in self.X])
        knn_distances = np.sort(distances, axis=1)[:, 1:self.k+1]  # Skip the distance to itself (0)
        avg_distances = knn_distances.mean(axis=1)

        mean_distance = avg_distances.mean()
        std_distance = avg_distances.std()

        outlier_threshold = mean_distance + self.outlier_threshold * std_distance
        outliers = avg_distances > outlier_threshold

        return outliers
    #FIM ALTERACOES




    def _predict(self, X):
        X = self.scaler.transform(X)
        return np.array([self._predict_x(x) for x in X])



    def _predict_x(self, x):
        distances = (self.distance_func(x, example) for example in self.X)

        neighbors = sorted(zip(distances, self.y, self.outliers), key=lambda pair: pair[0])
        neighbors = neighbors[:self.k]

        #INICIO ALTERACOES (exclusao dos outliers, resolucao de empates)
        neighbors_targets = [target for _, target, outlier in neighbors if not outlier]
        distances = [distance for distance, _, outlier in neighbors if not outlier]

        if not neighbors_targets:  #nao ha neighbors
            return None

        counter = Counter(neighbors_targets)
        most_common_labels = counter.most_common()
        
        if not most_common_labels:  #nao ha empates nem esta vazio
            return neighbors_targets[0]

        max_count = most_common_labels[0][1]
        tied_labels = [label for label, count in most_common_labels if count == max_count]

        if len(tied_labels) > 1:
            tied_neighbors = [target for _, target, outlier in neighbors if target in tied_labels and not outlier]
            for target in tied_neighbors:
                if target in tied_labels:
                    return target

        return most_common_labels[0][0]
        #FIM ALTERACOES!


    def aggregate(self, neighbors_targets, distances):

        raise NotImplementedError("")





class KNNClassifier(KNNBase):

    def aggregate(self, neighbors_targets, distances):

        #INICIO ALTERACOES (weighting)
        if not distances:
            raise ValueError("Lista vazia?")

        weights = [1 / distance if distance != 0 else 1e-10 for distance in distances]

        weighted_votes = Counter()
        for target, weight in zip(neighbors_targets, weights):
            weighted_votes[target] += weight
        #FIM ALTERACOES!

        most_common_label = weighted_votes.most_common(1)[0][0]
        return most_common_label





#-----




"""
MUDANÇAS IMPLEMENTADAS!!!


Validação dos parametros de input:
- no metodo __init__ (apenas admite valores corretos de k, distance_func e outlier_threshold);

   
Empates em resultados:
- no método _predict_x (escolhe a label mais proxima);
   

Deteção de outliers:
- adicionado método _identify_outliers (identifica outliers de acordo com o threshold)
- método fit integrado neste ficheiro


Normalização:
- na função fit (usando StandardScaler)


Weighting:
- no método aggregate (atribuindo mais peso a instâncias mais próximas, e menos peso a instâncias mais distantes)



MUDANCAS MAIS SIGNIFICANTES PARA A RESOLUCAO DOS OUTLIERS SAO:
*DETEÇÃO DE OUTLIERS!
*WEIGHTING!
"""