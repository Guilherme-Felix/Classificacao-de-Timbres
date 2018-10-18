import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd


def geraConjTreinoTeste(feat, labels, tam_classe, n_treino):

    '''
    Dado um vetor ordenado, por classe,
    retorna os conjuntos de treino e teste dado o numero de elementos
    para treino.

    Neste caso, o vetor de rotulos esta ordenado por classe:
    da posicao 0  ate a 51 - classe 0
    da posicao 52 ate a 103 - classe 1
    ...
    Preciso escolher aleatoriamente um determinado numero de elementos
    para treino e os demais para teste (digamos 26 para cada finalidade).

    Entrada:
    @Params
    feat:       np.array - Vetor de atributos
    labels:     np.array - Vetor de rotulos
    tam_classe: int - Tamanho das classes (todas do mesmo tamanho)
    n_treino:   int - Numero de elementos para treino
                      (igual para todas as classes)
    Retorno:
    @return
    learn_data:   np.array - Conjunto de treinamento
    learn_labels: np.array - Conjunto de rotulos do treinamento
    test_data:    np.array - Conjunto de teste
    test_labels:  np.array - Conjunto de rotulos de teste
    '''

    np.random.seed(42)

    N = len(feat)
    step = tam_classe
    indexes = []
    learn_data = []
    learn_labels = []
    test_data = []
    test_labels = []

    for i in range(0, N, step):
        arr = np.arange(i, i+step)
        perm = np.random.permutation(arr)

        learn_mask = perm[:n_treino]
        test_mask = perm[n_treino:]

        learn_data.append(feat[learn_mask])
        learn_labels.append(labels[learn_mask])

        test_data.append(feat[test_mask])
        test_labels.append(labels[test_mask])

        indexes.append(perm)

    learn_data = np.array(learn_data)
    learn_labels = np.array(learn_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    indexes = np.array(indexes)

    return learn_data, learn_labels, test_data, test_labels, indexes


# ========================================================
#                     CLASSIFICADOR
# ========================================================

# Numero de elementos usados para treino em cada classe

n_training_samples = 26
print "Tamanho do conjunto de testes: ", n_training_samples

# learnset_data, learnset_labels, testset_data, testset_labels = geraConjTreinoTeste()
# train_mask = indices[:-n_training_samples]
# test_mask = indices[-n_training_samples:]

# Definicao de conjunto de treinamento
# learnset_data = atributos[train_mask]
# learnset_labels = rotulos[train_mask]
#
# # Definicao do conjunto de testes
# testset_data = atributos[test_mask]
# testset_labels = rotulos[test_mask]

# os.system('spd-say "Classifier Set!" -r -50')
#
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(learnset_data, learnset_labels)
#
# predict_labels = knn.predict(testset_data)
#
# y_real = pd.Series(testset_labels, name="Real")
# y_prev = pd.Series(predict_labels, name="Previsto")
#
# matriz_confusao = pd.crosstab(y_real, y_prev)
