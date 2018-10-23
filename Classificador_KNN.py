import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import itertools


def flat_list(lista):
    '''
    Dada uma lista de listas, retorna um np.array com todos
    os valores "corridos"
    Exemplo: Se a entrada for [ [1,2,3], [4, 5, 6] ]
    a saida sera [1, 2, 3, 4, 5, 6]

    @param lista : lista de listas
    @return np.array : np.array com todos os elementos da lista
    '''

    return np.array(list(itertools.chain.from_iterable(lista)))


def geraConjTreinoTesteClasse(feat, labels, tam_classe, n_treino):

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

    learn_data = flat_list(learn_data)
    learn_labels = flat_list(learn_labels)
    test_data = flat_list(test_data)
    test_labels = flat_list(test_labels)

    indexes = flat_list(indexes)

    return learn_data, learn_labels, test_data, test_labels, indexes


def geraConjTreinoTeste(atributos, rotulos, tam_teste):
    '''
    Dado um vetor de atributos, um vetor de rotulos, divide esse conjunto em
    dois, um de treino e teste, sendo que o de teste tem tamanho tam_teste e
    o de treino tem tamanho len(atributos) - tam_teste

    Entrada
    @Params
    atributos : np.array - Vetor de atributos
    rotulos   : np.array - Vetor de rotulos
    tam_teste : int      - Numero de elementos do conjunto de teste

    Retorno:
    @return
    learn_data:   np.array - Conjunto de treinamento
    learn_labels: np.array - Conjunto de rotulos do treinamento
    test_data:    np.array - Conjunto de teste
    test_labels:  np.array - Conjunto de rotulos de teste
    '''
    np.random.seed(42)
    indices = np.random.permutation(len(rotulos))
    learn_mask = indices[:-tam_teste]
    test_mask = indices[-tam_teste:]

    learn_data = atributos[learn_mask]
    learn_labels = rotulos[learn_mask]

    test_data = atributos[test_mask]
    test_labels = rotulos[test_mask]

    return learn_data, learn_labels, test_data, test_labels


def classificadorKNN(learn_data, learn_labels, test_data, k=1):

    '''
    Instancia um classificador KNN, faz o treinamento e testa com o conjuto
    passado. E possivel definir o numero de vizinhos no argumento k.

    Entrada
    @Params
    learn_data   : numpy.ndarray - Vetor de atributos para treinamento
    learn_labels : numpy.ndarray - Vetor de rotulos para treinamento
    test_data    : numpy.ndarray - Vetor de atributos para teste
    k            : int           - Numero de vizinhos

    Saida
    @return
    predict_labels : numpy.ndarray - Vetor de rotulos previstos pelo modelo

    '''
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(learn_data, learn_labels)
    predict_labels = knn.predict(test_data)

    return predict_labels


def geraMatrizConfusao(testset_lbls, predict_labels):
    """
    Dado os rotulos do conjunto de teste e os rotulos previstos pelo modelo
    gera uma matriz de confusao

    Entrada
    @Params
    testset_lbls    : numpy.ndarray - Vetor de rotulos do conjunto de testes
    predict_labels  : numpy.ndarray - Vetor de rotulos produzidos pelo modelo

    Saida
    @return

    matriz_confusao : pandas.dataframe - Matriz de confusao do modelo
    """

    y_real = pd.Series(testset_lbls, name="Real")
    y_prev = pd.Series(predict_labels, name="Previsto")

    matriz_confusao = pd.crosstab(y_real, y_prev)
    print matriz_confusao

    return matriz_confusao


"""
 Fazendo dois tipos de divisao para conjunto de treino e testes:

 1 - Divide-se o conjunto total (das 416 amostras) em
 dois pedacos, aleatoriamente, sem saber quantos elementos de cada
 conjunto foram escolhidos.

 2 - Garante-se o mesmo numero de elementos de teste em cada
 classe, ou seja, sabe-se que para cada uma das 8 classes, 26 foram
 usados para treino e os demais serao usados para teste.

"""
# learnset_dt, learnset_lbls, testset_dt, testset_lbls = geraConjTreinoTeste(feat, lbl, test_size)
#
# # Define o classificador. Neste caso, 1NN
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(learnset_dt, learnset_lbls)
#
# predict_labels = knn.predict(testset_dt)
#
# y_real = pd.Series(testset_lbls, name="Real")
# y_prev = pd.Series(predict_labels, name="Previsto")
#
# matriz_confusao = pd.crosstab(y_real, y_prev)
# print matriz_confusao
