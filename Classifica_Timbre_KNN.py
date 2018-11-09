import numpy as np
from scipy.io import wavfile as wav
# from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import itertools
import Classificador_KNN as cl
TAMANHO_IDEAL = 2**16


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


def pre_processa(dados, left_channel=0, debug=False):
    '''
    Dado um array  que representa um audio, retorna o array
    cujo tamanho e' potencia de 2.
    E' extraido a parte central do array.
    '''

    dados = dados[:, left_channel]
    tam_amostra = len(dados)
    fora = (tam_amostra - TAMANHO_IDEAL) / 2
    dados = dados[fora:-fora]
    if (len(dados) % 2 == 1):
        dados = np.delete(dados, 0)
    if (debug is True):
        print "Tamanho da amostra: ", tam_amostra
        print "Tamanho apos recorte: ", len(dados)
    dados = np.array(dados, dtype=np.float32)

    return dados


def gera_fft(dados):
    '''
    Dado um array MONO que representa um audio, retorna os coeficientes
    da fft.
    '''

    fft_dados = np.fft.fft(dados)
    return fft_dados


def normaliza(dados):
    '''
    Dado um audio como entrada (e o tamanho da respectiva amostra)
    retorna um array fft normalizado, ou seja, cada elemento do array
    e dividido pelo modulo da maior amplitude(no dominio do tempo).
    Tamanho padrao da amostra e 2^16
    '''

    dados_norm = dados / max(np.abs(dados))

    return dados_norm


def rotula(tam, Nel):
    '''
    Funcao usada para definir o rotulo dos arquivos de entrada.
    tam = int, tamanho do vetor de dados a serem classificados
    Nel = int, numero de elementos pertencentes a cada classe

    '''

    step = Nel
    label = 0
    lbl = np.zeros(tam, dtype=int)

    for i in range(0, tam, Nel):
        lbl[i:i+step] = label
        label = label+1

    return lbl

# ==============================================
#               INICIO DO PROCEDIMENTO
# ==============================================


PLOT = False
# lista com os nomes dos arquivos
diretorio = 'AmostrasTratadas/'
arquivos = os.listdir(diretorio)
arquivos.sort()

atributos = []

TAM = len(arquivos)

for j in range(TAM):
    # Abre arquivo
    print "Arquivo: ", j, '\n'
    sr, x = wav.read(diretorio+arquivos[j])  # sr = sample rate = 44100
    x = pre_processa(x)
    x = normaliza(x)
    X = gera_fft(x)

    # Guarda a magnitude de X
    mX = abs(X)

    # Guarda a magnitude de X na escala log
    mX_log = 20*np.log10(abs(X))

    # Pega apenas a metade do vetor

    N = len(X)
    mX = mX[:N/2]

    mX_log = mX_log[:N/2]

    mX_cpy = mX.copy()
    mX_cpy = sorted(mX_cpy, reverse=True)
    mask = mX_cpy[:10]

    # determinar os indices dos 10 maiores componentes
    idx = []
    for i in range(len(mask)):
        index = np.where(mX == mask[i])
        idx.append(index)

    idxs = [i[0][0] for i in idx]

    k = np.arange(N)
    T = N/float(sr)
    freq = k/T

    if(PLOT is True):
        plt.plot(freq[:N/2], mX)
        plt.plot(freq[idxs], mX[idxs], 'ro', markersize=3)
        plt.xticks(freq[idxs])
        plt.xlabel("Frequencia")
        plt.ylabel("Magnitude")
        plt.title("Magnitude em log10_" + arquivos[j])
        plt.show()

    vetor = [freq[idxs], mX[idxs], X[idxs]]
    vetor = flat_list(vetor)
    atributos.append(vetor)

atributos = np.array(atributos)

# os.system('spd-say "Features extracted successfully!" -r -80')

num_classes = 8
num_amostras = TAM/num_classes  # numero de amostras de cada classe
rotulos = rotula(len(arquivos), num_amostras)
classes = ['Cello', 'Clarineta', 'Flauta', 'Oboe',
           'Trombone', 'Trompete', 'Viola', 'Violino']

# ========================================================
#                     CLASSIFICADOR
# ========================================================


# Fazendo dois tipos de divisao para conjunto de treino e testes:
#
# 1 - Divide-se o conjunto total (das 416 amostras) em
# dois pedacos, aleatoriamente, sem saber quantos elementos de cada
# conjunto foram escolhidos.
#
# 2 - Garante-se o mesmo numero de elementos de teste em cada
# classe, ou seja, sabe-se que para cada uma das 8 classes, 26 foram
# usados para treino e os demais serao usados para teste.

# Metodo 1 - Dividindo todo o dataset em dois pedacos iguais para treino e
# teste)
#tam_teste = len(rotulos)/2

#cj_treino, rot_treino, cj_teste, rot_teste = cl.geraConjTreinoTeste(atributos,
#                                                                    rotulos,
#                                                                   tam_teste)
#previsto = cl.classificadorKNN(cj_treino, rot_treino, cj_teste)
#m_conf = cl.geraMatrizConfusao(rot_teste, previsto)

# Metodo 2 - Dividindo cada grupo ao meio. Metade para teste e a outra metade
# para treino

tam_classe = 52
tam_teste = tam_classe/2

X_train, Y_train, X_test, Y_test, _ = cl.geraConjTreinoTesteClasse(atributos,
                                                                   rotulos,
                                                                   tam_classe,
                                                                   tam_teste)
grp_prev = cl.classificadorKNN(X_train, Y_train, X_test)
m_conf_grp = cl.geraMatrizConfusao(Y_test, grp_prev)

# Gera os graficos
#DEST = 'Figuras/'
#NOME1 = 'CM_ConjuntoTotalDiv2.png'
#plt.figure()
#cl.plot_confusion_matrix(m_conf, classes, title='Conjunto Total dividido em 2')
#plt.savefig(DEST+NOME1)
#plt.show()

#NOME2 = 'CM_DivisaoPorConjunto.png'
#plt.figure()
#cl.plot_confusion_matrix(m_conf_grp, classes, title='Divisao feita por grupo')
#plt.savefig(DEST+NOME2)
#plt.show()
