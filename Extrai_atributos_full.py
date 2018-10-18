import numpy as np
from scipy.io import wavfile as wav
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import itertools
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
    Dado um array MONO que representa um audio, retorna a magnitude
    da fft.(modulo de cada X(n)).
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


def permuta(vet, tam_classe):

    '''
    Retorna um vetor de indices permutados por classe
    '''

    np.random.seed(42)

    N = len(vet)
    step = tam_classe
    indexes = []

    for i in range(0, N, step):
        arr = np.arange(i, i+step)
        perm = np.random.permutation(arr)
        indexes.append(perm)
        indexes = np.array(indexes)

    return indexes

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

    # Guarda a magnitude de X na escala log
    mX = abs(X)

    # Pega apenas a metade do vetor

    N = len(X)
    mX = mX[:N/2]
    mX_cpy = mX.copy()
    mX_cpy = sorted(mX_cpy, reverse=True)
    mask = mX_cpy[:10]

    # determinar os indices dos 10 maiores componentes
    idx = []
    for i in range(len(mask)):
        index = np.where(mX == mask[i])
        idx.append(index)

    idxs = [i[0][0] for i in idx]
    # print(idx)
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

os.system('spd-say "Features extracted successfully!" -r -80')

num_classes = 8
num_amostras = TAM/num_classes  # numero de amostras de cada classe
rotulos = rotula(len(arquivos), num_amostras)
