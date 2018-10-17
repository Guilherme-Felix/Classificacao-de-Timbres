import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os 
import itertools
import pandas as pd
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
    dados = dados[:,left_channel]
    tam_amostra = len(dados)
    fora = (tam_amostra - TAMANHO_IDEAL) / 2
    dados = dados[fora:-fora]
    if( len(dados) % 2 == 1):
        dados = np.delete(dados, 0)
    if (debug == True):
        print "Tamanho da amostra: ", tam_amostra
        print "Tamanho apos recorte: ", len(dados)
    dados = np.array(dados, dtype=np.float32)
    return dados

def gera_fft(dados):
    '''
    Dado um array MONO que representa um audio, retorna a magnitude da fft.(modulo de cada X(n)).
    '''
    fft_dados = np.fft.fft(dados)
    return fft_dados

def normaliza(dados):
    '''
    Dado um audio como entrada (e o tamanho da respectiva amostra) 
    retorna um array fft normalizado, ou seja, cada elemento do array
    e dividido pelo modulo da maior amplitude(no dominio do tempo). Tamanho padrao da amostra e 2^16
    '''
    dados_norm = dados / max(np.abs(dados))
    return dados_norm

def rotula(tam, Nel):
    step = Nel
    l = 0
    lbl = np.zeros(tam,dtype=int)
    for i in range(0, tam, Nel):
        lbl[i:i+step] = l
        l = l+1
    return lbl

#==============================================
#               INICIO DO PROCEDIMENTO
#==============================================

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
    sr, x = wav.read(diretorio+arquivos[j]) # sr = sample rate = 44100
    x = pre_processa(x)
    x = normaliza(x)
    X = gera_fft(x)

    # Guarda a magnitude de X na escala log
    mX = abs(X)

    # Pega apenas a metade do vetor

    N = len(X)
    mX = mX[:N/2]
    
    mX_cpy = mX.copy()
    mX_cpy = sorted(mX_cpy,reverse=True)
    mask = mX_cpy[:10]

    # determinar os indices dos 10 maiores componentes
    idx = []
    for i in range(len(mask)):
        index = np.where(mX == mask[i])
        idx.append(index)


    idxs = [ i[0][0] for i in idx]
    #print(idx)
    k = np.arange(N)
    T = N/float(sr)
    freq = k/T
    if(PLOT==True):
        plt.plot(freq[:N/2], mX)
        plt.plot(freq[idxs], mX[idxs], 'ro',markersize=3)
        plt.xticks(freq[idxs])
        plt.xlabel("Frequencia")
        plt.ylabel("Magnitude")
        plt.title("Magnitude em log10_"+ arquivos[j])
        plt.show()

    vetor = [freq[idxs], mX[idxs], X[idxs]]
    vetor = flat_list(vetor)
    atributos.append(vetor)

atributos = np.array(atributos)

os.system('spd-say "Features extracted successfully!" -r -80')

#========================================================
#                       CLASSIFICADOR
#========================================================

num_classes = 8
num_amostras = TAM/num_classes  # numero de amostras de cada classe
rotulos = rotula(len(arquivos),num_amostras)

np.random.seed(42)
# gera permutacao de indices para conjunto de treinamento
indices = np.random.permutation(len(arquivos))

n_training_samples = 78
train_mask = indices[:-n_training_samples]
test_mask = indices[-n_training_samples:]

# Definicao de conjunto de treinamento
learnset_data = atributos[train_mask]
learnset_labels = rotulos[train_mask]

# Definicao do conjunto de testes
testset_data = atributos[test_mask]
testset_labels = rotulos[test_mask]

os.system('spd-say "Classifier Set!" -r -50')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(learnset_data, learnset_labels)

predict_labels = knn.predict(testset_data)

y_real = pd.Series(testset_labels, name="Real")
y_prev = pd.Series(predict_labels, name="Previsto")

matriz_confusao = pd.crosstab(y_real, y_prev)

