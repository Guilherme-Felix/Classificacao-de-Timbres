import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as pt
import os 
from sklearn.cluster import KMeans

TAMANHO_IDEAL = 2**17

# lista com os nomes dos arquivos
diretorio = 'AMOSTRAS/'
arquivos = os.listdir(diretorio)
arquivos.sort()

def pre_processa(dados):
    '''
    Dado um array  que representa um a'udio, retorna o array 
    cujo tamanho e' potencia de 2.
    E' extraido a parte central do array.
    '''
    amostra = amostra[:,0]
    fora = (dados.size - TAMANHO_IDEAL) / 2
    dados = dados[fora:-fora]
    return dados

def gera_fft(dados)
    '''
    Dado um array MONO que representa um audio, retorna sua fft,
    com seus coeficientes em modulo.
    '''
    fft_dados = np.fft.rfft(dados)
    fft_dados = np.abs(fft_dados)
#    fft_norm = [fft_dados[i] / dados.size for i in range( len(fft_dados) )]
    return fft_dados

def normaliza(fft_dados, tam=TAMANHO_IDEAL):
    '''
    Dado uma fft como entrada (e o tamanho da respectiva amostra) 
    retorna um array fft normalizado, ou seja, cada elemento do array
    e dividido pelo tamanho da amostra. Tamanho padrao da amostra e 2^17
    '''
    fft_norm = [fft_dados[i] / tam for i in range( len(fft_dados) )]
    return fft_norm

######################################################################
#
#                   INICIO DO PROCEDIMENTO
#
######################################################################

conjuntoFFT = []

for i in range ( len(arquivos) ):
    sample_rate, amostra = wav.read(diretorio+arquivos[i])
    amostra = pre_processa(amostra)
    fft_amostra = gera_fft_norm(amostra)
    conjuntoFFT.append(fft_amostra)

#clt = KMeans()
#clt.fit(conjuntoFFT.T)

