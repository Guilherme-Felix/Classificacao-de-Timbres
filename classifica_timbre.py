import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os 
from sklearn.cluster import KMeans

TAMANHO_IDEAL = 2**16

# lista com os nomes dos arquivos
diretorio = 'AMOSTRAS/'
arquivos = os.listdir(diretorio)
arquivos.sort()

def pre_processa(dados, left_channel=0, debug=False):
    '''
    Dado um array  que representa um a'udio, retorna o array 
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
    return dados

def gera_fft(dados):
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
    e dividido pelo tamanho da amostra. Tamanho padrao da amostra e 2^16
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
    print i

    sample_rate, amostra = wav.read(diretorio+arquivos[i])

    print diretorio+arquivos[i] , '\n' 

    amostra = pre_processa(amostra, debug=True)
    
    print amostra ,'\n'
    
    fft_amostra = gera_fft(amostra)
    conjuntoFFT.append(fft_amostra)

# CONFERIR O TAMANHO DE CADA AMOSTRA!!!! #
conjuntoFFT = np.array(conjuntoFFT)
print "============================="
print "FIM DO PROCEDIMENTO"
print conjuntoFFT
clt = KMeans(n_clusters=8)
clt.fit(conjuntoFFT)


