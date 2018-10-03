import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os 
from sklearn.cluster import KMeans
import re
import pandas as pd
from scipy.stats import mode
TAMANHO_IDEAL = 2**16

# lista com os nomes dos arquivos
diretorio = 'AmostrasTratadas/'
arquivos = os.listdir(diretorio)
arquivos.sort()

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
    return dados

def gera_fft(dados):
    '''
    Dado um array MONO que representa um audio, retorna a magnitude da fft.(modulo de cada X(n).
    '''
    fft_dados = np.fft.fft(dados)
    fft_dados = np.abs(fft_dados)
    return fft_dados

def normaliza(dados, metodo=1, tam=TAMANHO_IDEAL):
    '''
    Dado um audio como entrada (e o tamanho da respectiva amostra) 
    retorna um array fft normalizado, ou seja, cada elemento do array
    e dividido pelo modulo da maior amplitude(no dominio do tempo). Tamanho padrao da amostra e 2^16
    '''
    dados_norm = dados / max(np.abs(dados))
    return dados_norm

def rotula_arquivos(arquivos):
    '''
    Dado o vetor de arquivos de entrada, produz um vetor saida com o rotulo de cada arquivo
    0 - Bassoon
    1 - Cello
    2 - Clarineta
    3 - Flauta
    4 - Oboe
    5 - Sax
    6 - Trombone
    7 - Trompete
    8 - Viola
    '''
    nomes = ['Basso*', 'Cello*', 'Clar*', 'Flau*', 'Oboe*', 'Sax*', 'Tromb*', 'Tromp*', 'Viola*']
    y_real = np.zeros(len(arquivos), dtype=int)
    lbl = 0
    for i in range (len(arquivos)):
        nm = arquivos[i]
        if( re.match(nomes[lbl], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl
        elif (re.match(nomes[lbl+1], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 1
        elif (re.match(nomes[lbl+2], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 2
        elif (re.match(nomes[lbl+3], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 3
        elif (re.match(nomes[lbl+4], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 4
        elif (re.match(nomes[lbl+5], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 5
        elif (re.match(nomes[lbl+6], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 6
        elif (re.match(nomes[lbl+7], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 7
        elif (re.match(nomes[lbl+8], nm )):
            print 'OK - ' + nm
            y_real[i] = lbl + 8
            
    return y_real
        
############################################
#                                          #
#          INICIO DO PROCEDIMENTO          #
#                                          #
############################################

conjuntoAmostra = []
conjuntoFFT = [] 

for i in range ( len(arquivos) ):
    print i
    sample_rate, amostra = wav.read(diretorio+arquivos[i])

    print diretorio+arquivos[i] , '\n' 
    amostra = np.array(amostra,dtype=np.float32)
    amostra = pre_processa(amostra)
    
    print amostra ,'\n'
    amostra_norm = normaliza(amostra)
    print amostra_norm ,'\n'

    conjuntoAmostra.append(amostra)
        
    fft_amostra = gera_fft(amostra)
    conjuntoFFT.append(fft_amostra)

conjuntoAmostra = np.array(conjuntoAmostra)
conjuntoFFT = np.array(conjuntoFFT)

print "============================="
print "FIM DO PROCEDIMENTO"

rotulos_reais = rotula_arquivos(arquivos)


num_instrumentos = 9
print " Define nº de clusters = " + str(num_instrumentos)
clt = KMeans(n_clusters=9, random_state=0)
clusters = clt.fit_predict(conjuntoFFT)

labels = np.zeros_like(clusters)
for i in range(num_instrumentos):
    mask = (clusters == i)
    labels[mask] = mode( rotulos_reais[mask] )[0]

y_real = pd.Series(rotulos_reais, name="Real")
y_prev = pd.Series(labels, name="Previsto")

matriz_confusao = pd.crosstab(y_real, y_prev)

