import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os
import MFCC

TAMANHO_IDEAL = 2**16

diretorio = 'AmostrasTratadas/'
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

print arquivos
print "Tudo ok"

audio1 = arquivos[0]
sr, audio1 = wav.read(diretorio+audio1)
audio1 = pre_processa(audio1)

dur_jan = 20e-3    # duracao da janela em ms
passo_jan = 10e-3  # passo da janela em ms

tam_jan = dur_jan*sr
tam_passo = passo_jan*sr
tempo1 = np.linspace(0,20,tam_jan)
tempo2 = np.linspace(10,30, tam_jan)
frames1 = MFCC.divide_em_frames(audio1,sr,dur_jan,passo_jan)

print frames1
print "Tudo continua OK"

#======================= PLOT ===================
#plt.figure(1)
#plt.subplot(121)
#plt.plot(tempo1, frames1[0])
#plt.grid()
#
#plt.subplot(122)
#plt.plot(tempo2, frames1[1])
#plt.grid()
#
#plt.show()

dft_frames0 = MFCC.DFT(frames1[0])
dft_frames0_numpy = np.fft.fft(frames1[0])

print "Minha Implementacao"
print dft_frames0
print "Numpy"
print dft_frames0_numpy

for i in range(len(dft_frames0)/2):
    print dft_frames0[i],dft_frames0_numpy[i]

# Estima o erro do valor obtido com a dct implementada diretamente e a da biblioteca numpy

erro_medio_absoluto = np.abs((dft_frames0 - dft_frames0_numpy)/2)

# Ordem de grandeza do erro, arredondado "para cima"
ordem_grandeza_ema = np.ceil(np.log10(erro_medio_absoluto))

ordem_grandeza_ema [ordem_grandeza_ema == -np.inf] = np.nan
EMA = int( np.nanmean( ordem_grandeza_ema ) )





