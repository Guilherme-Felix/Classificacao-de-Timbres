import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os 
TAMANHO_IDEAL = 2**16

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

# Abre arquivo
sr, x = wav.read('AmostrasTratadas/Clarineta_F4.wav') # sr = sample rate = 44100
x = pre_processa(x)
x = normaliza(x)
X = gera_fft(x)

# Guarda a magnitude de X na escala log
mX = 20*np.log10(abs(X))

# Pega apenas a metade do vetor

N = len(X)
mX = mX[:N/2]

bins = 300 # Eh possivel que esse parametro tenha que ser ajustado de acordo com o audio...
new_samples = []
for i in range(0, N, bins-1):
	
	vect = mX[i:i+bins]
	if vect.shape[0] > 0:
		max_value = np.amax(vect)
		new_samples.append(max_value)

new_samples = np.array(new_samples)
mX_cpy = new_samples.copy()
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

plt.plot(freq[:N/2], mX)
plt.plot(freq[idxs], mX[idxs], 'ro',markersize=3)
plt.xticks(freq[idxs])
plt.xlabel("Frequencia")
plt.ylabel("Magnitude")
plt.title("Magnitude em log10")
plt.show()

# anotar: [frequencia, "amplitude" na escala log, valor complexo] OU
#         [frequencia, amplitude do sinal (abs), valor complexo]
