import numpy as np
from scipy.io import wavfile as wav
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


# ==============================================
#               INICIO DO PROCEDIMENTO
# ==============================================


PLOT = True
# lista com os nomes dos arquivos
diretorio = 'AmostrasTratadas/'
arquivos = os.listdir(diretorio)
arquivos.sort()

# Instrumento e Nota
instr1 = 'Flauta_C5.wav'
idx1 = arquivos.index(instr1)
instr2 = 'Trombone_C5.wav'
idx2 = arquivos.index(instr2)

# Vetores com as saidas
saida = []
nomes = []
for j in (idx1, idx2):
    # Abre arquivo
    nomes.append(arquivos[j])
    print "Arquivo: ", arquivos[j], '\n'
    sr, x = wav.read(diretorio+arquivos[j])  # sr = sample rate = 44100
    x = pre_processa(x)  # Fazer 'fade in / out'
    x = normaliza(x)
    X = gera_fft(x)

    # Guarda a magnitude de X
    mX = abs(X)

    # Guarda a magnitude ao quadrado
    mX_sq = mX**2

    # Guarda a magnitude de X na escala log
    mX_log = 20*np.log10(abs(X))

    # Pega apenas a metade do vetor

    N = len(X)
    mX = mX[:N/2]
    x = x[:N/2]
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

    saida.append([x, mX, idxs])

# Definicao do vetor de frequencias
k = np.arange(N)
T = N/float(sr)
freq = k/T

# Vetor dos instantes
tempo = k/float(sr)

# Selecao de exatamente uma oscilacao para o C4 = 261.76 Hz
f_do = 261.76    # Hz
T_do = 1/f_do    # s, aproximadamente 0.0382 s
delta = int(sr*T_do)

# PLOT DOMINIO DO TEMPO - TAMANHO IGUAL 2^15 = 0.7 s

pastaDestino = 'Figuras/'

if(PLOT is True):
    for j in (0, 1):
        Titulo = "Dominio do Tempo: " + nomes[j][:-7]
        nomeArq = "Tempo_" + nomes[j][:-7] + ".png"
        DEST = pastaDestino + nomeArq

        plt.clf()
        plt.plot(tempo[:2**15], saida[j][0], 'b')
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude")
        plt.title(Titulo)
        plt.savefig(DEST)
#        plt.show()

# PLOT DOMINIO DO TEMPO - TAMANHO IGUAL a um periodo C5 = 0.0038s

if(PLOT is True):
    for j in (0, 1):
        Titulo = "Dominio do Tempo: " + nomes[j][:-7]
        nomeArq = "UmPeriodoC4_" + nomes[j][:-7] + ".png"
        DEST = pastaDestino + nomeArq

        plt.clf()
        plt.plot(tempo[:delta], saida[j][0][:delta], 'b')
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude")
        plt.title(Titulo)
        plt.savefig(DEST)

# PLOT DOMINIO DA FREQUENCIA - PARA OS 10 MAIORES VALORES mX

if(PLOT is True):
    for j in (0, 1):
        Titulo = "Dominio do Frequencia: " + nomes[j][:-7]
        nomeArq = "Frequencia_" + nomes[j][:-7] + ".png"
        DEST = pastaDestino + nomeArq

        plt.clf()
        plt.plot(freq[:2**12], saida[j][1][:2**12], 'r')

#       plt.plot(freq[idxs], mX[idxs], 'ro', markersize=3)
        plt.xticks(freq[idxs])
        plt.xlabel("Frequencia (Hz)")
        plt.ylabel("Magnitude")
        plt.title(Titulo)
        plt.savefig(DEST)
#        plt.show()

os.system('spd-say "Features extracted successfully!" -r -80')
