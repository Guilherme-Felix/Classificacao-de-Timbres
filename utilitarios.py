import numpy as np
from scipy.io import wavfile as wav
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


def preProcessamento(arquivo):
    '''
    Funcao que encapsula todo o pre-processamento para a classificacao.
    - Abre o arquivo.
    - Extrai a porcao central
    - Normaliza (divide todas as posicoes pelo valor maximo)
    - Gera a fft com o mesmo tamanho da amostra
    - Guarda a magnitude (mX)
    - Guarda a magnitude na escala log (20*log10(mX))
    - Guarda a magnitude quadrada (mX^2)
    - Guarda a fase (phX)
    - Guarda a metade dos vetores
    - Determina os indices dos 10 maiores componentes
    - Define vetor de frequencias
    - Define o vetor de tempo
    - Seleciona dois periodos para Plot (referencia C4 = 261.76 Hz)

    @param : arquivo - Caminho para arquivo.wav a ser processado
    @return : saida - lista contendo a taxa de amostragem do sinal,
    o sinal x, a magnitude da fft, a fase da fft, magnitude quadrada
    e os indices dos 10 maiores valores da magnitude.
    '''

    sr, x = wav.read(arquivo)  # sr = sample rate = 44100
    x = pre_processa(x)        # Fazer 'fade in / out'
    x = normaliza(x)
    X = gera_fft(x)

    # Guarda a magnitude de X
    mX = abs(X)

    # Guarda a magnitude de X na escala log
    mX_log = 20*np.log10(abs(X))

    # Guarda a magnitude ao quadrado
    mX_sq = mX**2

    # Guarda a fase
    phX = np.angle(X)

    # Pega apenas a metade do vetor
    N = len(X)
#   x = x[:N/2]
    mX = mX[:N/2]
    mX_log = mX_log[:N/2]
    mX_sq = mX_sq[:N/2]
    phX = phX[:N/2]

    mX_cpy = mX.copy()
    mX_cpy = sorted(mX_cpy, reverse=True)
    mask = mX_cpy[:10]

    # determinar os indices dos 10 maiores componentes. MAGNITUDE
    idx = []
    for i in range(len(mask)):
        index = np.where(mX == mask[i])
        idx.append(index)

    idxs = [i[0][0] for i in idx]

    # Vetores com as saidas
    saida = [sr, x, mX, mX_sq, phX, idxs], X

    return saida
