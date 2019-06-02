import numpy as np
import matplotlib.pyplot as plt
import utilitarios as utils
import separaNota as sN


def plotDominioTempo(instrumento, sr, x, nota, pastaDestino='Figuras/'):
    '''
    Plota grafico no dominio do tempo, pelo trecho equivalente a 2 periodos.
    Nota de referencia - Do 32.70 Hz

    @param instrumento : nome do arquivo de entrada
    @param sr   : sample rate, taxa de amostragem do arquivo
    @param x    : arquivo de entrada.
    @param nota : nota a ser plotada. Exemplo: C3, C4, C5;
    @param pastaDestino : pasta para guardar a imagem.
    '''
    N = len(x)
    k = np.arange(N)

    # Vetor dos instantes
    tempo = k/float(sr)

    # Selecao de exatamente uma oscilacao. C0 = 32.703194

    oitava = int(nota[1])
    f_0 = 32.703194           # freq. de referencia
    f_do = f_0*2**(oitava-1)  # Hz
    T_do = 1/f_do             # s
    delta = 4*(int(sr*T_do))    # pega dois periodos do Do.

    # Titulo e nome de arquivo de saida
    Titulo = "Dominio do Tempo: " + instrumento[:-7]
    nomeArq = "DoisPeriodos_" + nota + '_' + instrumento[:-7] + ".png"
    DEST = pastaDestino + nomeArq

    plt.clf()
    plt.plot(tempo[:delta], x[:delta], 'b')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.title(Titulo)
    plt.savefig(DEST)


def plotDominioFreq(instrumento, sr, mX, nota, pastaDestino='Figuras/'):
    '''
    Plota grafico no dominio da frequencia
    Nota de referencia - Do 32.70 Hz

    @param instrumento : nome do arquivo de entrada
    @param sr   : sample rate, taxa de amostragem do arquivo
    @param x    : arquivo de entrada.
    @param nota : nota a ser plotada. Exemplo: C3, C4, C5;
    @param pastaDestino : pasta para guardar a imagem.
    '''

    N = 2*len(mX)
    # Definicao do vetor de frequencias
    k = np.arange(N)
    T = N/float(sr)
    freq = k/T

    # Marcacoes no eixo da frequencia
#    k1 = np.arange(0, 6, 2)     # Do, Re, Mi
#    k2 = np.arange(5, 12, 2)    # Fa, Sol, La, Si
#    k = np.append(k1, k2)
#    k = np.append(k, k + 12)    # 2 oitavas
#    K = 2**(k/12.)

#    oitava = int(nota[1])
#    f_0 = 32.703194           # freq. de referencia
#    f = f_0*2**(oitava-1)     # Hz
#    f_ticks = np.round(f*K, 3)

    delta = len(mX)/4

    Titulo = "Dominio da Frequencia: " + instrumento[:-7]
    nomeArq = "Frequencia_" + nota + '_' + instrumento[:-7] + ".png"
    DEST = pastaDestino + nomeArq

    plt.clf()
    plt.plot(freq[:delta], mX[:delta], 'r', markersize=1)

#   plt.plot(freq[:2**12], saida[j][2][:2**12], 'b.')
#   plt.plot(freq[idxs], mX[idxs], 'ro', markersize=3)
#   plt.xticks(f_ticks)

    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Magnitude")
    plt.title(Titulo)
    plt.grid()
    plt.savefig(DEST)


# Nota desejada
nota = 'C5'

# Lista de instrumentos executando a mesma nota.
inst_list = sN.separaInstrumentosMesmaNota(nota)

inst_list = utils.flat_list(inst_list)

# Realiza pre processamento dos arquivos na lista.
Saida = []

for i in range(len(inst_list)):

    arq = sN.diretorio+inst_list[i]
    out = utils.preProcessamento(arq)

    plotDominioTempo(inst_list[i], out[0], out[1], nota)
    plotDominioFreq(inst_list[i], out[0], out[2], nota)
#   Saida.append(out)
