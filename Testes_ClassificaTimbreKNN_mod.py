import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os
import itertools
import Classificador_KNN_original as cl
# import Classificador_KNN_B as cl
import datetime as dt
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


def classifica(feat, labels, N_class, class_size, train_size, classifier='1NN'):
    '''
    Funcao que chama o classificador (por padrao 1NN), gera a matriz de
    confusao e mede a acuracia do classificador (traco da matriz dividido
    pelo numero de arquivos de teste multiplicado pelo numero de classes

    Input:
        feat:        vetor de atributos:         np.array
        labels:      vetor de rotulos:           np.array
        N_class:     quantidade de classes:      int
        class_size:  tamanho de cada classe:     int
        train_size:  tamanho do conj. de treino: int
        classifier:  classificador a ser usado : string (1NN padrao)

    Output:
        Acc: acuracia da matriz de confusao: float
        M_C: matriz de confusao:             np.array

    '''

    X_train, Y_train, X_test, Y_test, _ = cl.geraConjTreinoTesteClasse(feat,
                                                                       labels,
                                                                       class_size,
                                                                       train_size)
    grp_prev = cl.classificadorKNN(X_train, Y_train, X_test)
    m_conf_grp = cl.geraMatrizConfusao(Y_test, grp_prev)

    Acc = float(np.trace(m_conf_grp))/(N_class * train_size)

    return Acc, m_conf_grp


def plotResultados(titulo, matConf, NOME='Resultados', DEST='./Resultados/'):
    '''
    Funcao para gerar os diagramas (matriz de confusao).
    Entrada:
        titulo  : string   titulo do diagramas
        matConf : np.array Matriz de Confusao
        DEST    : string   Pasta de destino
    '''
    matConf = matConf.astype(int)

    now = dt.datetime.now()
    agr = now.strftime("_%d_%m_%y_%H_%M")
    ext = ".png"

    plt.figure()
    title = titulo
    cl.plot_confusion_matrix(matConf, classes, title=title)
    plt.savefig(DEST + NOME + agr + ext)


def gravaArquivoSaida(titulo, atrib, acc, n_atrib=0,
                      DEST='./Resultados/', NOME='Res'):
    '''
    Guarda os resultados em um arquivo de saida .txt

    Entrada:
        DEST : Pasta de destino : string
        NOME : Nome do arquivo de saida: string
    '''
    now = dt.datetime.now()
    agr = now.strftime("_%d_%m_%y_%H_%M")
    ext = ".txt"

    f = open(DEST + NOME + agr + ext, 'a+')

    print >> f, titulo
    print >> f, "Acuracia: ", ("%.4f" % acc)
    print >> f, "Conjunto de atributos: ", atrib
    if (n_atrib != 0):
        print >> f, "Numero de atributos: ", n_atrib
    print >> f, 80*'-', '\n'

    f.close()


def preProcessamento():
    '''
    Funcao que encapsula todo o pre processamento e producao de vetores de
    atributos. Retorna uma lista de atributos e uma lista com os rotulos

    @return : np.array features:
                        feat[0] = [freq, mX, mX^2]
                        feat[1] = [freq, mX]
                        feat[2] = [freq, mX^2]
                        feat[3] = [mX, mX^2]
                        feat[4] = [freq, mX, ph]
                        feat[5] = [freq, mX, ph, mX^2],
                        Vetor de atributos

    '''

    # lista com os nomes dos arquivos
    diretorio = 'AmostrasTratadas/'
    arquivos = os.listdir(diretorio)
    arquivos.sort()

    # Vetor de atributos
    atributos1 = []  # [freq, mX, mX^2]
    atributos2 = []  # [freq, mX]
    atributos3 = []  # [freq, mX^2]
    atributos4 = []  # [mX, mX^2]
    atributos5 = []  # [freq, mX, ph]
    atributos6 = []  # [freq, mX, ph, mX^2]

    VetorAtributos = ['[freq, mX,mX^2]', '[freq, mX]',
                      '[freq, mX^2]', '[mX, mX^2]', '[freq, mX, ph]',
                      '[freq, mX, ph, mX^2]']

    TAM = len(arquivos)

    for j in range(TAM):
        # Abre arquivo
        print "Arquivo: ", j, '\n'
        sr, x = wav.read(diretorio+arquivos[j])  # sr = sample rate = 44100
        x = pre_processa(x)
        x = normaliza(x)
        X = gera_fft(x)

        # Guarda a magnitude de X
        mX = abs(X)
        # Guarda a fase (em rad)
        ph = np.angle(X)

        # Guarda a magnitude ao quadrado
        mX_sq = mX**2

        # Guarda a magnitude de X na escala log
        mX_log = 20*np.log10(abs(X))

        # Pega apenas a metade do vetor

        N = len(X)
        mX = mX[:N/2]

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

        k = np.arange(N)
        T = N/float(sr)
        freq = k/T

        vetor1 = [freq[idxs], mX[idxs], mX_sq[idxs]]
        vetor2 = [freq[idxs], mX[idxs]]
        vetor3 = [freq[idxs], mX_sq[idxs]]
        vetor4 = [mX[idxs], mX_sq[idxs]]
        vetor5 = [freq[idxs], mX[idxs], ph[idxs]]
        vetor6 = [freq[idxs], mX[idxs], ph[idxs], mX_sq[idxs]]

        vetor1 = flat_list(vetor1)
        vetor2 = flat_list(vetor2)
        vetor3 = flat_list(vetor3)
        vetor4 = flat_list(vetor4)
        vetor5 = flat_list(vetor5)
        vetor6 = flat_list(vetor6)

        atributos1.append(vetor1)
        atributos2.append(vetor2)
        atributos3.append(vetor3)
        atributos4.append(vetor4)
        atributos5.append(vetor5)
        atributos6.append(vetor6)

    atributos1 = np.array(atributos1)
    atributos2 = np.array(atributos2)
    atributos3 = np.array(atributos3)
    atributos4 = np.array(atributos4)
    atributos5 = np.array(atributos5)
    atributos6 = np.array(atributos6)

    feat = [atributos1, atributos2, atributos3,
            atributos4, atributos5, atributos6]

    return feat, VetorAtributos


def Fwd(feat, rotulos, num_classes, tam_classe, tam_teste):

    S = len(feat)
    acc_fwd = np.zeros(S)     # guarda as melhores acuracias para o forward
    N_feat_fwd = np.zeros(S)  # guarda o numero de atributos para o forward
    # guarda as matrizes de confusao para os melhores features do forward
    m_confs_fwd = np.zeros([S, num_classes, num_classes])
    for i in range(S):
        max_feat = feat[i].shape[1]

        for j in range(max_feat):

            feature = feat[i][:, :j+1]
            ac, mc = classifica(feature, rotulos, num_classes,
                                tam_classe, tam_teste)
            if (ac > acc_fwd[i]):
                acc_fwd[i] = ac
                N_feat_fwd[i] = j
                m_confs_fwd[i] = mc
#            else:
#                break

    return acc_fwd, N_feat_fwd, m_confs_fwd


def Bwd(feat, rotulos, num_classes, tam_classe, tam_teste):

    S = len(feat)
    acc_bwd = np.zeros(S)     # guarda as melhores acuracias para o forward
    N_feat_bwd = np.zeros(S)  # guarda o numero de atributos para o forward
    # guarda as matrizes de confusao para os melhores features do forward
    m_confs_bwd = np.zeros([S, num_classes, num_classes])
    for i in range(S):
        max_feat = feat[i].shape[1]

        for j in range(max_feat):

            feature = feat[i][:, j:]
            ac, mc = classifica(feature, rotulos, num_classes,
                                tam_classe, tam_teste)
            if (ac > acc_bwd[i]):
                acc_bwd[i] = ac
                N_feat_bwd[i] = -j+30
                m_confs_bwd[i] = mc
#            else:
#                break

    return acc_bwd, N_feat_bwd, m_confs_bwd
# =========================================================


# lista com os nomes dos arquivos
diretorio = 'AmostrasTratadas/'
arquivos = os.listdir(diretorio)
arquivos.sort()
TAM = len(arquivos)

num_classes = 8
num_amostras = TAM/num_classes  # numero de amostras de cada classe
rotulos = rotula(len(arquivos), num_amostras)
classes = ['Cello', 'Clarineta', 'Flauta', 'Oboe',
           'Trombone', 'Trompete', 'Viola', 'Violino']

tam_classe = 52
tam_teste = tam_classe/2
