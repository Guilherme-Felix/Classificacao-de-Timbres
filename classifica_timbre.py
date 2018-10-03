import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
from sklearn.cluster import KMeans
import peakutils
TAMANHO_IDEAL = 2**16

# =========================================================================
# SUGESTOES PARA TRATAR O VIBRATO
#
# Divide o tamanho do vetor pelo sample_rate. Isso indica quantos s o vetor
# inteiro corresponde. 
# 2**16 posicoes correspondem a 2**16/sample-rate segundos.
# A ideia e escolher um intervalo de tempo menor do que isso, digamos 0.5 s,
# manter o tamanho do vetor em 2**16, completando o restante com zeros.
# Fazer a fft para essa amostra, deslocar a "janela" para frente 
# (os 0.5 segundos) e repetir o processo.
# Plotar a fft de cada trecho tridimensionalmente (ou bidimensionalmente) 
# para tentar visualizar a modulação na frequencia
#==========================================================================

# lista com os nomes dos arquivos
diretorio = 'UmaNota/'
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
#    fft_dados = np.real(fft_dados)
    fft_dados = np.abs(fft_dados)
#    fft_norm = [fft_dados[i] / dados.size for i in range( len(fft_dados) )]
    return fft_dados

def normaliza(fft_dados, metodo=1, tam=TAMANHO_IDEAL):
    '''
    ORGANIZAR PARA DEIXAR UM PARAMETRO PARA ESCOLHER QUAL METODO DE 
    NORMALIZACAO
    0 - METODO ATUAL - DIVIDE PELO TAMANHO DO VETOR
    1 - METODO A USAR - DIVIDE PELA MAIOR AMPLITUDE
    Dado uma fft como entrada (e o tamanho da respectiva amostra) 
    retorna um array fft normalizado, ou seja, cada elemento do array
    e dividido pelo tamanho da amostra. Tamanho padrao da amostra e 2^16
    '''
    if (metodo==0):
        #fft_norm = [fft_dados[i] / tam for i in range( len(fft_dados) )]
        fft_norm = fft_dados / tam 
    else:
        fft_norm = fft_dados / max(np.abs(fft_dados))
    return fft_norm

def n_posicoes(janela, periodo_pos):
    '''
    Dado o intervalo de tempo desejado (janela) e o periodo de cada 
        subdivisao do vetor (tempo em segundos correspondente a cada 
        posicao do vetor da amostra de audio) retorna o numero de posicoes
        correspondente a janela
    '''
    return int( janela / periodo_pos)

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
#    fft_norm = normaliza(fft_amostra)
    conjuntoFFT.append(fft_amostra)

conjuntoAmostra = np.array(conjuntoAmostra)
conjuntoFFT = np.array(conjuntoFFT)

print "============================="
print "FIM DO PROCEDIMENTO"

# determina as frequencias
size = conjuntoFFT[0].size
freq = np.linspace(0,sample_rate,size)

# algoritmo que encontra o pico em uma vizinhanca
max_idx = peakutils.indexes(conjuntoFFT[0], thres=0.1, min_dist=1)

# plot dos picos
plt.plot(freq[:3200], conjuntoFFT[0,:3200], 'r-', linewidth=1)
plt.plot(freq[max_idx], conjuntoFFT[0,max_idx], 'k+')
plt.grid()
plt.show()

# Parametros para geracao de graficos no dominio do tempo e da frequencia
duracao_amostra = conjuntoAmostra[0].size / sample_rate
duracao_pos = 1. / sample_rate # periodo de uma posicao no vetor da amostra
tempo = np.arange(0, duracao_amostra, duracao_pos)

# posicao usada para fatiar o vetor para plotar
periodo_nota = 1. / freq[max_idx[0]] # Um ciclo da onda emitida TEM ERRO
janela_tempo = periodo_nota # tamanho da janela em segundos
pos = n_posicoes(janela_tempo, duracao_pos) 
pos = 3.79*pos # Gambiarra para pegar um periodo corretamente

#plt.plot(tempo[:pos], conjuntoAmostra[0,:pos])
# plot with various axes scales
plt.figure(1)
 
# Dominio do tempo Instrumento 0 
plt.subplot(221)
plt.plot(tempo[:pos], conjuntoAmostra[0,:pos], 'r-')
plt.title('Dominio do tempo: ' + arquivos[0][:-3], size='small')
plt.grid(True)

# Dominio da frequencia Instrumento 0 
plt.subplot(222)
plt.plot(freq[:3200], conjuntoFFT[0,:3200], 'b-')
plt.title('Dominio da Frequencia: ' + arquivos[0][:-3], size='small')
plt.grid(True)


# Dominio do tempo Instrumento 5 
plt.subplot(223)
plt.plot(tempo[:pos], conjuntoAmostra[5,:pos], 'r-')
plt.title('Dominio do tempo: ' + arquivos[5][:-3], size='small')
plt.grid(True)

# Dominio da frequencia Instrumento 5 
plt.subplot(224)
plt.plot(freq[:3200], conjuntoFFT[5,:3200], 'b-')
plt.title('Dominio da Frequencia: ' + arquivos[5][:-3], size='small')
plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
#plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, 
                    hspace=0.25, wspace=0.35)
plt.savefig("Tempo_frequencia_dois_instr_10_18.png")
plt.show()

# max_val = np.zeros(conjuntoFFT.shape[0])
# for row in range(conjuntoFFT.shape[0]):
#    max_idx[row] = np.argmax(conjuntoFFT[row])
#    max_val[row] = np.max(conjuntoFFT[row])


# Gera Estimadores - instancias do KMeans
estimators = [ ('9 clusters', KMeans(n_clusters=9)), 
               ('11 clusters', KMeans(n_clusters=11)),
               ('13 clusters', KMeans(n_clusters=13))]

#clt = KMeans(n_clusters=11)
#clt.fit(conjuntoFFT)

# Roda 3 estimadores 
titles = ['9 clusters', '11 clusters', '13 clusters']
fignum = 1
for name, est in estimators:
        print "Rodando KMeans para " + name
	fig = plt.figure(fignum, figsize=(8,6))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	
	est.fit(conjuntoFFT)
	labels = est.labels_
	
	ax.scatter(max_idx, max_val, c=labels.astype(np.float), edgecolor='k')
	
	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Pos_Max_Coef.')
	ax.set_ylabel('Max_Val_Coef.')
	ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1
    fig.savefig(name+".png")
	fig.show()
	

