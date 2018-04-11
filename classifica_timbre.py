import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
from sklearn.cluster import KMeans

TAMANHO_IDEAL = 2**16

# lista com os nomes dos arquivos
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
    fft_norm = normaliza(fft_amostra, len(fft_amostra))
    conjuntoFFT.append(fft_norm)

# CONFERIR O TAMANHO DE CADA AMOSTRA!!!! #
conjuntoFFT = np.array(conjuntoFFT)

max_idx = np.zeros(conjuntoFFT.shape[0])
max_val = np.zeros(conjuntoFFT.shape[0])
for row in range(conjuntoFFT.shape[0]):
    max_idx[row] = np.argmax(conjuntoFFT[row])
    max_val[row] = np.max(conjuntoFFT[row])

print "============================="
print "FIM DO PROCEDIMENTO"

# Gera Estimadores - instancias do KMeans
estimators = [ ('8 clusters', KMeans(n_clusters=8)), 
			   ('11 clusters', KMeans(n_clusters=11)),
			   ('13 clusters', KMeans(n_clusters=13))]

#clt = KMeans(n_clusters=13)
#clt.fit(conjuntoFFT)

# Roda 3 estimadores 
titles = ['8 clusters', '11 clusters', '13 clusters']
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
	fig.show()
	

