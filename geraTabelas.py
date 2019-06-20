import numpy as np
import separaNota as sN
import utilitarios as utils
import pandas as pd
'''
    Script produzido para gerar tabelas a serem incluidas na monografia
    Gera dados contendo as 10 primeiras posicoes do vetor de frequencias,
    as 10 primeiras posicoes dos coef. de Fourier, 10 primeiras magnitudes
    (da FFT)
    Gera dados contendo os 10 coef. de Fourier com maior magnitude,
    os respectivos modulos e as freq correspondentes.
'''

# Nota escolhida C5
nota = 'C5'
ins_lst = sN.separaInstrumentosMesmaNota(nota)
ins_lst = utils.flat_list(ins_lst)

# Guarda um vetor com os dados do pre Processamento
Saida = []
idxs = []
X = []
for i in range(len(ins_lst)):
    arq = sN.diretorio+ins_lst[i]
    out, coef = utils.preProcessamento(arq)
    idxs.append(out[5])
    X.append(coef)
    Saida.append(out)

# Definicao do vetor de frequencias
N = len(Saida[0][1])
sr = Saida[0][0]

k = np.arange(N)
T = N/float(sr)
freq = k/T

# txt de saida
tbl_out = open('tabelas.txt', 'w')

delta = 10
for i in range(len(ins_lst)):

    X_ = X[i][:delta]

    mX = Saida[i][2][:delta]
    mX = [round(mx, 4) for mx in mX]

    mX_sq = Saida[i][3][:delta]
    mX_sq = [round(m, 4) for m in mX_sq]

    f_ = freq[:delta]
    f_ = [round(f, 4) for f in f_]

    d = {'X': X_, '|X|': mX, 'X^2': mX_sq, 'Frequencia': f_}

    df = pd.DataFrame(d)

    print >> tbl_out, 20*'-', ' ', ins_lst[i], ' ', 20*'-'
    print >> tbl_out, df.to_latex()
    print >> tbl_out, 80*'-', '\n'

print >> tbl_out, 80*'='
print >> tbl_out, 'TABELAS COM OS ATRIBUTOS'

for i in range(len(ins_lst)):

    X_ = X[i][idxs[i]]

    mX = Saida[i][2][idxs[i]]
    mX = [round(mx, 4) for mx in mX]

    mX_sq = Saida[i][3][idxs[i]]
    mX_sq = [round(m, 4) for m in mX_sq]

    f_ = freq[idxs[i]]
    f_ = [round(f, 4) for f in f_]

    d = {'X': X_, '|X|': mX, 'X^2': mX_sq, 'Frequencia': f_}

    df = pd.DataFrame(d)

    print >> tbl_out, 20*'-', ' ', ins_lst[i], ' ', 20*'-'
    print >> tbl_out, df.to_latex()
    print >> tbl_out, 80*'-', '\n'

print >> tbl_out, 80*'='
print >> tbl_out, 'FIM DO PROCEDIMENTO'
tbl_out.close()
