# import numpy as np
# from scipy.io import wavfile as wav
# import matplotlib.pyplot as plt
import os
import re
# import itertools
TAMANHO_IDEAL = 2**16

# lista com os nomes dos arquivos
diretorio = 'AmostrasTratadas/'
arquivos = os.listdir(diretorio)
arquivos.sort()

# separa os arquivos e grupos por notas

Notas = ["A", "B", "C", "D", "E", "F", "G"]
# lista que guarda arquivos de mesma nota
ArquivosNotas = []
for i in range(len(Notas)):
    res = "." + Notas[i] + "[0-9]."
    arq = [x for x in arquivos if re.search(res, x)]
    ArquivosNotas.append(arq)
