from scipy.io import wavfile as wav
import taglib as tgl
import os

diretorio = 'AmostrasCompleto/'
arquivos = os.listdir(diretorio)
arquivos.sort()
N = len(arquivos)

for i in range(N):
    song = tgl.File(diretorio+arquivos[i])
    print 'Tags para o arquivo: ' + arquivos[i] + str(song.tags)
    sr, x = wav.read(diretorio+arquivos[i])
    print 'Sample Rate =',sr

