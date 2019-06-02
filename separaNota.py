import os
import re


# lista com os nomes dos arquivos
diretorio = 'AmostrasTratadas/'
arquivos = os.listdir(diretorio)
arquivos.sort()
Notas = ["A", "B", "C", "D", "E", "F", "G"]


def separaNotas():
    '''
    Do conjunto de 416 arquivos, organiza-os em uma lista por notas.
    Exemplo de saida:
    ["Cello_C3.wav", "Cello_C4.wav", "Cello_C5.wav, "Clarineta_C3.wav",
     "Clarineta_C4.wav"]

    @return lisa : lista de listas contendo as amostras de cada mostra ordenada
    Apenas notas naturais, nao apresenta sustenidos.
    '''

    # lista que guarda arquivos de mesma nota
    ArquivosNotas = []

    for i in range(len(Notas)):
        subs = "." + Notas[i] + "[0-9]."
        arq = [x for x in arquivos if re.search(subs, x)]
        ArquivosNotas.append(arq)

    return ArquivosNotas


def separaInstrumentosMesmaNota(nota):
    '''
    Do conjunto de 416 arquivos separa todos os instrumentos que executam
    a nota passada por parametro e retorna o resultado em lista.
    Exemplo de saida: ["Cello_C5.wav", "Clarineta_C5.wav", "Flauta_C5.wav"]
    '''

    # serparacao por intrumentos para gerar graficos
    classes = ['Cello', 'Clarineta', 'Flauta', 'Oboe',
               'Trombone', 'Trompete', 'Viola', 'Violino']

    # nota desejada
    Nota = nota[0]
    oitava = nota[1]
    idx = Notas.index(Nota)
    Instrumentos = []

    for i in range(len(classes)):
        subs = classes[i] + "." + Notas[idx] + oitava + "."
        inst = [x for x in arquivos if re.search(subs, x)]
        Instrumentos.append(inst)

    return Instrumentos
