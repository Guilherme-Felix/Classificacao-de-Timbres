import numpy as np
'''
Tentativa de determinar os MFCC para cada frame de audio.
Cada frame ira conter um vetor de coefientes cepstrais (iremos usar 3 e 4 primeiros de cada um).

Etapas:
    1 - Divide a amostra inteira em "frames" - Tipicamente 25ms de duracao.
    2 - Aplica a DCT a cada frame. Utiliza apenas a parte real (a amplitude ou magnitude - desconsidera-se a fase)
    (Ver Beth Loga - Cambridge)
    3 - Aplica o log no vetor de amplitudes
    4 - Converte o resultado para a escala "Mel"
    5 - Aplica uma transformacao nos coeficientes obtidos para descorrelacionar suas componetes
    (Em reconhecimento de voz a transformacao de Kahunnen-Loeve e aproximada usando uma DCT). 
    Tentaremos usar uma DCT aqui.
'''

def divide_em_frames(audio, samplerate, winlen=20e-3, winstep=10e-3):
    '''
    Divide um sinal de audio em blocos (frames)

    :param audio:      Sinal de audio a ser dividido. Deve ser um array N*1
    :param samplerate: Taxa de amostragem do audio a ser tratado
    :param winlen:     Tamanho da janela de tempo, em ms. Padrao, nesse caso, 20 ms.
    :param winstep:    Tamanho do passo da janela de tempo. Padrao, nesse caso, 10 ms.

    :returns: Um numpy array contendo o sinal dividido em NUM_JANELAS 
    '''
    tam_janela = np.int(winlen*samplerate)
    tam_passo_janela = np.int(winstep*samplerate)

    frames = []
    
    it = 0
    while (it < len(audio) ):
        frames.append(audio[it:it+tam_janela])
        it = it + tam_passo_janela

    frames = np.array(frames)
    return frames


