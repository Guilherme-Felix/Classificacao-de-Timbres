<h1> Classificação de timbres. </h1>
O objetivo deste trabalho é classificar um conjunto de 416 arquivos de áudio em classes de instrumento, ou seja, identificar o timbre de cada um deles.

O conjunto de dados (disponível na pasta <i> AmostrasTratadas </i>) consiste em 52 arquivos de áudio, em formato .wav a 44100 Hz stereo para cada 8 instrumentos: cello, clarineta, flauta, oboé, trombone, trompete, viola e violino.

Até o momento, o classificador usado é KNN (K-Nearest Neighbours), que consiste essencialmente em, dada uma métrica (como a distância euclidiana, por exemplo) classificar um novo elemento de acordo com a proximidade com um conjunto de elementos já classificados. Assim, já se conhecendo a classificação de alguns elementos, e classificação de um novo é dada de acordo com a métrica estabelecida: o novo elemento pertence à mesma classe dos elementos já classificados mais próximos dele (seus "vizinhos").
