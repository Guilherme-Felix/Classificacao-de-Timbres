import numpy as np 

k1 = np.arange(0, 6, 2)     # Do, Re, Mi
k2 = np.arange(5, 12, 2)    # Fa, Sol, La, Si
k = np.append(k1, k2)
k = np.append(k, k + 12)    # 2 oitavas
K = 2**(k/12.)

oitava = int(nota[1])
f_0 = 32.703194           # freq. de referencia
f = f_0*2**(oitava-1)     # Hz
