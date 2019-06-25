import Testes_ClassificaTimbreKNN_mod as mod
import time

feat, VetAtrib = mod.preProcessamento()

# Roda para cada conjunto de atributos
acc = []      # guarda as acuracias para os 6 vetores de atributos
m_confs = []  # guarda as matrizes de confusao para os 6 vetores

S = len(feat)
rnd = 100
avg_acc = mod.np.zeros(S)

ac_f, nFeat_f, mc_f = mod.Fwd(feat, mod.rotulos, mod.num_classes,
                              mod.tam_classe, mod.tam_teste)
ac_b, nFeat_b, mc_b = mod.Bwd(feat, mod.rotulos, mod.num_classes,
                              mod.tam_classe, mod.tam_teste)

print 'Progressivo'
print ac_f
print nFeat_f
print 40*'-'

print 'Regressivo'
print ac_b
print nFeat_b
print 40*'-'

idx_max = mod.np.argmax(ac_f[:4])
max_ac_f = ac_f[idx_max]
m_conf_f = mc_f[idx_max]

idx_max = mod.np.argmax(ac_b[:4])
max_ac_b = ac_b[idx_max]
m_conf_b = mc_b[idx_max]

nome = "Progressivo"
tit = "Acuracia: " + str(round(max_ac_f, 2))
mod.plotResultados(tit, m_conf_f, nome)

time.sleep(1)

nome = "Regressivo"
tit = "Acuracia: " + str(round(max_ac_b, 2))
mod.plotResultados(tit, m_conf_b, nome)

for i in range(S - 2):
    tit = "Acuracia Progressivo"
    mod.gravaArquivoSaida(tit, VetAtrib[i], ac_f[i], nFeat_f[i])
    tit = "Acuracia Regressivo"
    mod.gravaArquivoSaida(tit, VetAtrib[i], ac_b[i], nFeat_b[i])

print "Tudo certo!"
