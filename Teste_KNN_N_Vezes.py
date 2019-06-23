import Testes_ClassificaTimbreKNN_mod as mod

feat, VetAtrib = mod.preProcessamento()

# Roda para cada conjunto de atributos
acc = []      # guarda as acuracias para os 6 vetores de atributos
m_confs = []  # guarda as matrizes de confusao para os 6 vetores

S = len(feat)
rnd = 100
avg_acc = mod.np.zeros(S)

for r in range(rnd):
    for i in range(S):
        ac, mc = mod.classifica(feat[i], mod.rotulos, mod.num_classes,
                                mod.tam_classe, mod.tam_teste)
        acc.append(ac)
        m_confs.append(mc)

        avg_acc[i] = (avg_acc[i] + ac)
        print "Vetor: ", i, "Acuracia media: ", avg_acc[i]

avg_acc = avg_acc/float(rnd)
print 'Vetor Acuracia: ', avg_acc
tit = "Acuracia media: 100 rodadas"
for i in range(S):
    mod.gravaArquivoSaida(tit, VetAtrib[i], avg_acc[i])

print "Tudo certo!"
