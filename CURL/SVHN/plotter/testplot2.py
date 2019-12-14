import numpy as np
import matplotlib.pyplot as plt
import matplotlib


fig, ax = plt.subplots(1,1)


runs = 4
pathroot = '../plots/'
pathtail = '.npy'
tests = ['0.001-0.099',
         '0.002-0.098',
         '0.005-0.095',
         '0.010-0.090',
         '0.020-0.080',
         '0.050-0.050',
         '0.100-0.000']
allsups = np.zeros((len(tests), runs))
allunsups = np.zeros((len(tests), runs))

for n in range(runs):
    for i in range(len(tests)):
        sup = np.load(pathroot + f'sup-{n+2}-' + tests[i] + pathtail)
        unsup = np.load(pathroot + f'unsup-{n+2}-' + tests[i] + pathtail)
        allsups[i, n] = sup.max()
        allunsups[i, n] = unsup.max()

benefit = np.zeros_like(allsups)
benefit = allunsups - allsups
benefitmean = np.mean(benefit, axis=-1)
benefitstd = np.std(benefit, axis=-1)

fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]

#ax.plot(fractions, benefitmean, c='C1')
#ax.plot(fractions, benefitmean+benefitstd, c='C1')
#ax.plot(fractions, benefitmean-benefitstd, c='C1')

for n in range(runs):
    ax.plot(fractions, benefit[:,n], label=f"run {n}")

ax.set_xscale('log')
ax.set_xticks(fractions)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend()

ax.set_xlabel("Fraction labeled")
ax.set_ylabel("Test Accuracy Gain")
#fig.savefig('curlboost-better2.pdf', bbox_inches='tight')
plt.show()
