import numpy as np
import matplotlib.pyplot as plt
import matplotlib


fig, ax = plt.subplots(1,1)


runs = 8
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
        sup = np.load(pathroot + f'sup-{n+1}-' + tests[i] + pathtail)
        unsup = np.load(pathroot + f'unsup-{n+1}-' + tests[i] + pathtail)
        allsups[i, n] = sup.max()
        allunsups[i, n] = unsup.max()

supmean = np.mean(allsups, axis=-1)
unsupmean = np.mean(allunsups, axis=-1)
supsd = np.std(allsups, axis=-1)
unsupsd = np.std(allunsups, axis=-1)

fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]

ax.plot(fractions, supmean, c='C1', label="Supervised Only")
ax.fill_between(fractions, supmean+supsd, supmean-supsd, color='C1', alpha=0.2)

#markers = ['o', '.', '+', 'x', '*', '<', '4', 'P']
#for n in range(runs):
#    for i in range(len(tests)):
#        ax.scatter(fractions[i], allsups[i, n], marker=markers[n], c='C1')
#        ax.scatter(fractions[i], allunsups[i, n], marker=markers[n], c='C2')

ax.plot(fractions, unsupmean, c='C2', label="Bootstrap CURL")
ax.fill_between(fractions, unsupmean+unsupsd, unsupmean-unsupsd, color='C2', alpha=0.2)

ax.set_xscale('log')
ax.set_xticks(fractions)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title('Bootstrapping CURL')
ax.set_xlabel("Fraction Labeled")
ax.set_ylabel("Test Accuracy")
ax.legend()
fig.savefig('bootstrap.pdf', bbox_inches='tight')
plt.show()
