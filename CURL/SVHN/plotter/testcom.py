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
allsim = []
allcont = []

for n in range(runs):
    allsim.append([])
    allcont.append([])
    for i in range(len(tests)):
        sim = np.load(pathroot + f'sim-{n+2}-' + tests[i] + pathtail)
        cont = np.load(pathroot + f'cont-{n+2}-' + tests[i] + pathtail)
        allsim[n].append(sim)
        allcont[n].append(cont)

ax.plot(allsim[0][4][::5] )
ax.plot(allcont[0][4][::5] )

#ax.set_xscale('log')

ax.set_xlabel("iterations")
ax.set_ylabel("value")
#fig.savefig('curlboost-better2.pdf', bbox_inches='tight')
plt.show()
