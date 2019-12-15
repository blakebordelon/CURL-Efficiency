import numpy as np
import matplotlib.pyplot as plt
import matplotlib


fig, ax = plt.subplots(1,1)


pathroot = '../plots/'
pathtail = '.npy'

sup = np.load(pathroot + 'sup' + pathtail)
unsup = np.load(pathroot + 'unsup' + pathtail)

avgsup = np.mean(sup[-20:])
avgunsup = np.mean(unsup[-20:])
print(f'avgsup = {avgsup}')
print(f'avgunsup = {avgunsup}')

ax.plot(sup, c='C1', label="Supervised Only")
ax.plot(unsup, c='C2', label="CURL")

#ax.set_title('Bootstrapping CURL')
#ax.set_xlabel("Fraction Labeled")
#ax.set_ylabel("Test Accuracy")
ax.legend()
plt.show()
