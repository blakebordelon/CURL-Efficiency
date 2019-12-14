import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,1)

supfiles = ['sup-04-002.npy',
            'sup-04-004.npy',
            'sup-04-008.npy',
            'sup-04-016.npy',
            'sup-16-002.npy',
            'sup-16-004.npy',
            'sup-16-008.npy',
            'sup-16-016.npy',
            ]

unsupfiles = ['unsup-04-002.npy',
              'unsup-04-004.npy',
              'unsup-04-008.npy',
              'unsup-04-016.npy',
              'unsup-16-002.npy',
              'unsup-16-004.npy',
              'unsup-16-008.npy',
              'unsup-16-016.npy',
              ]

supacc = []
curlacc = []

for i in range(len(supfiles)):
    supacc.append(np.load(supfiles[i]))
    curlacc.append(np.load(unsupfiles[i]))

supbatches = []
curlbatches = []

for i in range(len(supfiles)):
    supbatches.append(np.arange(supacc[i].shape[0])*1000 )
    curlbatches.append(np.arange(curlacc[i].shape[0])*1000 + supbatches[i][-1])

for i in range(len(supfiles)):
    ax.plot(supbatches[i], supacc[i], linestyle='--', c='C'+str(i))
    ax.plot(curlbatches[i], curlacc[i], c='C'+str(i), label=supfiles[i][4:-4])

ax.set_title("Boost Comparisons")
ax.set_xlabel("supervised batches")
ax.set_ylabel("test accuracy")
ax.legend()
fig.savefig('curlboost-many.pdf', bbox_inches='tight')
plt.show()

