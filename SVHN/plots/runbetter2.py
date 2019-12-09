import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,1)
supacc = np.load('bimp-0015_2.npy')
curlacc = np.load('bimp-04-0015_2.npy')
supacc4 = np.load('naive-0015.npy')
curlacc4 = np.load('naive-04-0015.npy')

supbatches = np.arange(supacc.shape[0])*1000
curlbatches = np.arange(curlacc.shape[0])*1000 + supbatches[-1]
supbatches4= np.arange(supacc4.shape[0])*1000
curlbatches4= np.arange(curlacc4.shape[0])*1000 + supbatches4[-1]

ax.plot(supbatches, supacc, linestyle='--', c='C1')
#ax.plot([supbatches[-1],curlbatches[0]], [supacc[-1],curlacc[0]], c='b' )
run1 = ax.plot(curlbatches, curlacc, c='C1', label='better')


ax.plot(supbatches4, supacc4, linestyle='--', c='C4')
run4 = ax.plot(curlbatches4, curlacc4, c='C4', label='dumb')
#ax.plot([supbatches4[-1],curlbatches4[0]], [supacc4[-1],curlacc4[0]], c='b', label='CURL')

ax.set_title("Boost Comparisons (no cheat, 110 labeled, 2930 unlabeled)")
ax.set_xlabel("supervised batches")
ax.set_ylabel("Test Accuracy")
ax.legend()
fig.savefig('curlboost-better2.pdf', bbox_inches='tight')
plt.show()

