import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,1)
supacc = np.load('bimp-0015_2.npy')
curlacc = np.load('bimp-04-0015_2.npy')
supacc2 = np.load('bimp-0015.npy')
curlacc2 = np.load('bimp-04-0015.npy')
supacc3 = np.load('attractor-0015.npy')
curlacc3 = np.load('attractor-04-0015.npy')
supacc4 = np.load('naive-0015.npy')
curlacc4 = np.load('naive-04-0015.npy')

supbatches = np.arange(supacc.shape[0])*1000
curlbatches = np.arange(curlacc.shape[0])*1000 + supbatches[-1]
supbatches2= np.arange(supacc2.shape[0])*1000
curlbatches2= np.arange(curlacc2.shape[0])*1000 + supbatches2[-1]
supbatches3= np.arange(supacc3.shape[0])*1000
curlbatches3= np.arange(curlacc3.shape[0])*1000 + supbatches3[-1]
supbatches4= np.arange(supacc4.shape[0])*1000
curlbatches4= np.arange(curlacc4.shape[0])*1000 + supbatches3[-1]

ax.plot(supbatches, supacc, linestyle='--', c='C1')
ax.plot([supbatches[-1],curlbatches[0]], [supacc[-1],curlacc[0]], c='b' )
run1 = ax.plot(curlbatches, curlacc, c='C1', label='better')

ax.plot(supbatches2, supacc2, linestyle='--', c='C2')
run2 = ax.plot(curlbatches2, curlacc2, c='C2', label='better2')
#ax.plot([supbatches2[-1],curlbatches2[0]], [supacc2[-1],curlacc2[0]], c='b')

ax.plot(supbatches3, supacc3, linestyle='--', c='C3')
run3 = ax.plot(curlbatches3, curlacc3, c='C3', label='lighthouses')
#ax.plot([supbatches3[-1],curlbatches3[0]], [supacc3[-1],curlacc3[0]], c='b')

ax.plot(supbatches4, supacc4, linestyle='--', c='C4')
run4 = ax.plot(curlbatches4, curlacc4, c='C4', label='dumb')
#ax.plot([supbatches4[-1],curlbatches4[0]], [supacc4[-1],curlacc4[0]], c='b', label='CURL')

ax.set_title("Boost Comparisons (no cheat, 110 labeled, 2930 unlabeled)")
ax.set_xlabel("supervised batches")
ax.set_ylabel("Test Accuracy")
ax.legend()
fig.savefig('curlboost-better.pdf', bbox_inches='tight')
plt.show()

