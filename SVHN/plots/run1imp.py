import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,1)
supacc = np.load('imp-0015.npy')
curlacc = np.load('imp-04-0015.npy')
supacc2 = np.load('imp-0030.npy')
curlacc2 = np.load('imp-04-0030.npy')
supacc3 = np.load('imp-0060.npy')
curlacc3 = np.load('imp-10-0060.npy')

supbatches = np.arange(supacc.shape[0])*1000
curlbatches = np.arange(curlacc.shape[0])*1000 + supbatches[-1]
supbatches2= np.arange(supacc2.shape[0])*1000
curlbatches2= np.arange(curlacc2.shape[0])*1000 + supbatches2[-1]
supbatches3= np.arange(supacc3.shape[0])*1000
curlbatches3= np.arange(curlacc3.shape[0])*1000 + supbatches3[-1]

ax.plot(supbatches, supacc, linestyle='--', c='C1')
ax.plot([supbatches[-1],curlbatches[0]], [supacc[-1],curlacc[0]], c='b' )
run1 = ax.plot(curlbatches, curlacc, c='C1', label='110 labeled, 2930 unlabeled')

ax.plot(supbatches2, supacc2, linestyle='--', c='C2')
run2 = ax.plot(curlbatches2, curlacc2, c='C2', label='220 labeled, 2930 unlabeled')
ax.plot([supbatches2[-1],curlbatches2[0]], [supacc2[-1],curlacc2[0]], c='b')

ax.plot(supbatches3, supacc3, linestyle='--', c='C3')
run3 = ax.plot(curlbatches3, curlacc3, c='C3', label='440 labeled, 7325 unlabeled')
ax.plot([supbatches3[-1],curlbatches3[0]], [supacc3[-1],curlacc3[0]], c='b', label='CURL')

ax.set_title("Boost from CURL (no cheat)")
ax.set_xlabel("supervised batches")
ax.set_ylabel("Test Accuracy")
ax.legend()
fig.savefig('curlboost-imp.pdf', bbox_inches='tight')
plt.show()

