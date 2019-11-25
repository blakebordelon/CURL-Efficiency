import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1,1)
supacc = np.load('0015.npy')
curlacc = np.load('04-0015.npy')

ax.plot(supacc, label='Supervised')
ax.plot(curlacc, label='Unsupervised')
ax.set_title("Supervised vs Sup + CURL")
ax.set_xlabel("batches")
ax.set_ylabel("Test Accuracy")
ax.legend()
#fig.savefig('plot/width_v_depth.pdf', bbox_inches='tight')
plt.show()

