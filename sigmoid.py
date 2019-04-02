import numpy as np
import matplotlib.pyplot as plt

def sigm(x):
    return np.divide(np.exp(x),(1 + np.exp(x)))

xarr = np.linspace(-10,10,1000)

yarr = sigm(xarr)

fig, ax = plt.subplots()

ax.plot(xarr, yarr,)# c='xkcd:hot pink')
ax.axhline(0.5, ls='--', c='xkcd:pastel red', lw=1)
ax.axvline(0, ls='--', c='xkcd:pastel red', lw=1)

ax.set_xlabel('$x_{i}^T\cdot w$')
ax.set_ylabel('$p(y = +1|x)$')

ax.set_title('Logistic Sigmoid Function')

ax.set_yticks([0,0.25,0.5,0.75,1])

fig.tight_layout()
fig.savefig('./figs/sigmoid.pdf')
# plt.show()