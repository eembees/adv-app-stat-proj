# logistic regression on breast cancer dataset
import numpy as np
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

_, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]
#
# print(coefs[:,-1].shape)
# exit()
coef_indices_to_use = ind = np.argpartition(np.abs(coefs[:,-1]), -10)[-10:]

for i in coef_indices_to_use:
    plt.plot(xx, coefs[i], label = cancer.feature_names[i])

ymin, ymax = plt.ylim()
# plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
# plt.axis('tight')
plt.legend()
plt.show()

