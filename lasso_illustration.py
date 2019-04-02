import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer, make_classification, load_diabetes
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import l1_min_c
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split

n_coefs = 9
cs = np.logspace(-2, 0, 15)

# load data
# dataset = load_breast_cancer()
# X = dataset.data
# y = dataset.target

# df = pd.read_csv("/Users/mag/PycharmProjects/graded_class/SAHeart.data.txt")
#
# df['famhist'] =np.array(df.famhist.values == 'Present', dtype=np.float)
#
# X=df.iloc[:,1:-1].values
# y=df.iloc[:,-1].values
# names = df.iloc[:,1:-1].keys()


# df = pd.read_csv('/Users/mag/PycharmProjects/graded_class/heart.csv')
#
# X = df.iloc[:,:-1].values
# y=df.target.values
#
# names = df.iloc[:,:-1].keys()


# X, y = make_classification(n_samples=1000,n_features=15, n_redundant=0, n_informative=3, random_state=1, n_clusters_per_class=1,n_classes=2)


diab = load_diabetes()

X = diab.data
y = diab.target
names = diab.feature_names

# dataset = load_breast_cancer()
# names = dataset.feature_names
# names = [name for name in names if 'error' not in name]
#
# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# print(df.head())
#
# X = df[names].values
# # X = dataset.data
# y = dataset.target

# standardize the features






# engineer some more features

## Make division
# divisionTransformer = FunctionTransformer(np.reciprocal, validate=True)
# X_train_trans = divisionTransformer.transform(X)
# X = np.concatenate([X, X_train_trans], axis=1)
#
# # add polynomial features
# X = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(X)
# # X_train_new = np.concatenate([X_train_new, X_train_poly], axis=1)


# split the data
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


_, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]
#

coef_indices_to_use = np.argpartition(np.abs(coefs[:,-1]), -n_coefs)[-n_coefs:]

fig, ax = plt.subplots(figsize=(6,4))
for i in coef_indices_to_use:
    ax.plot(xx, coefs[i], label = names[i])

ymin, ymax = ax.get_ylim()

# for x in xx[:n_coefs]:
#     ax.axvline(x, ymin, ymax, ls='--', linewidth=1)

# ax.set_xlabel('|coef| / max|coef|')
ax.set_xlabel('Shrinkage factor s = $\lambda/\sum^n_{i=1} | w_i |$')
ax.set_ylabel('Coefficients')
ax.set_title('LASSO Path on diabetes dataset')
# plt.axis('tight')
ax.legend(loc='upper left')

fig.tight_layout()
fig.savefig('./figs/LASSO_paths.pdf')

