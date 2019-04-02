import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import l1_min_c
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split

import warnings
warnings.filterwarnings("ignore")

n_coefs = 10
# cs = np.logspace(-3, 0, 15)

# load data
# dataset = load_breast_cancer()
# names = dataset.feature_names
# names = [name for name in names if 'error' not in name]
#
# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# # print(df.head())
#
# X = df[names].values
# # X = dataset.data
# y = dataset.target


df = pd.read_csv("/Users/mag/PycharmProjects/graded_class/SAHeart.data.txt")

df['famhist'] =np.array(df.famhist.values == 'Present', dtype=np.float)

X=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
names = list(df.iloc[:,1:-1].keys())


# df = pd.read_csv('/Users/mag/PycharmProjects/graded_class/heart.csv')
#
# X = df.iloc[:,:-1].values
# y=df.target.values
#
# names = df.iloc[:,:-1].keys()


# X, y = make_classification(n_samples=1000,n_features=15, n_redundant=0, n_informative=3, random_state=1, n_clusters_per_class=1,n_classes=2)




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
#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

cs = l1_min_c(X, y, loss='log') * np.logspace(-0.2, 4, 15)

clf = linear_model.LogisticRegressionCV(cv=5,
                                        Cs=cs,
                                        penalty='l1',
                                        solver='saga',
                                        # tol=1e-6,
                                        # max_iter=int(1e6),
                                        n_jobs=-1)

clf.fit(X_train, y_train)

coef_paths = clf.coefs_paths_[1]
cs = clf.Cs_

scores = clf.scores_[1]

best_score_ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
# print(best_score_ind)
# print(scores.shape)
# exit()
best_fold_ind = best_score_ind[0]
best_c_ind = best_score_ind[1]

# now for the plotting
names.append('w0') # added to account for intercept fitting in clf

fig, axes = plt.subplots(nrows=2, figsize = (6,10))
ax = axes.ravel()
for j, p in enumerate(coef_paths[best_fold_ind].T):
    ax[0].plot(np.log10(cs), p, label=names[j])
    # ax[0].plot(cs, p, label=names[j] if fi == 0 else None)
# for fi, ps in enumerate(coef_paths):
#     for j, p in enumerate(ps.T):
#         if fi == best_fold_ind:
#             ax[0].plot(cs, p, label=names[j])
#         # ax[0].plot(cs, p, label=names[j] if fi == 0 else None)

for fi, foldscore in enumerate(scores):
    ax[1].plot(np.log10(cs), foldscore, label='fold {}'.format(fi+1))


ax[0].axvline(np.log10(cs[best_c_ind]), ls='--',c='r',lw=1)
ax[1].axvline(np.log10(cs[best_c_ind]), ls='--',c='r',lw=1)


ax[0].set_title('Coefficient profiles for best performing fold ({})'.format(best_fold_ind+1))
ax[0].legend()
ax[1].set_title('Error rates profiles for each cross validation')
ax[1].legend()
# ax[1].set_ylim(0,1)

# ax.legend()
fig.tight_layout()
fig.savefig('./figs/LogRegCV.pdf')
# print(type(coef_paths.keys()))
# print(type(coef_paths.items()))
# print(type(coef_paths.values()))
# print(list(coef_paths.keys()))
# print(coef_paths.items())
# print(coef_paths.values())
# print(coef_paths.shape)
# exit()
#
#
#
# coefs_ = np.array(coefs_)
# # print(coefs_.shape)
# coef_indices_to_use = np.argpartition(np.abs(coefs_[-1,:]), -1 * n_coefs)[-1 * n_coefs:]
# # print(coef_indices_to_use)
#
# figp, axp = plt.subplots()
# for i in coef_indices_to_use:
#     try:
#         axp.plot(np.log10(cs), coefs_[:,i], label = names[i]) #marker = 'o')
#     except IndexError:
#         axp.plot(np.log10(cs), coefs_[:,i], label = str(i)) #marker = 'o')
#
# axp.legend()
# figp.savefig('figs/subset_brc_{}.png'.format(n_coefs))
#

# g = sns.pairplot(df,
#                  vars=names[coef_indices_to_use],
#                  hue='target',
#                  diag_kind='hist',
#                  height=2,
#                  markers='+',
#                  plot_kws=dict(s=5, edgecolor="b", linewidth=1),
#                  )
# plt.show()

# pipe = make_pipeline(
#     StandardScaler(),
#     divisionTransformer,
#     PolynomialFeatures(degree=2, interaction_only=True),
#     linear_model.LogisticRegression(penalty='l1', solver='liblinear',C=1, max_iter=1000)
#     # linear_model.lars_path(method='lasso',verbose=True)
# )
#
# # print(pipe)
# # subset selection using LASSO
# # evaluate_model(X_train, y_train, pipe)
#
# pipe.fit(X_train,y_train)
#
# y_pred = pipe.predict(X_test)
#
# corrects = np.where((y_pred == y_test))[0]
# print(len(corrects))
# print(len(y_pred))
# # print()