import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import l1_min_c
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split

n_coefs = 5
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


df = pd.read_csv('/Users/mag/PycharmProjects/graded_class/heart.csv')

X = df.iloc[:,:-1].values
y=df.target.values

names = df.iloc[:,:-1].keys()


# X, y = make_classification(n_samples=1000,n_features=15, n_redundant=0, n_informative=3, random_state=1, n_clusters_per_class=1,n_classes=2)




# standardize the features






# engineer some more features

## Make division
# divisionTransformer = FunctionTransformer(np.reciprocal, validate=True)
# X_train_trans = divisionTransformer.transform(X)
# X = np.concatenate([X, X_train_trans], axis=1)
#
# # add polynomial features
X = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(X)
# # X_train_new = np.concatenate([X_train_new, X_train_poly], axis=1)



# split the data
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(X[:,0],X[:,1],X[:,2],c=y)
# plt.show()

# cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3, 16)
clf = linear_model.LogisticRegression(penalty='l1', solver='saga',
                                      tol=1e-6, #max_iter=int(1e6),
                                      warm_start=True)

coefs_ = []
for c in cs:
    print('fitting with c ')
    print(c)
    clf.set_params(C=c)
    clf.fit(X_train, y_train)
    coefs_.append(clf.coef_.ravel().copy())

coefs_ = np.array(coefs_)
# print(coefs_.shape)
coef_indices_to_use = np.argpartition(np.abs(coefs_[-1,:]), -1 * n_coefs)[-1 * n_coefs:]
# print(coef_indices_to_use)

figp, axp = plt.subplots()
for i in coef_indices_to_use:
    try:
        axp.plot(np.log10(cs), coefs_[:,i], label = names[i]) #marker = 'o')
    except IndexError:
        axp.plot(np.log10(cs), coefs_[:,i], label = str(i)) #marker = 'o')

axp.legend()
figp.savefig('figs/subset_std_{}.png'.format(n_coefs))





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