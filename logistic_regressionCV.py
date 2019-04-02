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

grade_cols = [
    'g_12',
    'g_10',
    'g_7',
    'g_4',
    'g_02',
    'g_00',
    'g_n3',
]

num_cols = [
    'g_12',
    'g_10',
    'g_7',
    'g_4',
    'g_02',
    'g_00',
    'g_n3',
    'ej_modt',
]

faculty_dict = {
    'Det Sundhedsvidenskabelige Fakultet': 'SUND',
    'Det Humanistiske Fakultet': 'HUM',
    'Det Natur- og Biovidenskabelige Fakultet': 'SCIENCE',
    'Det Samfundsvidenskabelige Fakultet': 'SAMF',
    'Det Juridiske Fakultet': 'SAMF',
    # 'Det Juridiske Fakultet': 'JUR',
    'Det Teologiske Fakultet': 'HUM',
    # 'Det Teologiske Fakultet': 'TEO',
}


grade_cols_frac = [g.replace('g', 'p') for g in grade_cols]
grade_cols_frac.append('p_ej_modt')


frac_dict = dict(zip(num_cols, grade_cols_frac))


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

#
# df = pd.read_csv("/Users/mag/PycharmProjects/graded_class/SAHeart.data.txt")
#
# df['famhist'] =np.array(df.famhist.values == 'Present', dtype=np.float)
#
# X=df.iloc[:,1:-1].values
# y=df.iloc[:,-1].values
# names = list(df.iloc[:,1:-1].keys())


# df = pd.read_csv('/Users/mag/PycharmProjects/graded_class/heart.csv')
#
# X = df.iloc[:,:-1].values
# y=df.target.values
#
# names = df.iloc[:,:-1].keys()


# X, y = make_classification(n_samples=1000,n_features=15, n_redundant=0, n_informative=3, random_state=1, n_clusters_per_class=1,n_classes=2)




# standardize the features

# Here for the university data
participant_thres = 20
# here we read the data from teh scraping
df = pd.read_json('./scraping/output.json')

# add faculty code
df['fac'] = [faculty_dict[val] for val in df.faculty]
# df['fv'] = [fac_vectorize_dict[val] for val in df.fac]

# remove all non numerical grade data (for example, if the course is too small for statistics, or if it is pass/fail)
df.dropna(subset=['g_10'], inplace=True)

# removing the pass and fail columns, those are NaN for all the remaining grades
df.drop(columns=['ikke_bestaet', 'bestaet'], inplace=True)

# fill nan as 0
df.fillna(0, inplace=True)

# make number of students in course
df['n_students'] = df[num_cols].sum(axis=1)

# df = df[df.n_students > participant_thres] # if thresholding by number of students
# print(len(df))

for old_col in num_cols:
    df[frac_dict[old_col]] = df[old_col]/df.n_students


df['target'] = np.array(df.fac.values == 'SCIENCE',dtype=int)
# print(df[df.fac.values == 'SCIENCE'])
# print(df.target.tail())

X = df[num_cols].values
y = df.target.values
names = num_cols.copy()

## Make division

# divisionTransformer = FunctionTransformer(np.reciprocal, validate=True)
# X_train_trans = divisionTransformer.transform(X)
# X_train_trans[X_train_trans>1000] = 1000.
# X = np.concatenate([X, X_train_trans], axis=1)
# #
# # add polynomial features
# X = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(X)
# # X_train_new = np.concatenate([X_train_new, X_train_poly], axis=1)


# split the data
X = StandardScaler().fit_transform(X)
#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# cs = l1_min_c(X, y, loss='log') * np.logspace(-0.2, 4, 15)
cs=20

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

best_score = scores[best_score_ind]

# now for the plotting
names.append('w0') # added to account for intercept fitting in clf

fig, axes = plt.subplots(nrows=2, figsize = (6,10))
ax = axes.ravel()
for j, p in enumerate(coef_paths[best_fold_ind].T):
    try:
        ax[0].plot(np.log10(cs), p, label=names[j])
    except IndexError:
        ax[0].plot(np.log10(cs), p, label='1/{}'.format(names[j - len(names)]))

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
ax[0].set_ylabel('Coefficient value')
ax[1].set_ylabel('True Positive Rate (zoomed)')
ax[1].set_xlabel('log10(C value)')


ax[0].set_title('Coefficient profiles for best performing fold ({})'.format(best_fold_ind+1))
ax[0].legend(loc='lower right')
ax[1].set_title('CV Classification rates. Max: {:.3f} at c={:.2f} '.
                format(best_score,np.log10(cs[best_c_ind])))
ax[1].legend(loc='upper right')\

# ax[1].set_ylim(0,1)

# ax.legend()
fig.tight_layout()
fig.savefig('./figs/LogRegCV_KU.pdf')
