import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from pathlib import Path

# set oclumns settings

grade_cols = [
    'g_12',
    'g_10',
    'g_7',
    'g_4',
    'g_02',
    'g_00',
    'g_n3',
]

grade_cols_frac = [g.replace('g', 'p') for g in grade_cols]
grade_cols_frac.append('p_ej_modt')

# columns we dont want here


df_test = pd.read_pickle('./dataframes/df_test.pkl')
df_train = pd.read_pickle('./dataframes/df_train.pkl')

# label encoding here
le = LabelEncoder()

le.fit(df_train.fac)

dtrain = xgb.DMatrix(df_train[grade_cols_frac].values, label=le.transform(df_train.fac))

dtest = xgb.DMatrix(df_test[grade_cols_frac].values, label=le.transform(df_test.fac))

test_Y = le.transform(df_test.fac)
print(np.unique(test_Y))
exit()

watchlist = [(dtrain, 'train'), (dtest, 'test')]

param = {
    'max_depth': 10,
    'objective': 'multi:softmax',
    'nthread': 4,
    'num_class': len(list(le.classes_)),
}

num_round = 20

bst = xgb.train(param, dtrain, num_round, watchlist, )
bst.dump_model('dump.raw.txt')

# get prediction
pred = bst.predict(dtest)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

fig2, ax2 = plt.subplots(figsize=(6, 10))
xgb.plot_tree(bst, ax=ax2, rankdir='LR')
fig2.tight_layout()
fig2.savefig(Path('./Figure_1.pdf'), dpi=1000)
