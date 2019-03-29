import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 20)
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# column names
orig_cols = [
    'faculty',
    'g_12',
    'g_10',
    'g_7',
    'g_4',
    'g_02',
    'g_00',
    'g_n3',
    'ikke_bestaet',
    'bestaet',
    'ej_modt',
    'institute',
    'term',
    'title',
    # 'fac',
]

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

term_list = [
    'Summer-2014',
    'Winter-2014',
    'Summer-2015',
    'Winter-2015',
    'Summer-2016',
    'Winter-2016',
    'Summer-2017',
    'Winter-2017',
    'Summer-2018',
    'Winter-2018',
]

# user dicts
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

fac_vectorize_dict = {
    'SUND':np.array([1,0,0,0]),
    'SCIENCE':np.array([0,1,0,0]),
    'SAMF':np.array([0,0,1,0]),
    'HUM':np.array([0,0,0,1]),
}

participant_thres = 300

grade_cols_frac = [g.replace('g', 'p') for g in grade_cols]
grade_cols_frac.append('p_ej_modt')


frac_dict = dict(zip(num_cols, grade_cols_frac))



# here we read the data from teh scraping
df = pd.read_json('/Users/mag/PycharmProjects/graded_class/output.json')

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

df = df[df.n_students > participant_thres]
print(len(df))


# get fractional values for grades
for old_col in num_cols:
    df[frac_dict[old_col]] = df[old_col]/df.n_students



fig, axes = plt.subplots(2,2)
ax = axes.ravel()

for i, (name, group) in enumerate(df.groupby('fac')):
    # print(group[grade_cols_frac[::-1]].mean())
    ax[i].set_title(name)
    ax[i].bar(range(len(grade_cols_frac)),group[grade_cols_frac[::-1]].mean(),label = name, alpha=0.5, color='xkcd:dark red')
    ax[i].set_xticks(range(len(grade_cols_frac)))
    ax[i].set_xticklabels([s.replace('p_','').replace('ej_modt','NS') for s in grade_cols_frac[::-1] ])

fig.tight_layout()
fig.savefig('figs/distplot.png')


fig2, ax2 = plt.subplots()
for i, (name, group) in enumerate(df.groupby('fac')):
    ax2.bar(np.array(range(len(grade_cols_frac))) + 0.2*i, group[grade_cols_frac[::-1]].mean(),0.2, label=name, alpha=0.5,
          # color='xkcd:dark red'
            )
ax2.set_xticks(range(len(grade_cols_frac)))
ax2.set_xticklabels([s.replace('p_', '').replace('ej_modt', 'NS') for s in grade_cols_frac[::-1]])
ax2.legend()

fig2.tight_layout()
fig2.savefig('figs/overlay.png', dpi=300)
