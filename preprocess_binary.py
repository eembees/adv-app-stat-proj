import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 20)
from sklearn.preprocessing import MultiLabelBinarizer


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
    'Det Sundhedsvidenskabelige Fakultet': 1,
    'Det Natur- og Biovidenskabelige Fakultet': 1,
    'Det Humanistiske Fakultet': 0,
    'Det Samfundsvidenskabelige Fakultet': 0,
    'Det Juridiske Fakultet': 0,
    # 'Det Juridiske Fakultet': 'JUR',
    'Det Teologiske Fakultet': 0,
    # 'Det Teologiske Fakultet': 'TEO',
}
    # faculty_dict = {
#     'Det Sundhedsvidenskabelige Fakultet': 'SCIENCE',
#     'Det Natur- og Biovidenskabelige Fakultet': 'SCIENCE',
#     'Det Humanistiske Fakultet': 'HUM',
#     'Det Samfundsvidenskabelige Fakultet': 'HUM',
#     'Det Juridiske Fakultet': 'HUM',
#     # 'Det Juridiske Fakultet': 'JUR',
#     'Det Teologiske Fakultet': 'HUM',
#     # 'Det Teologiske Fakultet': 'TEO',
# }

fac_vectorize_dict = {
    'SUND':np.array([1,0,0,0]),
    'SCIENCE':np.array([0,1,0,0]),
    'SAMF':np.array([0,0,1,0]),
    'HUM':np.array([0,0,0,1]),
}



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

# get fractional values for grades
for old_col in num_cols:
    df[frac_dict[old_col]] = df[old_col]/df.n_students

# from here we can output our raw data
# print(df.groupby('fac').size())


# now split our dataset into train and test datasets. We will use the 2018 data to test our predictions
df_train = df[~df.term.str.contains('2018')]
df_test = df[df.term.str.contains('2018')]

# print(df_train.term.unique())

df_train.to_pickle('./dataframes/df_train_bin.pkl')
df_test.to_pickle('./dataframes/df_test_bin.pkl')