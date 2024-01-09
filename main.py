import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm


data = pd.read_csv('/Users/master/Downloads/sat_gpa_v1.csv')


print(data.head())



sns.lmplot(x='sat_sum', y='fy_gpa', data=data, scatter_kws={'alpha':0.5})
plt.title('Relationship between GPA and SAT Scores')
plt.xlabel('SAT Scores')
plt.ylabel('College GPA')
plt.show()


X = sm.add_constant(data['SAT_Scores'])
y = data['College_GPA']

model = sm.OLS(y, X).fit()


print(model.summary())

