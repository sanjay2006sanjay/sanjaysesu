!pip install pandas
!pip install numpy
!pip install seaborn
    
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import zipfile

!pip install plotly
!pip install jupyterthemes
import plotly.express as px

from jupyterthemes import jtplot

jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False)
cancer_df = pd.read_csv('cervical_cancer.csv')
cancer_df.tail(20)
cancer_df.info()
cancer_df.describe()
cancer_df = cancer_df.replace('?', np.nan)
cancer_df

plt.figure(figsize=(20,20))
sns.heatmap(cancer_df.isnull(), yticklabels = False)
plt.show()
cancer_df = cancer_df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
cancer_df

cancer_df = cancer_df.apply(pd.to_numeric)
cancer_df.info()
cancer_df.describe()
cancer_df.mean()

cancer_df =  cancer_df.fillna(cancer_df.mean())
cancer_df

plt.figure(figsize=(8,20))
sns.heatmap(cancer_df.isnull(), yticklabels = False)
plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
plt.show()
cancer_df.describe()
corr_matrix = cancer_df.corr()

corr_matrix
plt.figure(figsize = (30,30))
sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.tick_params(labelsize=8)
plt.show()
cancer_df.hist(bins = 10, figsize = (30,30), color='blue')
plt.show()
target_df = cancer_df['Biopsy']
input_df = cancer_df.drop(['Biopsy'], axis=1)
X = np.array(input_df).astype('float32')
y = np.array(target_df).astype('float32')

y = y.reshape(-1,1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5)
!pip install --upgrade pip
!pip install seaborn
!pip install xgboost
model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 50, n_estimators = 100)

model.fit(X_train, y_train)
result_train = model.score(X_train, y_train)

result_train
result_test = model.score(X_test, y_test)

result_test
y_predict = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_predict))
cm = confusion_matrix(y_predict, y_test)

sns.heatmap(cm, annot = True)

plt.show()
