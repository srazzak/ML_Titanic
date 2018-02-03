import sklearn as sk
import numpy as np
import pandas as pd
import csv

from sklearn import svm

train_df = pd.read_csv('C:/Users/User/Documents/Python/Kaggle/Titanic/input/train.csv')
test_df = pd.read_csv('C:/Users/User/Documents/Python/Kaggle/Titanic/input/test.csv')
combine = [train_df, test_df]

print(train_df.columns.values, '\n')

desc_pClass = train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
desc_Sex = train_df[["Sex", "Survived"]].groupby((['Sex']), as_index=False).mean().sort_values(by='Survived', ascending=False)

print(desc_pClass, '\n')
print(desc_Sex)

