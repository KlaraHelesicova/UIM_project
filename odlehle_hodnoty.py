from sklearn import preprocessing
import numpy as np
import pandas as pd

df = pd.read_csv('dataSepsis.csv',delimiter=';', usecols=[*range(0, 35), 38, 39])
median = df.median()
std = df.std()
outliers = (df - median).abs() > std
df[outliers] = np.nan
df.fillna(median, inplace=True)



x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

print(df)


