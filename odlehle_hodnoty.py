from sklearn import preprocessing
import numpy as np
import pandas as pd
# ,2,3,4,5,6,7,8,9,10,11,12,13,14
df = pd.read_csv('dataSepsis.csv',delimiter=';', usecols=[*range(0, 35), 38, 39])
median = df.median()
std = df.std()
outliers = (df - median).abs() > std
df[outliers] = np.nan
df.fillna(median, inplace=True)



x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

print(df)


