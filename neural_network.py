import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix

from sklearn.metrics import r2_score

df = pd.read_csv('dataSepsis.csv', delimiter=';', usecols=[*range(0, 35), 38, 39, 40])
d_gender = pd.read_csv('dataSepsis.csv', delimiter=';', usecols=["Gender"])
print(df.shape)
df.describe().transpose()

median = df.median()
std = df.std()
outliers = (df - median).abs() > std
df[outliers] = np.nan
df.fillna(median, inplace=True)

# returns a numpy array
# x = df.values
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)
#
# d_gender = d_gender.values # diky tomuto nahore neni Gender, ale 0
d_gender = pd.DataFrame(d_gender) # na tomhle nezalezi concat

frames = [df, d_gender]
result = pd.concat(frames, axis=1)
print(result)

target_column = ['isSepsis']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()

X = df[predictors].values
y = df[target_column].values.ravel()

kf = KFold(n_splits=10)
for train, test in kf.split(X):
    print("%s %s" % (train, test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    print(X_train.shape)
    print(X_test.shape)

    mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=5)
    mlp.fit(X_train, y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))

    print(confusion_matrix(y_test, predict_test))
    print(classification_report(y_test, predict_test))


