import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('dataSepsis.csv', delimiter=';', usecols=[*range(0, 35), 38, 39])
d_gender = pd.read_csv('dataSepsis.csv', delimiter=';', usecols=["Gender"])
d_isSepsis = pd.read_csv('dataSepsis.csv', delimiter=';', usecols=["isSepsis"])
df.describe().transpose()

median = df.median()
std = df.std()
outliers = (df - median).abs() > std
df[outliers] = np.nan
df.fillna(median, inplace=True)

x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(x_scaled)

d_gender = d_gender.values # diky tomuto nahore neni Gender, ale 0
d_gender = pd.DataFrame(d_gender) # na tomhle nezalezi concat
d_isSepsis = pd.DataFrame(d_isSepsis)

df['Gender'] = d_gender
df['isSepsis'] = d_isSepsis
result = df


target_column = ['isSepsis']
predictors = list(set(list(result.columns))-set(target_column))
result[predictors] = result[predictors]/result[predictors].max()
result.describe().transpose()

X = result[predictors].values
y = df[target_column].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=40)
print(X_train.shape)
print(X_test.shape)

mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))


FP = confusion_matrix(y_train, predict_train).sum(axis=0) - np.diag(confusion_matrix(y_train, predict_train))
FN = confusion_matrix(y_train, predict_train).sum(axis=1) - np.diag(confusion_matrix(y_train, predict_train))
TP = np.diag(confusion_matrix(y_train, predict_train))
TN = confusion_matrix(y_train, predict_train).sum() - (FP + FN + TP)

Se = TP/(TP+FN)
Sp = TN/(TN+FP)
PPV = TP/(TP+FP)
ACC = (TP+TN)/(TP+FP+FN+TN)
print(Se, Sp, PPV, ACC)


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    return TP, FP, TN, FN


print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))

FP = confusion_matrix(y_test, predict_test).sum(axis=0) - np.diag(confusion_matrix(y_test, predict_test))
FN = confusion_matrix(y_test, predict_test).sum(axis=1) - np.diag(confusion_matrix(y_test, predict_test))
TP = np.diag(confusion_matrix(y_test, predict_test))
TN = confusion_matrix(y_test, predict_test).sum() - (FP + FN + TP)


x = perf_measure(y_train, predict_train)
Se1 = x[0]/(x[0]+x[3])
Sp1 = x[2]/(x[2]+x[1])
print(x, Se1, Sp1)

Se = TP/(TP+FN)
Sp = TN/(TN+FP)
PPV = TP/(TP+FP)
ACC = (TP+TN)/(TP+FP+FN+TN)
print(Se, Sp, PPV, ACC)




