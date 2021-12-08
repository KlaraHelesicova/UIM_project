import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def load_data(name_file):
    data = pd.read_csv(name_file, delimiter=';', usecols=[*range(0, 35), 38, 39])
    data_gender = pd.read_csv(name_file, delimiter=';', usecols=["Gender"])
    data_isSepsis = pd.read_csv(name_file, delimiter=';', usecols=["isSepsis"])
    data.describe().transpose()
    return data, data_gender, data_isSepsis


def erase_nan(data):
    median = data.median()
    std = data.std()
    outliers = (data - median).abs() > std
    data[outliers] = np.nan
    data.fillna(median, inplace=True)
    return data


def scale_data(data, d_gender, d_isSepsis):
    data = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data)
    scaled_data = pd.DataFrame(x_scaled)
    d_gender = d_gender.values # diky tomuto nahore neni Gender, ale 0
    d_gender = pd.DataFrame(d_gender)
    d_isSepsis = pd.DataFrame(d_isSepsis)
    scaled_data['Gender'] = d_gender
    scaled_data['isSepsis'] = d_isSepsis
    return scaled_data


def neural_network(result):
    target_column = ['isSepsis']
    predictors = list(set(list(result.columns))-set(target_column))
    result[predictors] = result[predictors]/result[predictors].max()
    result.describe().transpose()

    X = result[predictors].values
    y = result[target_column].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(X_train.shape)
    print(X_test.shape)

    mlp = MLPClassifier(activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train, y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))
    print(get_measure(y_train, predict_train))

    print(confusion_matrix(y_test, predict_test))
    print(classification_report(y_test, predict_test))
    print(get_measure(y_test, predict_test))


def get_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)
    PPV = TP / (TP + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return Se, Sp, PPV, ACC


loaded = load_data("dataSepsis.csv")
notnan = erase_nan(loaded[0])
scaling = scale_data(notnan, loaded[1], loaded[2])
hrk_hrk = neural_network(scaling)