import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_data():
    data = pd.read_csv('data/agaricus-lepiota.data', index_col=False, header=None,
                       names=['target', 'x1', 'x2', 'x3', 'x4', 'x5',
                              'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12',
                              'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19',
                              'x20', 'x21', 'x22'])
    return data

def preprocess_data(data):
    x, y = deal_feature(data)
    x = deal_pca(x)
    return x, y

def deal_feature(data):
    l_data = data.loc[data['x11'].isin(['?'])]
    t_data = data.loc[data['x11'] != '?']
    t_x = t_data.drop(columns=['x11'])
    t_x = pd.get_dummies(t_x)
    t_y = t_data['x11']
    x11_l = LabelEncoder()
    t_y = x11_l.fit_transform(t_y)
    l_x = l_data.drop(columns=['x11'])
    l_x = pd.get_dummies(l_x)
    f = l_x.columns.intersection(t_x.columns)
    t_x = t_x.loc[:, f]
    l_x = l_x.loc[:, f]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(t_x, t_y)
    x11 = knn.predict(l_x)
    x11 = x11_l.inverse_transform(x11)
    l_data['x11'] = x11
    data = pd.concat([t_data, l_data])
    target_le = LabelEncoder()
    y = target_le.fit_transform(data['target'])
    x = pd.get_dummies(data.drop(columns=['target']))
    return x, y

def deal_pca(data):
    pca = PCA(n_components=70)
    return pca.fit_transform(data)

def train_and_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2)
    LR = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
    LR.fit(x_train, y_train)
    score = LR.score(x_test, y_test)
    return score

if __name__ == '__main__':
    data = load_data()
    x, y = preprocess_data(data)
    score = train_and_test(x, y)