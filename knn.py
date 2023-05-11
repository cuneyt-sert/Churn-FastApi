import pandas as pd

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier



df = pd.read_csv('ChurnData.csv')


X_train, x_test, Y_train, y_test = train_test_split(df.iloc[:,1:7], df.iloc[:,0:1], 
                                                    test_size = 0.2, random_state=42)

std_scale = StandardScaler()

X_train_scaled = std_scale.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=11, weights= 'uniform', metric= 'manhattan' )
knn_model=knn.fit(X_train_scaled, Y_train.values.ravel())

print(knn_model.predict([[25,15,10,5,1,1]]))

import pickle

filename = "knn_model.pkl"
pickle.dump(knn_model, open(filename, "wb"))

