from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


df = pd.read_csv('dataset/alldata_up.csv')
dataset = df.to_numpy()

X = dataset [:,0:3]
y = dataset [:,3:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
result = regr.predict(X_test)
res = regr.score(X_test, y_test)
# print(result)
print("Based on Mean Square Error, the accuracy of the model is: ", res)

path = "trainedmodels/MLP1.sav"
pickle.dump(regr, open(path, 'wb'))