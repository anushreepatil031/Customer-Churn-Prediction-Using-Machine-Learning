import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
customer_data = pd.read_csv(r'C:\Users\lenovo\AppData\Local\Programs\churn analysis\Churn_Modelling.csv')
columns = customer_data.columns.values.tolist()
print(columns)
dataset = customer_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
dataset =  dataset.drop(['Geography', 'Gender'], axis=1)
Geography = pd.get_dummies(customer_data.Geography).iloc[:,1:]
Gender = pd.get_dummies(customer_data.Gender).iloc[:,1:]
dataset = pd.concat([dataset,Geography,Gender], axis=1)
X =  dataset.drop(['Exited'], axis=1)
y = dataset['Exited']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
classifier.fit(X_train, y_train)  
predictions = classifier.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))  
print(accuracy_score(y_test, predictions ))
feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


