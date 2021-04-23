#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#feature scaling

#fitting simple linear regression into training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test results
y_pred=regressor.predict(X_test)

#visualizing the training set results
plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualizing the test set results
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

