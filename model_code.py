import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("salary_predict_dataset.csv")

data = data[['experience', 'test_score', 'interview_score', 'Salary']]
data = data.fillna(0)

X = np.array(data.drop(['Salary'],1))
y = np.array(data['Salary'])

model1 = LinearRegression()
model1.fit(X,y)

pickle.dump(model1,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


