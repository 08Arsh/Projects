## IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

## LOAD DATA 
data = pd.read_csv(r'C:\Users\Admin\Downloads\total_cases.csv',sep =',')
data = data[['Sn','India','China','United States','United Kingdom']]
print('-'*30);print('HEAD');print('-'*30)
print(data.head())

## PREPARE DATA
print('-'*30);print('PREPARED DATA');print('-'*30)
x = np.array(data['Sn'][550:691]).reshape(-1, 1)

#India
y = np.array(data['India'][550:691]).reshape(-1, 1)

#USA
z = np.array(data['United States'][550:691]).reshape(-1, 1)

#China
k = np.array(data['China'][550:691]).reshape(-1, 1)

#UK
q = np.array(data['United Kingdom'][550:691]).reshape(-1, 1)

#plt.show()
polyFeat = PolynomialFeatures(degree=4)
x = polyFeat.fit_transform(x)
#print(x)

## TRAINING DATA 
print('-'*30);print('TRAINING DATA');print('-'*30)
model = linear_model.LinearRegression()
model1 = linear_model.LinearRegression()
model2 = linear_model.LinearRegression()
model3 = linear_model.LinearRegression()

model.fit(x,y)
accuracy = model.score(x,y)
model1.fit(x,z)
accuracy1 = model1.score(x,z)
model2.fit(x,k)
accuracy2 = model2.score(x,k)
model3.fit(x,q)
accuracy3 = model3.score(x,q)


## PREDICTION
days = 10
x1 = np.array(list(range(550,691+days))).reshape(-1,1)


print('-'*30);print('PREDICTION');print('-'*30,)

figure, axis=plt.subplots(2,2)

#India
print(f'Accuracy:{round(accuracy*100,3)} %')
print(f'Prediction – Total Cases after {days} days in India:',end='')
print(int(model.predict(polyFeat.fit_transform([[691+days]]))))
y1 = model.predict(polyFeat.fit_transform(x1))
axis[0,0].set_title("India")
axis[0,0].set_xlabel("Days")
axis[0,0].set_ylabel("Cases")
axis[0,0].plot(y,'-m')
axis[0,0].plot(y1,'--r')
#plt.show()
print("\n")

#USA
print(f'Accuracy:{round(accuracy1*100,3)} %')
print(f'Prediction – Total Cases after {days} days in USA:',end='')
print(int(model1.predict(polyFeat.fit_transform([[691+days]]))))
z1 = model1.predict(polyFeat.fit_transform(x1))
axis[0,1].set_title("USA")
axis[0,1].set_xlabel("Days")
axis[0,1].set_ylabel("Cases")
axis[0,1].plot(z,'-m')
axis[0,1].plot(z1,'--r')
#plt.show()
print("\n")

#China
print(f'Accuracy:{round(accuracy2*100,3)} %')
print(f'Prediction – Total Cases after {days} days in China:',end='')
print(int(model2.predict(polyFeat.fit_transform([[691+days]]))))
k1 = model2.predict(polyFeat.fit_transform(x1))
axis[1,0].set_title("China")
axis[1,0].set_xlabel("Days")
axis[1,0].set_ylabel("Cases")
axis[1,0].plot(k,'-m')
axis[1,0].plot(k1,'--r')
#plt.show()
print("\n")

#United Kingdom
print(f'Accuracy:{round(accuracy3*100,3)} %')
print(f'Prediction – Total Cases after {days} days in United Kingdom:',end='')
print(int(model3.predict(polyFeat.fit_transform([[691+days]]))))
q1 = model3.predict(polyFeat.fit_transform(x1))
axis[1,1].set_title("United Kingdom")
axis[1,1].set_xlabel("Days")
axis[1,1].set_ylabel("Cases")
axis[1,1].plot(q,'-m')
axis[1,1].plot(q1,'--r')
plt.show()

