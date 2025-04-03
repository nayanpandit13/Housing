import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("delhi_housing_price.csv")

hp=df[['area','price']]

msk=np.random.rand(len(df))<0.8
train=hp[msk]
test=hp[~msk]

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
train_x = np.asanyarray(train[['area']])
train_y = np.asanyarray(train[['price']])
regr.fit (train_x, train_y)

test_x = np.asanyarray(test[['area']])
test_y = np.asanyarray(test[['price']])
print(regr.score(test_x,test_y))
#open a file where you want to store the data
file=open('model.pkl','wb')
pickle.dump(regr,file)
file.close()
