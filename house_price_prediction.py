
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder

#Load the dataset
data=pd.read_csv("./Housing.csv")
df=pd.DataFrame(data)
print(df)

#EDA OF THE DATA
print("\n--Basic information--\n")
print(data.info())
print("\nMissing values\n")
print(data.isnull().sum())
print("\nDescribe the data\n")
print(data.describe())

# Preprocessing:Convert categorical variables to numerical
categorical_cols=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
label=LabelEncoder()
for col in categorical_cols:
    df[col]=label.fit_transform(df[col])

#Splitting the data
x=df.drop('price',axis=1)
y=df['price']

#Train-Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

#Initialize and train the model
model=LinearRegression()
model.fit(x_train,y_train)

#Evalaute
y_pred=model.predict(x_test)
print("\nModel Performance\n")
print("MAE:",mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))

#Visualization
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred,alpha=0.6,color='royalblue')
plt.plot([y.min(),y.max()] ,[y.min(),y.max()],'r--',lw=2)
plt.xlabel('Actual Price',fontsize=12)
plt.ylabel('Predicted Price',fontsize=12)
plt.title('Actual vs Predicted House Prices',fontsize=14)
plt.grid(True,alpha=0.3)
plt.savefig("House_Price_graph")
plt.show()


