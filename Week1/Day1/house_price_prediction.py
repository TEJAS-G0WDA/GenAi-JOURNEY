#Installed the following libraries:
#pandas, numpy, scikit-learn, matplotlib

#Importing the Libraries
import pandas as pd #Handle tables/data
import numpy as np #Numerical operation
import matplotlib.pyplot as plt #graph plotting

from sklearn.model_selection import train_test_split  #Split train/test data
from sklearn.linear_model import LinearRegression # ML Algorithm
from sklearn.metrics import mean_absolute_error  #Accuracy Testing


#Sameple House Data
#House Size (in sq ft) and Price (Rupees)

""" data = {
    'House_Size': [500, 700, 900, 1100, 1300,
                   1500, 1700, 1900, 2100, 2300],

    'Price': [1000000, 1400000, 1800000, 2200000, 2600000,
              3000000, 3400000, 3800000, 4200000, 4600000]
} """


data = {
    'House_Size': [500,700,900,1100,1300,
                   1500,1700,1900,2100,2300],

    'Price': [
        1050000, 1380000, 1820000, 2150000, 2700000,
        2950000, 3450000, 3760000, 4300000, 4550000
    ]
}

df = pd.DataFrame(data)

#print(df)

#Separate Input & Output
X = df[['House_Size']]
y = df['Price']


#Split Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Creating the Linear Regression Model
model = LinearRegression()

#Train the Model
model.fit(X_train, y_train)

#Predict the House price
y_pred = model.predict(X_test)

print("Testing Values:", X_test)
print("Predicted Prices:")
print(y_pred)

#Check the Accuracy of the Model using Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

#To Check Learned Parameters
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

#Predict Custom Input
""" new_house = [[2500]]

predicted_price = model.predict(new_house)

print("Predicted Price for 2500 sq ft house:", predicted_price[0]) """

#Visualising The Regression Line
plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Price Prediction")

plt.show()
