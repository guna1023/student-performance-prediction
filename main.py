import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Data
data = pd.read_csv("dataset.csv")

X = data[['Hours_Studied', 'Attendance', 'Previous_Marks']]
y = data['Final_Result']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
previous = float(input("Enter previous marks: "))

prediction = model.predict([[hours, attendance, previous]])

print("Predicted Final Result:", round(prediction[0], 2))
