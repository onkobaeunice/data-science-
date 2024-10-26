
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv('road_accidents.csv')  # Ensure this file is in the same directory

# Independent variables
X = data[['Weather_Conditions', 'Road_Surface_Conditions', 'Lighting_Conditions', 
          'Number_of_Vehicles', 'Number_of_Casualties', 'Speed_Limit']]

# Dependent variable
y = data['Accident_Severity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('accident_severity_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete and saved as accident_severity_model.pkl")
