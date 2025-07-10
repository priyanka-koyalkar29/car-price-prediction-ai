from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset and train model
data = pd.read_csv('car_data.csv')
le = LabelEncoder()

data['Fuel_Type'] = le.fit_transform(data['Fuel_Type'])
data['Seller_Type'] = le.fit_transform(data['Seller_Type'])
data['Transmission'] = le.fit_transform(data['Transmission'])

X = data[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
y = data['Car_Price']

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('car_price_model.pkl', 'wb'))

# Reload model
model = pickle.load(open('car_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [int(request.form['year']),
              float(request.form['price']),
              int(request.form['kms']),
              int(request.form['fuel']),
              int(request.form['seller']),
              int(request.form['transmission']),
              int(request.form['owner'])]

    prediction = model.predict([inputs])[0]
    return render_template('index.html', prediction_text=f'Estimated Car Price: ₹ {round(prediction, 2)} Lakhs')

if __name__ == '__main__':
    app.run(debug=True)
