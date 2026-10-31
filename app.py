
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and preprocessor
data = joblib.load("best_warranty_fraud_model.joblib")

model = data["model"]
preprocessor = data["preprocessor"]

print("Model and preprocessor loaded successfully!")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        features = [
            'Region', 'State', 'Area', 'City',
            'Consumer_profile', 'Product_category',
            'Product_type', 'AC_1001_Issue',
            'AC_1002_Issue', 'AC_1003_Issue',
            'TV_2001_Issue', 'TV_2002_Issue',
            'TV_2003_Issue', 'Claim_Value',
            'Service_Centre', 'Product_Age',
            'Purchased_from', 'Call_details', 'Purpose'
        ]

        input_data = {}

        product_type = request.form.get('Product_type')

        for feature in features:

            value = request.form.get(feature)

            # Set hidden product issue fields to 0
            if feature.startswith('AC_') and product_type == 'TV':
                value = 0

            elif feature.startswith('TV_') and product_type == 'AC':
                value = 0

            input_data[feature] = value

        # Convert numerical features
        numeric_features = [
            'Claim_Value', 'Product_Age', 'Call_details'
        ]

        for feature in numeric_features:
            input_data[feature] = float(input_data[feature])

        # Convert numeric categorical features
        integer_features = [
            'AC_1001_Issue', 'AC_1002_Issue',
            'AC_1003_Issue', 'TV_2001_Issue',
            'TV_2002_Issue', 'TV_2003_Issue',
            'Service_Centre'
        ]

        for feature in integer_features:
            input_data[feature] = int(input_data[feature])

        # Create DataFrame in the correct feature order
        input_df = pd.DataFrame([input_data], columns=features)

        # Apply trained preprocessing
        transformed_data = preprocessor.transform(input_df)

        # Predict using Logistic Regression
        prediction = model.predict(transformed_data)[0]

        result = "Fraudulent" if prediction == 1 else "Genuine"

        return render_template(
            'result.html',
            prediction=result
        )

    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
