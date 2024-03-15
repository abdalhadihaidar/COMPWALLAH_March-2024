from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('task1creditscope (3).pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction='')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    features = [float(request.form.get('num-bank-accounts', 0)),
                float(request.form.get('num-credit-card', 0)),
                float(request.form.get('interest-rate', 0)),
                float(request.form.get('num-of-loan', 0)),
                float(request.form.get('delay-from-due-date', 0)),
                float(request.form.get('num-of-delayed-payment', 0)),
                float(request.form.get('changed-credit-limit', 0))]
    
    # Make prediction
    prediction = model.predict([features])[0]

    # Map numerical prediction values to labels
    prediction_label = {0: 'Good', 1: 'Poor', 2: 'Standard', 3: 'nan'}[prediction]

    # Return the prediction result
    return render_template('index.html', prediction=prediction_label)



if __name__ == '__main__':
    app.run(debug=True)
