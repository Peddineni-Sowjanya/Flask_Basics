from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model using pickle
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Create an array of input features
        features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Convert prediction to a readable form (e.g., species name)
        species = ["Setosa", "Versicolor", "Virginica"]
        predicted_species = species[int(prediction[0])]
        
        return render_template('index.html', prediction=predicted_species)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)

