from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("./reg_model_fish.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        species = data['Species']
        length1 = float(data['Length1'])
        length2 = float(data['Length2'])
        length3 = float(data['Length3'])
        height = float(data['Height'])
        width = float(data['Width'])

        # Prepare data for prediction
        species_data = {
            'Species_Bream': species == 'Bream',
            'Species_Parkki': species == 'Parkki',
            'Species_Perch': species == 'Perch',
            'Species_Pike': species == 'Pike',
            'Species_Roach': species == 'Roach',
            'Species_Smelt': species == 'Smelt',
            'Species_Whitefish': species == 'Whitefish',
        }

        # Create a DataFrame for the features
        df = pd.DataFrame({
            'Length1': [length1],
            'Length2': [length2],
            'Length3': [length3],
            'Height': [height],
            'Width': [width],
            **species_data
        })

        # Make prediction
        prediction = model.predict(df)

        # Return the prediction
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
