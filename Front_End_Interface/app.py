from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def predict_churn(Age, CreditScore, Tenure, Balance, NumofProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    # create array of input values
    input_data = [[Age, CreditScore, Tenure, Balance, NumofProducts, HasCrCard, IsActiveMember, EstimatedSalary]]

    # load decision tree from file
    with open('dtc_modelNS.pkl', 'rb') as file:
        tree = pickle.load(file)

    # use predict method to get predicted probability
    churn_probability = tree.predict_proba(input_data)[:, 1][0]

    # convert predicted value to 'Churn' or 'Not Churn'
    if churn_probability >= 0.5:
        churn_label = 'Churn'
    else:
        churn_label = 'Not Churn'

    # return predicted value as a JSON object
    return {'churn_probability': churn_label}


# Flask route for handling prediction requests
@app.route('/check_churn', methods=['POST'])
def check_churn():
    data = request.get_json()
    Age = int(data['Age'])
    CreditScore = int(data['CreditScore'])
    Tenure = int(data['Tenure'])
    Balance = float(data['Balance'])
    NumofProducts = int(data['NumofProducts'])
    HasCrCard = int(data['HasCrCard'])
    IsActiveMember = int(data['IsActiveMember'])
    EstimatedSalary = float(data['EstimatedSalary'])
    
    # Call predict_churn function to get predicted value
    try:
        result = predict_churn(Age, CreditScore, Tenure, Balance, NumofProducts, HasCrCard, IsActiveMember, EstimatedSalary)
    except Exception as e:
        # Handle any errors that might occur
        return jsonify({'error': str(e)})
    
    # Return predicted value as a JSON object
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
