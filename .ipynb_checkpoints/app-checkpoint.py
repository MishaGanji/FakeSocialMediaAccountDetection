from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_rf_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    profile_pic = int(request.form['profile_pic'])
    nums_length_username = float(request.form['nums_length_username'])
    fullname_words = int(request.form['fullname_words'])
    nums_length_fullname = float(request.form['nums_length_fullname'])
    name_equals_username = int(request.form['name_equals_username'])
    description_length = int(request.form['description_length'])
    external_url = int(request.form['external_url'])
    private = int(request.form['private'])
    posts = int(request.form['posts'])
    followers = int(request.form['followers'])
    follows = int(request.form['follows'])

    # Create a feature vector
    features = np.array([[profile_pic, nums_length_username, fullname_words, nums_length_fullname,
                          name_equals_username, description_length, external_url, private, posts,
                          followers, follows]])

    # Predict using the loaded model
    prediction = model.predict(features)
    prediction_text = 'Fake' if prediction[0] == 1 else 'Not Fake'

    return f'The profile is {prediction_text}.'

if __name__ == '__main__':
    app.run(debug=True)
