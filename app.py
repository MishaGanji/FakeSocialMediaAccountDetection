from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model with error handling
try:
    model = joblib.load('best_rf_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: Model file 'best_rf_model.pkl' not found. Using dummy predictions.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def calculate_username_ratio(username):
    """Calculate the ratio of numbers to total length in username"""
    if not username:
        return 0
    num_count = sum(1 for char in username if char.isdigit())
    return num_count / len(username) if len(username) > 0 else 0

def calculate_fullname_ratio(fullname):
    """Calculate the ratio of numbers to total length in fullname"""
    if not fullname:
        return 0
    num_count = sum(1 for char in fullname if char.isdigit())
    return num_count / len(fullname) if len(fullname) > 0 else 0

def count_fullname_words(fullname):
    """Count words in fullname"""
    if not fullname:
        return 0
    return len(fullname.strip().split())

def dummy_prediction(features):
    """Dummy prediction logic when model is not available"""
    profile_pic, nums_length_username, fullname_words, nums_length_fullname, \
    name_equals_username, description_length, external_url, private, posts, \
    followers, follows = features[0]
    
    # Simple heuristic-based prediction
    fake_indicators = 0
    
    # Check for suspicious patterns
    if profile_pic == 0:  # No profile picture
        fake_indicators += 1
    if nums_length_username > 0.5:  # Username has many numbers
        fake_indicators += 1
    if fullname_words < 2:  # Very short name
        fake_indicators += 1
    if description_length == 0:  # No description
        fake_indicators += 1
    if followers == 0 and posts == 0:  # No activity
        fake_indicators += 1
    if follows > followers * 10 and followers < 100:  # Suspicious follow ratio
        fake_indicators += 1
    
    # Return 1 (fake) if 3 or more indicators, 0 (real) otherwise
    return [1 if fake_indicators >= 3 else 0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        username = request.form.get('username', '').strip()
        fullname = request.form.get('fullname', '').strip()
        
        # Calculate derived features
        profile_pic = 1 if request.form.get('profile_pic') == 'on' else 0
        nums_length_username = calculate_username_ratio(username)
        fullname_words = count_fullname_words(fullname)
        nums_length_fullname = calculate_fullname_ratio(fullname)
        name_equals_username = 1 if username.lower() == fullname.lower().replace(' ', '') else 0
        description_length = int(request.form.get('description_length', 0))
        external_url = 1 if request.form.get('external_url') == 'on' else 0
        private = 1 if request.form.get('private') == 'on' else 0
        posts = int(request.form.get('posts', 0))
        followers = int(request.form.get('followers', 0))
        follows = int(request.form.get('follows', 0))

        # Create a feature vector
        features = np.array([[profile_pic, nums_length_username, fullname_words, nums_length_fullname,
                              name_equals_username, description_length, external_url, private, posts,
                              followers, follows]])

        # Predict using the loaded model or dummy prediction
        if model is not None:
            prediction = model.predict(features)
            confidence = model.predict_proba(features)[0].max() * 100
        else:
            prediction = dummy_prediction(features)
            confidence = 75.0  # Default confidence for dummy prediction

        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'confidence': round(confidence, 1),
            'prediction_text': 'Fake Profile' if prediction[0] == 1 else 'Authentic Profile',
            'features_used': {
                'Has Profile Picture': 'Yes' if profile_pic else 'No',
                'Username Number Ratio': f"{nums_length_username:.2f}",
                'Fullname Words': fullname_words,
                'Fullname Number Ratio': f"{nums_length_fullname:.2f}",
                'Name Equals Username': 'Yes' if name_equals_username else 'No',
                'Description Length': description_length,
                'Has External URL': 'Yes' if external_url else 'No',
                'Private Account': 'Yes' if private else 'No',
                'Posts Count': posts,
                'Followers Count': followers,
                'Following Count': follows
            }
        }

        return render_template('result.html', result=result)

    except ValueError as e:
        return f"Error: Please enter valid numbers for numeric fields. {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        
        # Extract features from JSON
        features = np.array([[
            data.get('profile_pic', 0),
            data.get('nums_length_username', 0),
            data.get('fullname_words', 0),
            data.get('nums_length_fullname', 0),
            data.get('name_equals_username', 0),
            data.get('description_length', 0),
            data.get('external_url', 0),
            data.get('private', 0),
            data.get('posts', 0),
            data.get('followers', 0),
            data.get('follows', 0)
        ]])

        # Predict
        if model is not None:
            prediction = model.predict(features)
            confidence = model.predict_proba(features)[0].max() * 100
        else:
            prediction = dummy_prediction(features)
            confidence = 75.0

        return jsonify({
            'prediction': int(prediction[0]),
            'confidence': round(confidence, 1),
            'prediction_text': 'Fake Profile' if prediction[0] == 1 else 'Authentic Profile'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)