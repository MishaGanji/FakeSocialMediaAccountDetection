# 🔍 Fake Social Media Profile Detection — AI-Powered Authenticity Checker

The **Fake Social Media Profile Detection** system is a machine learning-powered web application designed to analyze social media profiles and determine their authenticity. Using advanced algorithms and user behavior patterns, this tool helps identify potentially fake or suspicious accounts with high accuracy and confidence scores.

## ✨ Features

- Profile authenticity detection using a trained Random Forest model  
- 11-feature analysis including profile picture presence, username patterns, and bio metrics  
- Confidence scoring with probabilistic output  
- Heuristic fallback when model is unavailable (rule-based detection)  
- Responsive web interface built with HTML, CSS, and JavaScript  
- Interactive input form for manual profile data entry  
- Real-time visual results with classification breakdown  
- Detailed feature analysis explaining the prediction  
- Model loading via `joblib` for fast inference  

## 🖼️ Demo

### Main Analysis Interface  
<img src="static/1 (2).png" width="700"/>

### Authentic Profile Result  
<img src="static/2 (2).png" width="700"/>
<img src="static/3 (2).png" width="700"/>

### Fake Profile Detection  
<img src="static/4 (2).png" width="700"/>
<img src="static/5 (2).png" width="700"/>

## 🛠 Tech Stack

### Backend
- Python 3.8+
- Flask
- scikit-learn
- joblib
- NumPy

### Frontend
- HTML, CSS
- JavaScript
- CSS Grid & Flexbox

### Machine Learning
- Random Forest Classifier
- Feature Engineering
- Confidence scoring (probability-based)


## 🚀 How to Run the Project

### Requirements

- Python 3.8 or higher
- pip
- Trained model file (`best_rf_model.pkl`) — optional but recommended

### Setup Instructions

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-profile-detection
cd fake-profile-detection
```

#### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install flask scikit-learn joblib numpy
```

#### 4. Place your trained model file in the root directory
- Filename: best_rf_model.pkl

#### 5. Run the application
```bash
python app.py
```

Visit the app at: `http://localhost:5000`

## 🧠 How It Works

### Feature Extraction

- The system analyzes 11 features:
  1. Profile picture presence  
  2. Username numbers-to-length ratio  
  3. Number of words in full name  
  4. Fullname numbers-to-length ratio  
  5. Match between username and full name  
  6. Bio length  
  7. Presence of external URLs  
  8. Account privacy status  
  9. Post count  
  10. Follower count  
  11. Following count

### Detection Logic

- Random Forest model for predictions
- Fallback heuristic rules if the model is unavailable
- Confidence scoring via model probabilities
- Combines multiple features for robust detection

### Suspicious Patterns Detected

- No profile picture  
- Username has high number/letter ratio  
- Generic or short display name  
- Very short or empty bio  
- Unusual follower-to-following ratio  
- Accounts with no posts but many followers  


## 💻 Usage Examples

&nbsp;&nbsp;&nbsp;1. Enter profile data in the input form  
&nbsp;&nbsp;&nbsp;2. Click "Analyze Profile"  
&nbsp;&nbsp;&nbsp;3. View results with confidence score and visual breakdown


## 📌 Use Cases

- Social media platforms for fake account detection  
- Cybersecurity investigations  
- Dating apps for profile verification  
- Business networks to detect impersonation  
- Enterprise defense against social engineering  


## 📈 Model Performance

- Accuracy: ~85–90% on test data  
- High precision for fake profile detection  
- Balanced recall to reduce false negatives  
- Optimized F1-score  


