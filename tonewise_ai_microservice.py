# tonewise_ai_microservice.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from collections import defaultdict

# === Load Model File Paths ===
model_path = 'tonewise_model.pkl'
vectorizer_path = 'tonewise_vectorizer.pkl'

# === Try to Load the Pre-trained Model and Vectorizer ===
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

# === Initialize App ===
app = Flask(__name__)

# === Personal Commenter Tone Profiles ===
commenter_profiles = defaultdict(lambda: {
    'total_comments': 0,
    'positive': 0,
    'negative': 0,
    'neutral': 0
})

# === Casing Tone Analyzer ===
def analyze_casing(text):
    if text.isupper():
        return "STRONG EMPHASIS or URGENCY"
    elif text.islower():
        return "Neutral or Passive"
    elif text.istitle():
        return "Proper Formatting (Title Case)"
    else:
        return "Conversational or Mixed"

# === Commenter Profile Updater ===
def update_profile(commenter_id, prediction):
    profile = commenter_profiles[commenter_id]
    profile['total_comments'] += 1
    if prediction in profile:
        profile[prediction] += 1
    return profile

# === Core Comment Analysis Function ===
def analyze_comment_core(comment, commenter_id='anonymous'):
    vector = vectorizer.transform([comment])
    prediction = model.predict(vector)[0]
    probability = int(np.max(model.predict_proba(vector)) * 100)
    casing = analyze_casing(comment.strip())
    profile = update_profile(commenter_id, prediction.lower())
    return {
        'comment': comment,
        'sentiment': prediction,
        'confidence': f'{probability:.1f}%',
        'casing_tone': casing,
        'commenter_profile': profile
    }

# === API Endpoint ===
@app.route('/analyze-comment', methods=['POST'])
def analyze_comment():
    data = request.get_json()
    comment = data.get('comment')
    commenter_id = data.get('name', 'anonymous')

    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    try:
        result = analyze_comment_core(comment, commenter_id)
        return jsonify(result)
    except Exception as e:
        print(f"Error analyzing comment: {e}")
        return jsonify({'error': 'Failed to analyze comment', 'details': str(e)}), 500

# === Root Route for Testing ===
@app.route('/')
def index():
    return jsonify({"message": "Tonewise.ai Microservice is running 🧠"}), 200

# === Run the App ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)


