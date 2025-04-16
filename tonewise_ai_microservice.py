# tonewise_ai_microservice.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from collections import defaultdict

# === Load Models ===
model_path = 'tonewise_ai_model/tonewise_model.pkl'
vectorizer_path = 'tonewise_ai_model/tonewise_vectorizer.pkl'

# Load the pre-trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# === Initialize App ===
app = Flask(__name__)

# === Personal Commenter Tone Profiles ===
commenter_profiles = defaultdict(lambda: {'total_comments': 0, 'positive': 0, 'negative': 0, 'neutral': 0})

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

# === Personal Profile Updater ===
def update_profile(commenter_id, sentiment):
    profile = commenter_profiles[commenter_id]
    profile['total_comments'] += 1
    profile[sentiment] += 1
    return profile

# === Predict Function ===
def analyze_comment(comment, commenter_id='anonymous'):
    # Prepare text
    vector = vectorizer.transform([comment])
    prediction = model.predict(vector)[0]
    probability = np.max(model.predict_proba(vector)) * 100
    casing = analyze_casing(comment.strip())

    # Update profile
    profile = update_profile(commenter_id, prediction.lower())

    # Prepare response
    return {
        'comment': comment,
        'sentiment': prediction,
        'confidence': f"{probability:.1f}%",
        'casing_tone': casing,
        'commenter_profile': profile
    }

# === API Routes ===
@app.route('/analyze-comment', methods=['POST'])
def analyze_comment():
    data = request.get_json()

    # Extract fields from incoming request
    comment = data.get('comment')
    commenter_id = data.get('name', 'anonymous')  # Use 'name' field from the request

    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    # Process the comment
    result = analyze_comment(comment, commenter_id)
    return jsonify(result)

@app.route('/')
def index():
    return jsonify({"message": "Tonewise.ai Microservice is running 🔥"}), 200

# === Run App ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

