# =============================================================================
# Bean Leaf Disease Predictor — Flask Web Application
# =============================================================================
# WHAT THIS FILE DOES:
#   This is the "server" (backend) of the web app.
#   Flask is a lightweight Python web framework.
#   When a user uploads an image on the website, this file:
#     1. Receives the uploaded image
#     2. Preprocesses it (resize, normalise)
#     3. Loads the trained ResNet152V2 model
#     4. Runs a prediction
#     5. Sends the result back to the webpage
# =============================================================================

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename   # safely handle uploaded filenames

# ── App Configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)

# Folder where uploaded images are temporarily saved
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # create folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Only allow image file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model settings — must match what was used during training
IMG_HEIGHT  = 224
IMG_WIDTH   = 224
MODEL_PATH  = 'bean_leaf_resnet152v2.h5'

# Class names in the same order as training (alphabetical by default)
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Disease information shown to the user after prediction
# Based on the research paper's findings
DISEASE_INFO = {
    'angular_leaf_spot': {
        'status'     : 'DISEASED',
        'description': 'Angular Leaf Spot is caused by the bacterium Pseudomonas syringae. '
                       'It appears as water-soaked, angular lesions on leaves that turn brown '
                       'and necrotic. It spreads through rain, wind, and infected seeds.',
        'treatment'  : 'Remove infected leaves. Apply copper-based bactericides. '
                       'Avoid overhead irrigation. Use certified disease-free seeds.',
        'severity'   : 'Moderate to High',
        'color'      : '#e74c3c'
    },
    'bean_rust': {
        'status'     : 'DISEASED',
        'description': 'Bean Rust is caused by the fungus Uromyces appendiculatus. '
                       'It produces reddish-brown pustules (rust-coloured spots) on '
                       'the undersides of leaves. It thrives in humid, warm conditions.',
        'treatment'  : 'Apply fungicides (mancozeb or triazole-based). '
                       'Improve air circulation. Avoid wetting leaves. '
                       'Rotate crops annually.',
        'severity'   : 'High',
        'color'      : '#e67e22'
    },
    'healthy': {
        'status'     : 'HEALTHY',
        'description': 'The bean leaf appears healthy with no visible signs of disease. '
                       'Leaf colour, texture, and structure are normal. '
                       'Continue regular monitoring and good agricultural practices.',
        'treatment'  : 'No treatment needed. Maintain regular watering, '
                       'adequate sunlight, and balanced fertilisation.',
        'severity'   : 'None',
        'color'      : '#27ae60'
    }
}

# ── Load Model Once at Startup ────────────────────────────────────────────────
# We load the model when the app starts so every prediction is fast.
# Loading a model takes ~5 seconds; we don't want to do it on every request.
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")


# ── Helper Functions ──────────────────────────────────────────────────────────

def allowed_file(filename):
    """Check if the uploaded file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Prepare an image for model prediction.

    Steps:
      1. Load image from disk and resize to 224×224 (model's input size)
      2. Convert to NumPy array  →  shape: (224, 224, 3)
      3. Normalise pixels from [0, 255] → [0.0, 1.0]
      4. Add batch dimension   →  shape: (1, 224, 224, 3)

    The model always expects a BATCH of images, even for a single image.
    """
    img       = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0          # normalise
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    return img_array


def predict_disease(image_path):
    """
    Run the trained ResNet152V2 model on the image and return results.

    Returns a dict with:
      - predicted_class : string name of the predicted disease/healthy
      - confidence      : percentage confidence of the top prediction
      - all_probs       : probability for each of the 3 classes
      - info            : disease details (description, treatment, severity)
    """
    img_array    = preprocess_image(image_path)
    predictions  = model.predict(img_array, verbose=0)   # shape: (1, 3)
    predicted_idx = np.argmax(predictions[0])             # index of highest prob
    confidence    = float(predictions[0][predicted_idx]) * 100

    predicted_class = CLASS_NAMES[predicted_idx]

    # Build probability dict for all 3 classes
    all_probs = {
        CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        'predicted_class': predicted_class,
        'confidence'     : round(confidence, 2),
        'all_probs'      : all_probs,
        'info'           : DISEASE_INFO[predicted_class]
    }


# ── Routes (URL Endpoints) ────────────────────────────────────────────────────
# A "route" tells Flask what to do when a user visits a specific URL.

@app.route('/')
def index():
    """
    Home page route.
    When user visits http://localhost:5000, Flask renders index.html.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction route — handles the image upload and returns results.

    This route is called when the user clicks "Predict" on the webpage.
    It only accepts POST requests (form submissions with file data).

    Flow:
      1. Check a file was actually uploaded
      2. Validate the file type (jpg/png only)
      3. Save the file temporarily to static/uploads/
      4. Run the model prediction
      5. Return the result as JSON to the webpage
    """
    # Check if a file was included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Please select an image.'}), 400

    file = request.files['file']

    # Check the user actually selected a file (not empty)
    if file.filename == '':
        return jsonify({'error': 'No file selected. Please choose an image.'}), 400

    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG or PNG only.'}), 400

    # secure_filename removes dangerous characters from the filename
    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Run prediction
    result = predict_disease(save_path)

    # Add the image URL so the webpage can display it
    result['image_url'] = f'/static/uploads/{filename}'

    return jsonify(result)


# ── Start the App ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # debug=True means the server restarts automatically when you edit code
    # and shows detailed error messages in the browser
    print("\n" + "="*55)
    print("  Bean Leaf Disease Predictor is running!")
    print("  Open your browser and go to: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True)
