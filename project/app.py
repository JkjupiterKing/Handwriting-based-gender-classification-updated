import os
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import time
import json
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import cv2
import base64
import io
from PIL import Image
import mysql.connector
from datetime import datetime
from flask_mail import Mail, Message
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.secret_key = 'your_secret_key'



# Configure API key
genai.configure(api_key="AIzaSyDgxgXjIJ3Wzr53eF_TcAm_honkaMInsO4")

# MySQL Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'handwriting_gender_classification'
}


# app.py (Around line 30)

# CRITICAL FIX: Use the 'models/' prefix AND the latest public models.
# gemini-1.5-flash is the recommended multimodal model.
PREFERRED_MODEL = "models/gemini-1.5-flash" 
FALLBACK_MODEL = "models/gemini-2.5-flash"  # Using 2.5-flash as a fallback model


# Load CNN model
CNN_MODEL = load_model(r'D:\ML Projects\Handwriting-based-gender-classification\project\cnn_gender_model.h5')

# Initialize database connection
def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            cnn_gender VARCHAR(20),
            cnn_confidence FLOAT,
            gemini_gender VARCHAR(20),
            handedness VARCHAR(20),
            age_group VARCHAR(20),
            style_traits JSON,
            match_status VARCHAR(100)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS contact_messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) NOT NULL,
            subject VARCHAR(200) NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        print("✅ Database initialized successfully")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Database initialization error: {err}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Initialize database on app start
init_db()

# Database helper function
def save_to_database(data):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
        INSERT INTO analysis_results 
        (cnn_gender, cnn_confidence, gemini_gender, handedness, age_group, style_traits, match_status) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        traits_json = json.dumps(data.get('style_traits', {}))
        
        values = (
            data.get('cnn_gender', 'Unknown'),
            data.get('cnn_confidence', 0.0),
            data.get('gemini_gender', 'Unknown'),
            data.get('handedness', 'Unknown'),
            data.get('age_group', 'Unknown'),
            traits_json,
            data.get('match', '')
        )
        
        cursor.execute(query, values)
        conn.commit()
        print(f"💾 Saved to database. Row ID: {cursor.lastrowid}")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Database error ({err.errno}): {err.msg}")
        return False
    except Exception as e:
        print(f"❌ Unexpected database error: {str(e)}")
        return False
    finally:  # Add this finally block
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# Then define the save_contact_message function here
def save_contact_message(name, email, subject, message):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
        INSERT INTO contact_messages 
        (name, email, subject, message) 
        VALUES (%s, %s, %s, %s)
        """
        
        values = (name, email, subject, message)
        cursor.execute(query, values)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving contact message: {str(e)}")
        return False
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()



# Gemini Predictor Functions
def get_age_range(age_group):
    """Map age group to a numerical range."""
    age_map = {
        "child": "0-12",
        "teenager": "13-19",
        "adult": "20-59",
        "senior": "60+"
    }
    return age_map.get(age_group.lower(), "Unknown")

def clean_json_response(text):
    """Remove markdown code fences and extra formatting."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
    if text.startswith("json"):
        text = text[len("json"):].strip()
    return text.strip()

def predict_handwriting_features(image_bytes):
    """Predict handwriting attributes from an image."""
    model_name = PREFERRED_MODEL
    raw_text = ""
    for attempt in range(2):
        try:
            print(f"Using model: {model_name}")
            
           
            model = genai.GenerativeModel(model_name)
            

            prompt = (
                "You are a handwriting analysis AI. "
                "From the given handwriting sample, predict:\n"
                "1. Gender of the writer\n"
                "2. Whether they are left-handed or right-handed\n"
                "3. Approximate age group (child, teenager, adult, senior)\n"
                "4. Notable style traits.\n"
                "Respond ONLY with a valid JSON object.\n"
                "Keys: gender, handedness, age_group, style_traits."
            )

            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
            )

            raw_text = response.text
            print(f"🔍 Raw Model Output:\n{raw_text}\n")

            clean_text = clean_json_response(raw_text)
            result = json.loads(clean_text)
            result.setdefault("gender", "Unknown")
            result.setdefault("handedness", "Unknown")
            result.setdefault("age_group", "Unknown")
            result.setdefault("style_traits", {})

            return result

        except ResourceExhausted:
            print(f"⚠️ Quota exceeded for {model_name}. Switching to fallback...")
            model_name = FALLBACK_MODEL
            time.sleep(2)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print("🔹 Raw text was:", raw_text)
            if attempt == 0:
                model_name = FALLBACK_MODEL
                time.sleep(2)
                continue
            return {
                "gender": "Unknown",
                "handedness": "Unknown",
                "age_group": "Unknown",
                "style_traits": {}
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            if attempt == 0:
                model_name = FALLBACK_MODEL
                time.sleep(2)
                continue
            return {
                "gender": "Unknown",
                "handedness": "Unknown",
                "age_group": "Unknown",
                "style_traits": {}
            }

def predict_with_cnn(image_bytes):
    """Predict gender using the CNN model."""
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize to 64x64 (model's expected input)
        img = img.resize((64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        prediction = CNN_MODEL.predict(img_array)
        confidence = float(prediction[0][0])
        
        # Interpret results
        if confidence > 0.5:
            gender = "Male"
            confidence_percent = confidence * 100
        else:
            gender = "Female"
            confidence_percent = (1 - confidence) * 100
            
        return gender, round(confidence_percent, 2)
    
    except Exception as e:
        print(f"CNN Prediction Error: {str(e)}")
        return "Unknown", 0.0

# Webcam Functions
def capture_from_webcam():
    """Capture image from webcam and return as bytes"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    return None

def resize_image(image_bytes, max_size=1024):
    """Resize image to reduce file size while maintaining aspect ratio"""
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=85)
    return output.getvalue()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/cnn-demo')
def cnn_demo():
    return render_template('cnn_demo.html')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        image_bytes = capture_from_webcam()
        if image_bytes:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            return jsonify({
                "success": True,
                "image": image_b64,
                "message": "Image captured successfully"
            })
        return jsonify({
            "success": False,
            "message": "Failed to capture image"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict_route():
    image_bytes = None
    
    # Check if webcam image is sent
    if 'webcam_image' in request.form:
        image_data = request.form['webcam_image']
        if image_data.startswith('data:image'):
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
    
    # Check if file is uploaded
    elif 'image' in request.files:
        image_file = request.files['image']
        try:
            image_bytes = image_file.read()
            image_bytes = resize_image(image_bytes)
        except Exception as e:
            return jsonify({
                "gender": "Unknown",
                "handedness": "Unknown",
                "age_group": "Unknown",
                "style_traits": "Unknown",
                "error": f"Invalid image file: {str(e)}"
            }), 400
    
    if not image_bytes:
        return jsonify({
            "gender": "Unknown",
            "handedness": "Unknown",
            "age_group": "Unknown",
            "style_traits": "Unknown",
            "error": "No image provided"
        }), 400

    try:
        # Get predictions from the model
        prediction = predict_handwriting_features(image_bytes)
        
        # Format style_traits
        style_traits = prediction.get("style_traits", {})
        if isinstance(style_traits, dict):
            formatted_traits = ", ".join(style_traits.values())
        elif isinstance(style_traits, list):
            formatted_traits = ", ".join(style_traits)
        elif isinstance(style_traits, str):
            formatted_traits = style_traits
        else:
            formatted_traits = "Unknown"
        
        prediction["style_traits"] = formatted_traits.capitalize() + "."

        # Convert age group to age range
        if 'age_group' in prediction:
            prediction['age_group'] = get_age_range(prediction['age_group'])

        # Save to database
        save_to_database(prediction)
            
        return jsonify(prediction)
    except Exception as e:
        return jsonify({
            "gender": "Unknown",
            "handedness": "Unknown",
            "age_group": "Unknown",
            "style_traits": "Unknown",
            "error": str(e)
        }), 500

@app.route('/predict-cnn', methods=['POST'])
def predict_cnn():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # CNN Prediction
        cnn_gender, cnn_confidence = predict_with_cnn(image_bytes)
        
        # Gemini Prediction
        gemini_result = predict_handwriting_features(image_bytes)
        gemini_gender = gemini_result.get('gender', 'Unknown')
        
        # Compare results
        if cnn_gender != "Unknown" and gemini_gender != "Unknown":
            if cnn_gender.lower() == gemini_gender.lower():
                match = f"Both models agree: {cnn_gender}"
            else:
                match = f"Models disagree: CNN predicts {cnn_gender}, Gemini predicts {gemini_gender}"
        else:
            match = "Could not compare results"
        
        # Format additional insights
        insights = []
        if gemini_result.get('handedness'):
            insights.append(f"Handedness: {gemini_result['handedness']}")
        if gemini_result.get('age_group'):
            age_range = get_age_range(gemini_result['age_group'])
            insights.append(f"Age Range: {age_range}")
        if gemini_result.get('style_traits'):
            insights.append(f"Traits: {gemini_result['style_traits']}")
        
        additional_insights = ". ".join(insights)
        
        # Prepare response
        result = {
            "cnn_gender": cnn_gender,
            "cnn_confidence": cnn_confidence,
            "gemini_gender": gemini_gender,
            "match": match,
            "additional_insights": additional_insights
        }
        
        # Save to database
        save_to_database({
            "cnn_gender": cnn_gender,
            "cnn_confidence": cnn_confidence,
            "gemini_gender": gemini_gender,
            "match": match,
            **gemini_result
        })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "cnn_gender": "Unknown",
            "cnn_confidence": 0,
            "gemini_gender": "Unknown",
            "match": "Error occurred",
            "additional_insights": ""
        }), 500

@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        if not all([name, email, subject, message]):
            flash('All fields are required!', 'error')
            return redirect(url_for('contact'))
        
        # Save to database
        save_contact_message(name, email, subject, message)
        
        # Send email
        msg = Message(
            subject=f"New Contact Message: {subject}",
            sender=app.config['MAIL_USERNAME'],
            recipients=[app.config['MAIL_USERNAME']],
            body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        )
        mail.send(msg)
        
        flash('Thank you for your message! I will get back to you soon.', 'success')
        
    except Exception as e:
        flash('An error occurred. Please try again.', 'error')
        print(f"Error sending message: {str(e)}")
    
    return redirect(url_for('contact'))

if __name__ == "__main__":   
    app.run(debug=True, port=5001)