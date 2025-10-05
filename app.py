import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json 
from flask import Flask, request, jsonify
from functools import wraps
import time
import re 
from pymongo import MongoClient # Import PyMongo for MongoDB interaction
from pymongo.errors import ConnectionFailure # Import specific error: Changed from ConnectionError

# --- CONFIGURATION (UPDATE THESE PATHS) ---
# IMPORTANT: These files MUST be present in the same directory as this script.
MODEL_PATH = 'sentiment_rnn_model.keras'      
TOKENIZER_PATH = 'tokenizer.json'             
MAX_SEQUENCE_LENGTH = 100 

# --- MONGODB CONFIGURATION ---
# !!! IMPORTANT: REPLACE THIS URI with your actual MongoDB connection string (e.g., from Atlas or local server) !!!
MONGO_URI = "mongodb+srv://rawatvr44_db_user:FK2J4WYtOIXDeVue@erp.peib4ka.mongodb.net/?retryWrites=true&w=majority&appName=erp" # Placeholder for remote or local URI
DB_NAME = "brand_reviews_db"
# COLLECTION_NAME is now dynamic, but we keep a global connection reference
# COLLECTION_NAME = "reviews" 

# --- 1. MODEL, TOKENIZER, AND MONGO LOADING ---

loaded_model = None
loaded_tokenizer = None
mongo_client = None
mongo_db = None # Reference to the database object

def load_resources():
    """Loads the Keras model, tokenizer, and initializes MongoDB globally."""
    global loaded_model
    global loaded_tokenizer
    global mongo_client
    global mongo_db
    
    start_time = time.time()
    try:
        # --- Load Keras Resources ---
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as handle:
            json_string = handle.read()
            tokenizer_config = json.loads(json_string)
            loaded_tokenizer = tokenizer_from_json(tokenizer_config)
            
        # --- Initialize MongoDB Client ---
        try:
            mongo_client = MongoClient(MONGO_URI)
            # The ismaster command is a lightweight way to check a connection
            mongo_client.admin.command('ismaster')
            mongo_db = mongo_client[DB_NAME] # Set the global database reference
            print(f"✅ MongoDB connected to database: {DB_NAME}")
        except ConnectionFailure as ce: # Changed exception class to ConnectionFailure
            print(f"CRITICAL ERROR: MongoDB connection failed. Please check MONGO_URI and server status. Error: {ce}")
            raise RuntimeError(f"MongoDB connection failed: {ce}")
            
        print(f"✅ Resources loaded successfully in {time.time() - start_time:.2f}s.")
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Resource file not found: {e}")
        raise RuntimeError(f"Failed to load resources: {e}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load resources: {e}")
        # Ensure we close the Mongo connection on failure if it was opened
        if mongo_client:
             mongo_client.close()
        raise RuntimeError(f"Failed to load resources: {e}")


# --- 2. TEXT PREPROCESSING FUNCTIONS ---

def _preprocess_text(text):
    """
    NOTE: YOU MUST REPLACE THIS WITH YOUR EXACT PREPROCESSTEXT LOGIC!
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Simple placeholder cleaning: keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text) 
    
    return text

def _get_star_sentiment(star_rating):
    """Maps 1-5 star rating to Positive, Neutral, or Negative."""
    if star_rating >= 4:
        return "positive" # Return lowercase sentiment
    elif star_rating == 3:
        return "neutral" # Return lowercase sentiment
    else: # 1 or 2 stars
        return "negative" # Return lowercase sentiment

def predict_sentiment_from_comment(comment):
    """Processes a raw comment and returns the predicted sentiment (without probabilities)."""
    
    # 1. Clean the text using the assumed training function
    cleaned_comment = _preprocess_text(comment)

    # 2. Tokenization: Convert text to numerical indices
    new_sequence = loaded_tokenizer.texts_to_sequences([cleaned_comment])

    # 3. Padding: Ensure the sequence has the correct length
    new_padded_sequence = pad_sequences(
        new_sequence, 
        maxlen=MAX_SEQUENCE_LENGTH, 
        truncating='post' 
    )

    # 4. Prediction
    # Note: We still calculate raw_prediction, but only use it for the final star rating
    raw_prediction = loaded_model.predict(new_padded_sequence, verbose=0)

    # 5. Extract star rating
    predicted_label = np.argmax(raw_prediction, axis=1)[0]
    predicted_star_rating = int(predicted_label + 1)
    
    # 6. Map to 3-class sentiment
    sentiment = _get_star_sentiment(predicted_star_rating)
    
    # Return structure for the response/DB record
    return {
        "rating": predicted_star_rating, # Use "rating" key
        "sentiment": sentiment,          # Use "sentiment" key (lowercase)
    }


# --- 3. FLASK APP SETUP ---

app = Flask(__name__)

# Run resource loading when the application starts
with app.app_context():
    load_resources()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive comment, run prediction, and save to DB."""
    
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON format."}), 400

    required_fields = ['brand', 'comment']
    for field in required_fields:
        if field not in data or not data[field]:
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    brand = data['brand']
    comment = data['comment']
    
    try:
        prediction_results = predict_sentiment_from_comment(comment)
        
        # --- DYNAMIC COLLECTION SELECTION ---
        # Collection name = brand name (e.g., "Apple", "Samsung")
        # Ensure collection name is safe (alphanumeric/simple conversion)
        collection_name = brand.replace(" ", "_").strip().lower() 
        mongo_collection = mongo_db[collection_name]
        
        # --- STEP 4: PREPARE AND SAVE DATA TO MONGODB ---
        db_record = {
            "brand": brand,             # Kept for reference inside the document
            "body": comment,            # Changed key from 'comment' to 'body'
            "sentiment": prediction_results['sentiment'], 
            # Note: MongoDB stores integer rating. Example decimal 2.7 would require more complex modeling.
            "rating": prediction_results['rating'],          
        }
        
        # Insert the record into the dynamically selected MongoDB collection
        insert_result = mongo_collection.insert_one(db_record)
        print(f"MongoDB insert successful into '{collection_name}'. Document ID: {insert_result.inserted_id}")
        
        # Combine input and prediction into the response
        response = {
            "input": {
                "brand": brand,
                "comment": comment
            },
            "prediction": {
                "sentiment": prediction_results['sentiment'],
                "rating": prediction_results['rating'] # Integer rating 1-5
            },
            "database_id": str(insert_result.inserted_id) 
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": "Internal server error during prediction."}), 500

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    # Also check if MongoDB client is available
    mongo_status = "Connected" if mongo_client else "Disconnected"
    return jsonify({"API Status": "Running", "MongoDB Status": mongo_status})

if __name__ == '__main__':
    # You can run this file directly using: python api_server.py
    # To run on a typical hosted environment (like Google Cloud Run), you might use Gunicorn.
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
