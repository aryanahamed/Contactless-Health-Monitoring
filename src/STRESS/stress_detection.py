import os
import joblib
import numpy as np

# Constants
MODEL_ASSETS_BASE_DIR = os.path.join(os.path.dirname(__file__), 'trained_model_assets')
MODEL_PATH = os.path.join(MODEL_ASSETS_BASE_DIR, 'rf_model.joblib')
SCALER_PATH = os.path.join(MODEL_ASSETS_BASE_DIR, 'scaler.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_ASSETS_BASE_DIR, 'label_encoder.joblib')
FEATURE_COLUMNS = ["HR", "MEAN_RR", "SDNN", "RMSSD", "SDNN_RMSSD_RATIO"]

# Load the model
def load_stress_model_assets():
    rf_model = None
    scaler = None
    label_encoder = None
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        rf_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("Successfully loaded model")
    else:
        return None, None, None
    return rf_model, scaler, label_encoder


def predict_stress(hr, sdnn, rmssd, model, scaler, label_encoder):
    if not all([model, scaler, label_encoder]):
        print("Model files found")
        return None, None
        
    if hr is None or sdnn is None or rmssd is None:
        print("Vitals not found")
        return None, None

    # RR interval using HR
    mean_rr_val = None
    if hr > 0:
        mean_rr_val = (60.0 / hr) * 1000.0

    # SDNN and RMSSD ratio
    sdnn_rmssd_ratio_val = None
    if rmssd > 0:
        sdnn_rmssd_ratio_val = sdnn / rmssd

    current_features_values = np.array([[
        hr,
        mean_rr_val,
        sdnn,
        rmssd,
        sdnn_rmssd_ratio_val
    ]], dtype=np.float64)


    current_features_scaled = scaler.transform(current_features_values)
    predicted_encoded_label = model.predict(current_features_scaled)
    predicted_condition_name_array = label_encoder.inverse_transform(predicted_encoded_label)
    predicted_condition_name = predicted_condition_name_array[0]

    # Map the prediction to stress levels
    if predicted_condition_name == 'no stress':
        predicted_condition_name = 'Low Intensity'
    elif predicted_condition_name == 'interruption':
        predicted_condition_name = 'Medium Intensity'
    elif predicted_condition_name == 'time pressure':
        predicted_condition_name = 'High Intensity'
    
    predicted_probabilities = model.predict_proba(current_features_scaled)
    
    # confidence score for the prediction
    predicted_class_index = model.predict(current_features_scaled)[0]
    confidence_score = predicted_probabilities[0][predicted_class_index]
    
    # Format prediction with confidence score
    formatted_prediction = f"{predicted_condition_name} (Conf. {confidence_score:.2f})"
    
    # Use the following to show probabilities for each class
    class_probs = dict(zip(label_encoder.classes_, predicted_probabilities[0]))
    
    return formatted_prediction, class_probs, confidence_score


from collections import deque

# Store the predictions
prediction_history = deque(maxlen=15)

def predict_stress_with_smoothing(hr, sdnn, rmssd, model, scaler, label_encoder):
    raw_prediction, class_probs, confidence = predict_stress(hr, sdnn, rmssd, model, scaler, label_encoder)
    
    if raw_prediction is None:
        return None, None, None
    
    prediction_history.append((raw_prediction, class_probs, confidence))
    
    # majority voting
    if len(prediction_history) >= 15:
        predictions = [p[0].split(' (')[0] for p in prediction_history]  # Removing confidence part
        most_common = max(set(predictions), key=predictions.count)
        
        avg_confidence = np.mean([p[2] for p in prediction_history if p[0].split(' (')[0] == most_common])
        
        return f"{most_common} (Conf. {avg_confidence:.2f})", class_probs, avg_confidence
    
    return raw_prediction, class_probs, confidence