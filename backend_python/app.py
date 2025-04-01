import numpy as np
from flask import Flask, render_template, request, jsonify
import pywt
from tensorflow.keras.models import load_model
from joblib import load
import wfdb
import csv
from io import StringIO
import os
import tempfile
import logging
from scipy.signal import resample

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
TARGET_FS = 360  # Target sampling frequency for the model
WINDOW_DURATION = 0.5  # Window duration in seconds

# Load pre-trained model and scaler
model = load_model('ecg_model.h5')
scaler = load('scaler.joblib')

label_mapping = {
    0: 'Normal beat',
    1: 'Left bundle branch block beat',
    2: 'Right bundle branch block beat',
    3: 'Atrial premature beat',
    4: 'Premature ventricular contraction'
}

def safe_entropy_calculation(c):
    """Safe entropy calculation with input validation"""
    c = np.abs(c) + 1e-12
    c_normalized = c / np.sum(c)
    c_normalized = np.clip(c_normalized, 1e-12, 1.0)
    return -np.sum(c_normalized * np.log2(c_normalized))

def process_csv(csv_file, fs):
    """Process CSV with resampling to target frequency"""
    try:
        content = csv_file.read().decode('utf-8')
        reader = csv.reader(StringIO(content))
        
        # Validate header
        header = next(reader, None)
        if not header or 'hart' not in header[0].lower():
            raise ValueError("CSV header must contain 'hart' column")

        # Read signals
        signals = []
        for row_num, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                signals.append(float(row[0].strip()))
            except (IndexError, ValueError) as e:
                logger.warning(f"Skipping row {row_num}: {str(e)}")
                continue

        if len(signals) < 10:  # Minimum samples after resampling
            raise ValueError(f"Signal too short ({len(signals)} samples).")

        # Resample to target frequency
        if fs != TARGET_FS:
            num_samples = int(len(signals) * TARGET_FS / fs)
            signals = resample(signals, num_samples)
            fs = TARGET_FS

        # Create WFDB-compatible record with resampled signal
        record_name = "ecg-recording"
        p_signal = np.array(signals, dtype=np.float32).reshape(-1, 1)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            wfdb.wrsamp(record_name,
                       fs=fs,
                       units=['mV'],
                       sig_name=['MLII'],
                       p_signal=p_signal,
                       fmt=['16'],
                       write_dir=tmp_dir)
            
            return wfdb.rdrecord(os.path.join(tmp_dir, record_name))
    
    except Exception as e:
        logger.error(f"CSV processing failed: {str(e)}")
        raise

def extract_wavelet_features(segment):
    """Extract wavelet features from a segment"""
    if len(segment) < 16:
        raise ValueError("Segment too short for wavelet transform")
    
    try:
        coeffs = pywt.wavedec(segment, 'db4', level=4)
        std_coeffs = [np.std(c) for c in coeffs]
        energy_coeffs = [np.sum(c**2) for c in coeffs]
        entropy_coeffs = [safe_entropy_calculation(c) for c in coeffs]
        return np.concatenate([std_coeffs, energy_coeffs, entropy_coeffs]).reshape(1, -1)
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            return jsonify({'error': 'Empty file submitted'}), 400

        fs = int(request.form.get('fs', 360))
        if not 100 <= fs <= 1000:
            return jsonify({'error': 'Invalid sampling frequency (100-1000 Hz)'}), 400

        # Process and resample CSV
        record = process_csv(csv_file, fs)
        signal = record.p_signal[:, 0]
        fs = record.fs  # Get updated fs after resampling

        # Calculate window size based on target duration
        window_size = int(TARGET_FS * WINDOW_DURATION)
        step_size = window_size  # Non-overlapping windows

        if len(signal) < window_size:
            return jsonify({'error': f'Signal too short ({len(signal)} samples)'}), 400

        results = []
        valid_segments = 0
        min_confidence = 0.6

        for i in range(0, len(signal) - window_size + 1, step_size):
           segment = signal[i:i+window_size]
        
           if np.std(segment) < 0.01:
            continue
            
           try:
            # Normalize and reshape for CNN
                normalized = scaler.transform(segment.reshape(1, -1))
                cnn_input = normalized.reshape(1, window_size, 1)  # (batch, timesteps, channels)
            
                prediction = model.predict(cnn_input, verbose=0)
                confidence = float(np.max(prediction))
                
                if confidence >= min_confidence:
                    results.append({
                        'start': i / TARGET_FS,  # Convert to seconds
                        'end': (i + window_size) / TARGET_FS,
                        'prediction': label_mapping.get(np.argmax(prediction), 'Unknown'),
                        'confidence': confidence
                    })
                    valid_segments += 1
                    
           except Exception as e:
                logger.debug(f"Skipping segment {i}: {str(e)}")
                continue

        if not valid_segments:
            return jsonify({'error': 'No confident predictions found'}), 400

        return jsonify({
            'analysis': results,
            'summary': {
                'total_segments': valid_segments,
                'average_confidence': np.mean([r['confidence'] for r in results])
            }
        })

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Processing error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)