import os
import io
import base64
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from predict import load_drone_flight_model, predict_drone_flight


plt.switch_backend('Agg')

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once when the application starts
# loaded_model, loaded_scaler = load_drone_flight_model()
    
loaded_model, loaded_scaler = load_drone_flight_model(model_path='bilstm/best_drone_flight_model.pth', scaler_path='bilstm/drone_flight_scaler.pkl')


def generate_plots(filepath):
    """
    Generate multiple plots from the CSV file
    
    Returns:
        dict: Base64 encoded plot images
    """
    # Read the CSV
    df = pd.read_csv(filepath)
    
    # Invert specific columns
    df[['x', 'y', 'z', 'vx', 'vy', 'vz']] *= -1
    
    # Prepare plots dictionary
    plots = {}
    
    # Ensure we completely clear any existing plots
    plt.close('all')
    
    # 1. 3D Trajectory Plot
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot3D(df['x'], df['y'], df['z'], 'blue')
    ax.set_title('3D Trajectory ')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plots['trajectory_3d'] = plot_to_base64()
    plt.close()
    
    # 2. Velocity Plots (vx, vy, vz)
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(df['vx'], label='Velocity X')
    plt.title('Velocity X ')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    
    plt.subplot(132)
    plt.plot(df['vy'], label='Velocity Y', color='green')
    plt.title('Velocity Y ')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    
    plt.subplot(133)
    plt.plot(df['vz'], label='Velocity Z', color='red')
    plt.title('Velocity Z ')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    
    plt.tight_layout()
    plots['velocity_plots'] = plot_to_base64()
    plt.close()
    
    # [Rest of the function remains the same]
    # 3. Orientation Plots (roll, pitch, yaw)
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(df['roll'], label='Roll')
    plt.title('Roll')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    
    plt.subplot(132)
    plt.plot(df['pitch'], label='Pitch', color='green')
    plt.title('Pitch')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    
    plt.subplot(133)
    plt.plot(df['yaw'], label='Yaw', color='red')
    plt.title('Yaw')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    
    plt.tight_layout()
    plots['orientation_plots'] = plot_to_base64()
    plt.close()
    
    # 4. Angular Velocity Plots
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(df['rollspeed_p'], label='Roll Speed')
    plt.title('Roll Speed')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    
    plt.subplot(132)
    plt.plot(df['pitchspeed_q'], label='Pitch Speed', color='green')
    plt.title('Pitch Speed')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    
    plt.subplot(133)
    plt.plot(df['yawspeed_r'], label='Yaw Speed', color='red')
    plt.title('Yaw Speed')
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity')
    
    plt.tight_layout()
    plots['angular_velocity_plots'] = plot_to_base64()
    plt.close()
    
    return plots

def plot_to_base64():
    """
    Convert current matplotlib plot to base64 encoded image
    
    Returns:
        str: Base64 encoded plot image
    """
    # Create a bytes buffer to save the plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Close the buffer
    buf.close()
    
    return img_base64


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv(filepath):
    """
    Validate CSV file:
    1. Check if file is not empty
    2. Check if it has the correct headers
    """
    try:
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # Check if file is empty
        if df.empty:
            return False, "The CSV file is empty"
        
        # Define expected headers
        expected_headers = [
            'x', 'y', 'z', 
            'vx', 'vy', 'vz', 
            'roll', 'pitch', 'yaw', 
            'rollspeed_p', 'pitchspeed_q', 'yawspeed_r', 
            'Latitude', 'Longitude', 'Altitude'
        ]
        
        # Check headers
        if list(df.columns) != expected_headers:
            return False, f"Incorrect headers. Expected: {', '.join(expected_headers)}"
        
        return True, "Valid CSV file"
    
    except Exception as e:
        return False, f"Error processing file: {str(e)}"



@app.route('/', methods=['GET'])
def upload_form():
    """Render the upload form"""
    return render_template('upload.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle file upload and prediction"""
#     # Check if file was uploaded
#     if 'file' not in request.files:
#         return jsonify({
#             'error': True, 
#             'message': 'No file uploaded'
#         }), 400
    
#     file = request.files['file']
    
#     # Check if filename is empty
#     if file.filename == '':
#         return jsonify({
#             'error': True, 
#             'message': 'No selected file'
#         }), 400
    
#     # Check if file is allowed
#     if file and allowed_file(file.filename):
#         # Secure the filename and save
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             # Validate CSV
#             is_valid, validation_message = validate_csv(filepath)
            
#             if not is_valid:
#                 # Remove the uploaded file
#                 os.remove(filepath)
#                 return jsonify({
#                     'error': True, 
#                     'message': validation_message
#                 }), 400
            
#             # Predict drone flight
#             prediction = predict_drone_flight(
#                 loaded_model, 
#                 loaded_scaler, 
#                 filepath
#             )
#             try:
#                 plots = generate_plots(filepath)
                
#                 return jsonify({
#                     'error': False,
#                     'prediction': prediction,
#                     'plots': plots
#                 })
#             except Exception as e:
#                 # Remove the uploaded file in case of any error
#                 if os.path.exists(filepath):
#                     os.remove(filepath)
#                 return jsonify({
#                     'error': True, 
#                     'message': f'Plot generation error: {str(e)}'
#                 }), 500
#             os.remove(filepath)
#             # Remove the uploaded file after prediction
            
#             return jsonify({
#                 'error': False,
#                 'prediction': prediction
#             })
            
        
#         except Exception as e:
#             # Remove the uploaded file in case of any error
#             if os.path.exists(filepath):
#                 os.remove(filepath)
#             return jsonify({
#                 'error': True, 
#                 'message': f'Prediction error: {str(e)}'
#             }), 500
    
#     return jsonify({
#         'error': True, 
#         'message': 'Invalid file type'
#     }), 400
@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({
            'error': True, 
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    model_name = request.form.get('model')
    print(model_name)
    
    # Validate model selection
    if not model_name or model_name not in ['BiLSTM', 'Transformer', 'TemporalCNN', 'HybridCNNTransformer']:
        return jsonify({
            'error': True, 
            'message': 'Invalid model selected'
        }), 400
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'error': True, 
            'message': 'No selected file'
        }), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Validate CSV
            is_valid, validation_message = validate_csv(filepath)
            
            if not is_valid:
                # Remove the uploaded file
                os.remove(filepath)
                return jsonify({
                    'error': True, 
                    'message': validation_message
                }), 400
            
            # Dynamically load model based on selection
            if model_name == 'BiLSTM':
                print("model type: ", model_name)
                loaded_model, loaded_scaler = load_drone_flight_model(
                    model_path='best_drone_flight_model.pth', 
                    scaler_path='drone_flight_scaler.pkl'
                )
            elif model_name == 'Transformer':
                loaded_model, loaded_scaler = load_drone_flight_model(
                    model_path='transformer/best_drone_flight_model.pth', 
                    scaler_path='transformer/drone_flight_scaler.pkl'
                )
            elif model_name == 'TemporalCNN':
                loaded_model, loaded_scaler = load_drone_flight_model(
                    model_path='TemporalCNN/best_drone_flight_model.pth', 
                    scaler_path='TemporalCNN/drone_flight_scaler.pkl'
                )
            elif model_name == 'HybridCNNTransformer':
                loaded_model, loaded_scaler = load_drone_flight_model(
                    model_path='HybridCNNTransformer/best_drone_flight_model.pth', 
                    scaler_path='HybridCNNTransformer/drone_flight_scaler.pkl'
                )
            
            # Predict drone flight
            prediction = predict_drone_flight(
                loaded_model, 
                loaded_scaler, 
                filepath
            )
            
            try:
                plots = generate_plots(filepath)
                
                os.remove(filepath)  # Remove uploaded file after processing
                
                return jsonify({
                    'error': False,
                    'prediction': prediction,
                    'plots': plots
                })
            except Exception as e:
                # Remove the uploaded file in case of any error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    'error': True, 
                    'message': f'Plot generation error: {str(e)}'
                }), 500
        
        except Exception as e:
            # Remove the uploaded file in case of any error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': True, 
                'message': f'Prediction error: {str(e)}'
            }), 500
    
    return jsonify({
        'error': True, 
        'message': 'Invalid file type'
    }), 400
if __name__ == '__main__':
    app.run(debug=True)