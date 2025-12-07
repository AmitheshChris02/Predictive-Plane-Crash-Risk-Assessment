from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from models import db, RiskAlert
import pickle
import json
import os
from datetime import datetime
import pandas as pd

module3_bp = Blueprint('module3', __name__)

# Load the model at startup
module3_model = None
model_loaded = False

def load_module3_model():
    global module3_model, model_loaded
    if model_loaded:
        return True
        
    try:
        # Try to fixed model first
        model_path = os.path.join(current_app.root_path, 'models/module3_best_model_fixed.pkl')
        
        # If the fixed model doesn't exist, try to original
        if not os.path.exists(model_path):
            model_path = os.path.join(current_app.root_path, 'models/module3_best_model.pkl')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            current_app.logger.error(f"Model file not found: {model_path}")
            return False
            
        # Try to load the model with different protocols
        try:
            with open(model_path, 'rb') as f:
                module3_model = pickle.load(f)
        except Exception as e:
            # Try with a different approach if the first attempt fails
            try:
                import cloudpickle
                with open(model_path, 'rb') as f:
                    module3_model = cloudpickle.load(f)
            except ImportError:
                # If cloudpickle is not available, try to handle the error gracefully
                current_app.logger.error(f"Failed to load model: {str(e)}")
                flash('Error loading the Takeoff Risk model. Please check the model file format.', 'danger')
                return False
            except Exception as e2:
                current_app.logger.error(f"Failed to load model with cloudpickle: {str(e2)}")
                flash('Error loading the Takeoff Risk model. Please check the model file format.', 'danger')
                return False
        
        model_loaded = True
        current_app.logger.info("Module 3 model loaded successfully")
        return True
        
    except Exception as e:
        current_app.logger.error(f"Unexpected error loading model: {str(e)}")
        flash('Error loading the Takeoff Risk model. Please check the model file format.', 'danger')
        return False

@module3_bp.route('/module3/takeoff_risk')
def takeoff_risk():
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('auth.login'))
    
    # Try to load the model when accessing the page
    if not model_loaded:
        if not load_module3_model():
            flash('Takeoff Risk model is not available. Please check the model file.', 'danger')
    
    return render_template('module3_takeoff.html')

@module3_bp.route('/module3/predict', methods=['POST'])
def predict_takeoff():
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('module3.takeoff_risk'))
    
    # Load model if not already loaded
    if not model_loaded:
        if not load_module3_model():
            flash('Takeoff Risk model is not available. Please check the model file.', 'danger')
            return redirect(url_for('module3.takeoff_risk'))
    
    try:
        # Get form data
        wind_speed_kts = float(request.form.get('wind_speed_kts'))
        visibility_km = float(request.form.get('visibility_km'))
        runway_length_m = float(request.form.get('runway_length_m'))
        temp_c = float(request.form.get('temp_c'))
        aircraft_age_yrs = float(request.form.get('aircraft_age_yrs'))
        last_service_days = float(request.form.get('last_service_days'))
        load_factor_pct = float(request.form.get('load_factor_pct'))
        elevation_ft = float(request.form.get('elevation_ft'))
        crosswind_comp = float(request.form.get('crosswind_comp'))
        precipitation = request.form.get('precipitation')
        runway_condition = request.form.get('runway_condition')
        
        # Create a DataFrame with the correct column names
        input_data = pd.DataFrame({
            'wind_speed_kts': [wind_speed_kts],
            'visibility_km': [visibility_km],
            'runway_length_m': [runway_length_m],
            'temp_c': [temp_c],
            'aircraft_age_yrs': [aircraft_age_yrs],
            'last_service_days': [last_service_days],
            'load_factor_pct': [load_factor_pct],
            'elevation_ft': [elevation_ft],
            'crosswind_comp': [crosswind_comp],
            'precipitation': [precipitation],
            'runway_condition': [runway_condition]
        })
        
        # Make prediction
        prediction = module3_model.predict(input_data)[0]
        
        # Determine color based on prediction
        color = "success"  # green for Low
        if prediction == "High":
            color = "danger"  # red for High
        elif prediction == "Medium":
            color = "warning"  # orange for Medium
        
        # Save prediction to database
        user_id = session['user_id']
        features_dict = {
            'wind_speed_kts': wind_speed_kts, 'visibility_km': visibility_km,
            'runway_length_m': runway_length_m, 'temp_c': temp_c,
            'aircraft_age_yrs': aircraft_age_yrs, 'last_service_days': last_service_days,
            'load_factor_pct': load_factor_pct, 'elevation_ft': elevation_ft,
            'crosswind_comp': crosswind_comp, 'precipitation': precipitation,
            'runway_condition': runway_condition
        }
        
        risk_alert = RiskAlert(
            user_id=user_id,
            module_type="TakeoffRisk",
            prediction_label=prediction,
            features=json.dumps(features_dict),
            created_at=datetime.now()
        )
        db.session.add(risk_alert)
        db.session.commit()
        
        return render_template(
            'module3_takeoff.html',
            prediction=prediction,
            color=color,
            features=features_dict
        )
    
    except Exception as e:
        current_app.logger.error(f'Error processing prediction: {str(e)}')
        flash(f'Error processing prediction: {str(e)}', 'danger')
        return redirect(url_for('module3.takeoff_risk'))

@module3_bp.route('/risk_alerts')
def risk_alerts():
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('auth.login'))
    
    user_id = session['user_id']
    filter_type = request.args.get('filter', 'all')
    
    if filter_type == 'blackbox':
        alerts = RiskAlert.query.filter_by(user_id=user_id, module_type='BlackBox').order_by(RiskAlert.created_at.desc()).all()
    elif filter_type == 'takeoff':
        alerts = RiskAlert.query.filter_by(user_id=user_id, module_type='TakeoffRisk').order_by(RiskAlert.created_at.desc()).all()
    else:
        alerts = RiskAlert.query.filter_by(user_id=user_id).order_by(RiskAlert.created_at.desc()).all()
    
    return render_template('risk_alerts.html', alerts=alerts, filter_type=filter_type)