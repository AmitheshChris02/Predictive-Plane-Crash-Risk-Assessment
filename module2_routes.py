from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from models import db, RiskAlert
import json
import os
from datetime import datetime

module2_bp = Blueprint('module2', __name__)

@module2_bp.route('/module2/blackbox')
def blackbox():
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('auth.login'))
    
    return render_template('module2_blackbox.html')

@module2_bp.route('/module2/predict', methods=['POST'])
def predict_blackbox():
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('module2.blackbox'))
    
    try:
        # Get form data
        cycle = float(request.form.get('cycle'))
        altitude_ft = float(request.form.get('altitude_ft'))
        airspeed_kts = float(request.form.get('airspeed_kts'))
        engine_rpm = float(request.form.get('engine_rpm'))
        egt_temp = float(request.form.get('egt_temp'))
        oil_pressure = float(request.form.get('oil_pressure'))
        hydraulic_pressure = float(request.form.get('hydraulic_pressure'))
        vibration_level = float(request.form.get('vibration_level'))
        fuel_flow = float(request.form.get('fuel_flow'))
        generator_voltage = float(request.form.get('generator_voltage'))
        sensor_health = float(request.form.get('sensor_health'))
        
        # Rule-based prediction system
        prediction = "Normal"
        color = "success"
        
        # Calculate risk scores
        risk_score = 0
        
        # Engine risk factors
        if engine_rpm > 5000:
            risk_score += 3
        elif engine_rpm > 4000:
            risk_score += 2
        elif engine_rpm > 3000:
            risk_score += 1
            
        if egt_temp > 850:
            risk_score += 3
        elif egt_temp > 750:
            risk_score += 2
        elif egt_temp > 650:
            risk_score += 1
            
        if fuel_flow > 400:
            risk_score += 2
        elif fuel_flow > 250:
            risk_score += 1
        
        # Hydraulic risk factors
        if hydraulic_pressure < 1500:
            risk_score += 3
        elif hydraulic_pressure < 2000:
            risk_score += 2
        elif hydraulic_pressure < 2500:
            risk_score += 1
        
        # Electrical risk factors
        if generator_voltage < 18:
            risk_score += 3
        elif generator_voltage < 22:
            risk_score += 2
        elif generator_voltage < 24:
            risk_score += 1
        
        # Sensor risk factors
        if sensor_health < 0.5:
            risk_score += 3
        elif sensor_health < 0.7:
            risk_score += 2
        elif sensor_health < 0.85:
            risk_score += 1
        
        # Vibration affects all systems
        if vibration_level > 3.0:
            risk_score += 2
        elif vibration_level > 2.0:
            risk_score += 1
        
        # Make prediction based on risk score
        if risk_score >= 8:
            if engine_rpm > 5000 or egt_temp > 850:
                prediction = "Engine_Failure_Risk"
            elif hydraulic_pressure < 1500:
                prediction = "Hydraulic_Failure_Risk"
            elif generator_voltage < 18:
                prediction = "Electrical_Anomaly"
            else:
                prediction = "Engine_Failure_Risk"
            color = "danger"
        elif risk_score >= 5:
            if sensor_health < 0.7:
                prediction = "Sensor_Malfunction"
            else:
                prediction = "Engine_Failure_Risk"
            color = "warning"
        elif risk_score >= 2:
            prediction = "Normal"
            color = "warning"
        else:
            prediction = "Normal"
            color = "success"
        
        # Save prediction to database
        user_id = session['user_id']
        features_dict = {
            'cycle': cycle, 'altitude_ft': altitude_ft, 'airspeed_kts': airspeed_kts,
            'engine_rpm': engine_rpm, 'egt_temp': egt_temp, 'oil_pressure': oil_pressure,
            'hydraulic_pressure': hydraulic_pressure, 'vibration_level': vibration_level,
            'fuel_flow': fuel_flow, 'generator_voltage': generator_voltage, 'sensor_health': sensor_health
        }
        
        risk_alert = RiskAlert(
            user_id=user_id,
            module_type="BlackBox",
            prediction_label=prediction,
            features=json.dumps(features_dict),
            created_at=datetime.now()
        )
        db.session.add(risk_alert)
        db.session.commit()
        
        return render_template(
            'module2_blackbox.html',
            prediction=prediction,
            color=color,
            features=features_dict
        )
    
    except Exception as e:
        current_app.logger.error(f'Error processing prediction: {str(e)}')
        flash(f'Error processing prediction: {str(e)}', 'danger')
        return redirect(url_for('module2.blackbox'))