import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import desc
from models import db, User, RunwayAlert, RiskAlert
from auth import auth_bp
from module1_routes import module1_bp
from module2_routes import module2_bp
from module3_routes import module3_bp
from datetime import datetime
import json
from markupsafe import Markup

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aircraft_monitoring.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(module1_bp)
app.register_blueprint(module2_bp)
app.register_blueprint(module3_bp)


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for admin login
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Hardcoded admin credentials
        if username == 'admin' and password == 'admin123':
            session['is_admin'] = True
            flash('Admin login successful', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'danger')
    
    return render_template('admin_login.html')


# Route for admin dashboard
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        flash('Admin access required', 'danger')
        return redirect(url_for('admin_login'))
    
    # Get all companies
    companies = User.query.all()
    
    # Get aggregated stats
    total_runway_alerts = RunwayAlert.query.count()
    total_risk_alerts = RiskAlert.query.count()
    
    # Get runway alerts
    runway_alerts = db.session.query(
        RunwayAlert, User.company_name
    ).join(User).order_by(desc(RunwayAlert.created_at)).limit(10).all()
    
    # Get risk alerts
    risk_alerts = db.session.query(
        RiskAlert, User.company_name
    ).join(User).order_by(desc(RiskAlert.created_at)).limit(10).all()
    
    # Count each risk type
    blackbox_counts = {}
    takeoff_counts = {}
    
    for alert_type, count in db.session.query(
        RiskAlert.prediction_label, db.func.count(RiskAlert.id)
    ).filter(RiskAlert.module_type == 'BlackBox').group_by(RiskAlert.prediction_label).all():
        blackbox_counts[alert_type] = count
    
    for alert_type, count in db.session.query(
        RiskAlert.prediction_label, db.func.count(RiskAlert.id)
    ).filter(RiskAlert.module_type == 'TakeoffRisk').group_by(RiskAlert.prediction_label).all():
        takeoff_counts[alert_type] = count
    
    return render_template(
        'admin_dashboard.html',
        companies=companies,
        total_runway_alerts=total_runway_alerts,
        total_risk_alerts=total_risk_alerts,
        runway_alerts=runway_alerts,
        risk_alerts=risk_alerts,
        blackbox_counts=blackbox_counts,
        takeoff_counts=takeoff_counts
    )


# Route for admin logout
@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    flash('Admin logout successful', 'success')
    return redirect(url_for('index'))


# Add template filter for JSON
@app.template_filter('from_json')
def from_json(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}

if __name__ == '__main__':
    app.run(debug=True)
