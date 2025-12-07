from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    runway_alerts = db.relationship('RunwayAlert', backref='user', lazy=True)
    risk_alerts = db.relationship('RiskAlert', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.company_name}>'

class RunwayAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_filename = db.Column(db.String(200), nullable=False)
    bird_count = db.Column(db.Integer, default=0)
    pothole_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<RunwayAlert {self.id}: {self.video_filename}>'

class RiskAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    module_type = db.Column(db.String(50), nullable=False)  # "BlackBox" or "TakeoffRisk"
    prediction_label = db.Column(db.String(100), nullable=False)
    features = db.Column(db.Text)  # JSON string of input features
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<RiskAlert {self.id}: {self.module_type} - {self.prediction_label}>'