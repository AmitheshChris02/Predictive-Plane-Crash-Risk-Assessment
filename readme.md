
```markdown
# Aircraft Monitoring System

A comprehensive web-based aircraft monitoring and safety assessment system that integrates computer vision, machine learning, and real-time data analysis to provide comprehensive aircraft safety monitoring solutions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Modules](#modules)
- [Models Used](#models-used)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [API Routes](#api-routes)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

## 🎯 Overview

The Aircraft Monitoring System is a multi-module platform designed to enhance aircraft safety through:

1. **Runway Monitoring**: Real-time video analysis for bird and pothole detection
2. **Black Box Analysis**: Aircraft system health monitoring and anomaly detection
3. **Takeoff Risk Assessment**: ML-powered risk prediction based on weather, aircraft, and runway conditions

The system provides a centralized dashboard for aviation companies to monitor their fleet, receive alerts, and make data-driven safety decisions.

## ✨ Features

### Core Features
- **Multi-Company Support**: Separate dashboards for different aviation companies
- **Admin Dashboard**: Centralized monitoring of all companies and alerts
- **Real-time Video Processing**: Live video feed analysis with YOLO object detection
- **Risk Prediction**: Machine learning models for takeoff risk assessment
- **Alert System**: Automated alerts for runway hazards and system anomalies
- **Historical Data**: Track and review past alerts and predictions
- **User Authentication**: Secure login/registration system with password hashing

### Module-Specific Features

#### Module 1: Runway Monitoring
- Video upload and processing
- Real-time bird detection
- Pothole detection on runways
- Annotated video feed with bounding boxes
- Alert generation and storage

#### Module 2: Black Box Analysis
- Aircraft system parameter monitoring
- Rule-based risk assessment
- Multiple failure type detection:
  - Engine Failure Risk
  - Hydraulic Failure Risk
  - Electrical Anomaly
  - Sensor Malfunction
- Real-time parameter input and analysis

#### Module 3: Takeoff Risk Assessment
- Weather data integration
- Aircraft database with real airline data
- Multi-factor risk prediction (Low/Medium/High)
- SHAP explainability for predictions
- What-if scenario analysis
- Actionable recommendations


## 📦 Modules

### Module 1: Runway Monitoring (`module1_routes.py`)

**Purpose**: Detect birds and potholes on runways using computer vision

**Workflow**:
1. User uploads runway video
2. Video is processed frame-by-frame in background thread
3. Two YOLO models analyze each frame:
   - `best.pt`: Pothole detection
   - `best1.pt`: Bird/aircraft detection
4. Detections are annotated on video feed
5. Results stored in `RunwayAlert` table

**Key Features**:
- Real-time MJPEG video stream
- Background processing with threading
- Stop/start video control
- Detection flags (0/1 for presence/absence)

**Models**:
- **YOLO v8** (Ultralytics)
  - Model 1: `models/best.pt` - Pothole detection
  - Model 2: `models/best1.pt` - Bird/Plane detection
  - Confidence threshold: 0.5

### Module 2: Black Box Analysis (`module2_routes.py`)

**Purpose**: Monitor aircraft system health and predict failures

**Input Parameters**:
- Cycle, Altitude (ft), Airspeed (kts)
- Engine RPM, EGT Temperature
- Oil Pressure, Hydraulic Pressure
- Vibration Level, Fuel Flow
- Generator Voltage, Sensor Health

**Prediction Logic** (Rule-based):
- Risk scoring system based on parameter thresholds
- Risk categories:
  - **Normal**: Low risk score (< 2)
  - **Warning**: Medium risk (2-4)
  - **Engine_Failure_Risk**: High engine parameters
  - **Hydraulic_Failure_Risk**: Low hydraulic pressure
  - **Electrical_Anomaly**: Low generator voltage
  - **Sensor_Malfunction**: Poor sensor health

**Risk Scoring**:
- Engine factors: RPM, EGT, Fuel Flow
- Hydraulic factors: Pressure levels
- Electrical factors: Generator voltage
- Sensor factors: Health status
- Vibration: Affects all systems

### Module 3: Takeoff Risk Assessment (`module3_routes.py`)

**Purpose**: Predict takeoff risk using ML model and multiple factors

**Input Features**:
- Weather: Wind speed, visibility, temperature, precipitation
- Runway: Length, elevation, condition
- Aircraft: Age, service history, load factor, crosswind component

**Model**: 
- **Gradient Boosting Classifier** (scikit-learn)
- Model file: `models/module3_best_model_fixed.pkl`
- Output: Low / Medium / High risk classification

**Features Used**:
1. `wind_speed_kts` - Wind speed in knots
2. `visibility_km` - Visibility in kilometers
3. `runway_length_m` - Runway length in meters
4. `temp_c` - Temperature in Celsius
5. `aircraft_age_yrs` - Aircraft age in years
6. `last_service_days` - Days since last service
7. `load_factor_pct` - Passenger/cargo load percentage
8. `elevation_ft` - Runway elevation
9. `crosswind_comp` - Crosswind component
10. `precipitation` - Precipitation type (none/rain/snow)
11. `runway_condition` - Runway surface condition

**Additional Features** (from SRisk.py):
- Real-time weather API integration (Open-Meteo)
- Aircraft database with 30+ real aircraft
- SHAP explainability
- What-if scenario analysis
- Density altitude calculation

## 🤖 Models Used

### 1. YOLO (You Only Look Once) Models

**Framework**: Ultralytics YOLO v8

**Models**:
- `models/best.pt`: Custom-trained pothole detection model
- `models/best1.pt`: Custom-trained bird/aircraft detection model

**Classes**:
- Pothole model: `{0: "Potholes"}`
- Bird model: `{0: "Birds", 1: "Plane", 2: "bird"}`

**Configuration**:
- Confidence threshold: 0.5
- Real-time frame processing
- Bounding box visualization

### 2. Gradient Boosting Classifier

**Framework**: scikit-learn

**Model File**: `models/module3_best_model_fixed.pkl`

**Architecture**:
- Algorithm: Gradient Boosting Classifier
- Estimators: 200 trees
- Learning rate: 0.1
- Max depth: 5
- Random state: 42

**Training Data**:
- Synthetic dataset: 5000 samples
- Features: 11 input features
- Target: 3-class classification (Low/Medium/High risk)

**Preprocessing**:
- StandardScaler for feature normalization
- Categorical encoding for precipitation and runway condition

### 3. Autoencoder (Anomaly Detection)

**Model File**: `models/anomaly_model.pkl`

**Purpose**: Detect anomalies in aircraft sensor data

**Architecture**:
- Encoder-Decoder structure
- Reconstruction error-based detection
- Threshold-based anomaly flagging

**Features Monitored**:
- Altitude, Airspeed, Engine RPM
- Oil Pressure, Fuel Flow
- Vibration, Hydraulic Pressure

**Anomaly Types Detected**:
- Sensor Bias
- Hydraulic Leak
- Engine Failure
- Cabin Pressure Loss
- Bird Strike
- Fuel Leak
- Electrical Fault

**Note**: This model is implemented in `Sanomaly_app.py` (Streamlit application)

## 🛠️ Technology Stack

### Backend
- **Flask** 2.x - Web framework
- **SQLAlchemy** - ORM for database operations
- **SQLite** - Database (development)
- **Werkzeug** - Password hashing and security

### Machine Learning & Computer Vision
- **Ultralytics YOLO** - Object detection
- **scikit-learn** - ML models (Gradient Boosting)
- **OpenCV (cv2)** - Video processing
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation

### Frontend
- **Bootstrap 5** - UI framework
- **JavaScript** - Client-side interactivity
- **HTML5/CSS3** - Markup and styling

### Additional Libraries
- **SHAP** - Model explainability (in SRisk.py)
- **Matplotlib** - Visualization
- **Pickle/Cloudpickle** - Model serialization
- **Open-Meteo API** - Weather data (in SRisk.py)

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd gravitypakka
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install flask flask-sqlalchemy werkzeug ultralytics opencv-python numpy pandas scikit-learn matplotlib shap
```

**Note**: Create a `requirements.txt` with:
```
Flask==2.3.0
Flask-SQLAlchemy==3.0.5
Werkzeug==2.3.6
ultralytics==8.0.0
opencv-python==4.8.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
matplotlib==3.7.0
shap==0.42.0
```

### Step 4: Set Up Database
The database will be created automatically on first run. Ensure write permissions in the project directory.

### Step 5: Verify Model Files
Ensure the following model files exist:
- `models/best.pt` - Pothole detection model
- `models/best1.pt` - Bird detection model
- `models/module3_best_model_fixed.pkl` - Takeoff risk model
- `models/anomaly_model.pkl` - Anomaly detection model (optional)

### Step 6: Run the Application
```bash
on terminal 1
streamlit run Sanomaly_app.py --server.port 8501

on terminal 2 
streamlit run SRisk.py --server.port 8502

on terminal 3
python app.py
```

The application will be available at `http://localhost:5000`

## 🚀 Usage

### For Companies

1. **Registration**
   - Navigate to `/register`
   - Enter company name, email, and password
   - Login after registration

2. **Dashboard**
   - View total alerts (runway and risk)
   - Access all three modules
   - View latest prediction timestamps

3. **Module 1: Runway Monitoring**
   - Go to `/module1/runway_monitor`
   - Upload a runway video file
   - View real-time detection feed at `/module1/video_results`
   - Check alerts at `/module1/runway_alerts`

4. **Module 2: Black Box Analysis**
   - Go to `/module2/blackbox`
   - Enter aircraft system parameters
   - Submit to get risk prediction
   - View alerts at `/risk_alerts?filter=blackbox`

5. **Module 3: Takeoff Risk Assessment**
   - Go to `/module3/takeoff_risk`
   - Enter weather and runway conditions
   - Submit to get risk level (Low/Medium/High)
   - View alerts at `/risk_alerts?filter=takeoff`

### For Administrators

1. **Admin Login**
   - Navigate to `/admin/login`
   - Credentials: `admin` / `admin123`
   - Access admin dashboard

2. **Admin Dashboard** (`/admin/dashboard`)
   - View all registered companies
   - Total runway alerts across all companies
   - Total risk alerts across all companies
   - Recent alerts with company names
   - Risk type distribution (BlackBox and TakeoffRisk)

## 💾 Database Schema

### User Table
```sql
- id (Integer, Primary Key)
- company_name (String, 100 chars)
- email (String, 100 chars, Unique)
- password_hash (String, 200 chars)
- created_at (DateTime)
```

### RunwayAlert Table
```sql
- id (Integer, Primary Key)
- user_id (Integer, Foreign Key -> User.id)
- video_filename (String, 200 chars)
- bird_count (Integer, default=0)
- pothole_count (Integer, default=0)
- created_at (DateTime)
```

### RiskAlert Table
```sql
- id (Integer, Primary Key)
- user_id (Integer, Foreign Key -> User.id)
- module_type (String, 50 chars) - "BlackBox" or "TakeoffRisk"
- prediction_label (String, 100 chars)
- features (Text) - JSON string of input features
- created_at (DateTime)
```

## 🔌 API Routes

### Authentication Routes (`auth.py`)
- `GET/POST /register` - Company registration
- `GET/POST /login` - Company login
- `GET /logout` - Logout
- `GET /dashboard` - Company dashboard

### Module 1 Routes (`module1_routes.py`)
- `GET /module1/runway_monitor` - Upload page
- `GET /module1/video_results` - Video results page
- `POST /module1/upload_video` - Upload video
- `GET /module1/video_feed` - MJPEG video stream
- `GET /module1/video_status` - JSON status
- `GET /module1/stop_video` - Stop processing
- `GET /module1/runway_alerts` - View alerts
- `GET /module1/download_video/<filename>` - Download video

### Module 2 Routes (`module2_routes.py`)
- `GET /module2/blackbox` - Black box analysis page
- `POST /module2/predict` - Get prediction

### Module 3 Routes (`module3_routes.py`)
- `GET /module3/takeoff_risk` - Takeoff risk page
- `POST /module3/predict` - Get risk prediction
- `GET /risk_alerts` - View risk alerts (with filters)

### Admin Routes (`app.py`)
- `GET/POST /admin/login` - Admin login
- `GET /admin/dashboard` - Admin dashboard
- `GET /admin/logout` - Admin logout

### Main Routes (`app.py`)
- `GET /` - Home page

## 📁 Project Structure

```
gravitypakka/
├── app.py                      # Main Flask application
├── models.py                   # Database models
├── config.py                    # Configuration
├── auth.py                     # Authentication routes
├── module1_routes.py           # Runway monitoring module
├── module2_routes.py           # Black box analysis module
├── module3_routes.py           # Takeoff risk assessment module
├── SRisk.py                    # Enhanced risk assessment (Streamlit)
├── Sanomaly_app.py             # Anomaly detection (Streamlit)
│
├── models/                     # ML model files
│   ├── best.pt                 # YOLO pothole model
│   ├── best1.pt                # YOLO bird model
│   ├── module3_best_model_fixed.pkl
│   ├── anomaly_model.pkl
│   └── ...
│
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── admin_dashboard.html
│   ├── module1_*.html
│   ├── module2_*.html
│   └── module3_*.html
│
├── static/                     # Static files
│   ├── css/
│   ├── js/
│   ├── images/
│   └── videos/
│
├── uploads/                    # Uploaded videos
├── instance/                   # Database instance
│   └── aircraft_monitoring.db
├── logs/                       # Application logs
└── aircraft_db_enhanced.csv    # Aircraft database
```

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Weather Integration**: Direct API integration in Module 3
2. **Advanced Anomaly Detection**: Integration of autoencoder model into main app
3. **Email Notifications**: Alert notifications via email
4. **Mobile App**: React Native mobile application
5. **Advanced Analytics**: Dashboard with charts and trends
6. **Multi-language Support**: Internationalization
7. **API Documentation**: Swagger/OpenAPI documentation
8. **Docker Deployment**: Containerization for easy deployment
9. **Cloud Integration**: AWS/Azure deployment options
10. **Enhanced Security**: JWT tokens, rate limiting, CSRF protection

### Model Improvements
1. **Model Retraining Pipeline**: Automated retraining with new data
2. **Model Versioning**: Track model versions and performance
3. **A/B Testing**: Compare model performance
4. **Ensemble Methods**: Combine multiple models for better accuracy
5. **Transfer Learning**: Improve YOLO models with domain-specific data

## 📝 Notes

- **Security**: Change the secret key in production (`app.py` line 15)
- **Database**: SQLite is used for development. Consider PostgreSQL for production
- **Model Files**: Ensure model files are present before running the application
- **Video Processing**: Large videos may take time to process
- **Admin Credentials**: Change default admin credentials in production

## 🤝 Contributing

This is a project repository. Contributions are welcome! Please follow standard Git workflow:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Specify your license here]

## 👥 Authors

[Your name/team name]

## 🙏 Acknowledgments

- Ultralytics for YOLO framework
- scikit-learn community
- Flask development team
- Open-Meteo for weather data API

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Active Development
```

This README includes:
- Project overview and features
- System architecture diagram
- Module descriptions
- Models used (YOLO, Gradient Boosting, Autoencoder)
- Installation steps
- Usage instructions
- Database schema
- API routes
- Project structure
- Future enhancements
