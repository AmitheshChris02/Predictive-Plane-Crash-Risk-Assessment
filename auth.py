from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, RunwayAlert, RiskAlert

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Company registration."""
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validate inputs
        if not company_name or not email or not password or not confirm_password:
            flash('All fields are required', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered', 'danger')
            return render_template('register.html')

        # Create new user
        password_hash = generate_password_hash(password)
        new_user = User(
            company_name=company_name,
            email=email,
            password_hash=password_hash
        )

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Company login."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Validate inputs
        if not email or not password:
            flash('Email and password are required', 'danger')
            return render_template('login.html')

        # Check user credentials
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['company_name'] = user.company_name
            flash(f'Welcome, {user.company_name}!', 'success')
            return redirect(url_for('auth.dashboard'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    """Logout and clear session."""
    session.pop('user_id', None)
    session.pop('company_name', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@auth_bp.route('/dashboard')
def dashboard():
    """Company dashboard with stats for all three modules."""
    if 'user_id' not in session:
        flash('Please log in to access your dashboard', 'warning')
        return redirect(url_for('auth.login'))

    user_id = session['user_id']
    company_name = session['company_name']

    # Get stats for the dashboard
    total_runway_alerts = RunwayAlert.query.filter_by(user_id=user_id).count()
    total_risk_alerts = RiskAlert.query.filter_by(user_id=user_id).count()

    # Get latest prediction timestamp (from either runway or risk alerts)
    latest_runway = (
        RunwayAlert.query
        .filter_by(user_id=user_id)
        .order_by(RunwayAlert.created_at.desc())
        .first()
    )
    latest_risk = (
        RiskAlert.query
        .filter_by(user_id=user_id)
        .order_by(RiskAlert.created_at.desc())
        .first()
    )

    latest_timestamp = None
    if latest_runway and latest_risk:
        latest_timestamp = max(latest_runway.created_at, latest_risk.created_at)
    elif latest_runway:
        latest_timestamp = latest_runway.created_at
    elif latest_risk:
        latest_timestamp = latest_risk.created_at

    return render_template(
        'dashboard.html',
        company_name=company_name,
        total_runway_alerts=total_runway_alerts,
        total_risk_alerts=total_risk_alerts,
        latest_timestamp=latest_timestamp
    )
