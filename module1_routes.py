from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    current_app,
    Response,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from models import db, RunwayAlert, User
import os
import threading
import json
from datetime import datetime
import cv2
from ultralytics import YOLO
import numpy as np

module1_bp = Blueprint('module1', __name__)

# ==== YOLO models & detection state (ONLY here) ====

# Load YOLO models once
model_pothole = YOLO("models/best.pt")    # trained on potholes
model_bird = YOLO("models/best1.pt")      # trained on birds/planes

classnames_pothole = {0: "Potholes"}
classnames_bird = {0: "Birds", 1: "Plane", 2: "bird"}

CONF_THRESH = 0.5

# Global detection state for Module 1
current_video_path = None          # full path of currently processed video
current_video_filename = None      # just filename (for DB + download)
processing_lock = threading.Lock()
# bird_count / pothole_count will now be 0 or 1 (detected / not detected)
detection_results = {"bird_count": 0, "pothole_count": 0, "processing": False}
stop_requested = False             # flag to notify background thread to stop


def draw_boxes(frame, boxes, confs, classes, names, color=(0, 255, 0)):
    """Draw bounding boxes with labels on the frame."""
    for box, conf, cls in zip(boxes, confs, classes):
        if conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        label = f"{names.get(cls, 'Unknown')} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame


def process_video(video_path, user_id):
    """
    Background thread:
    - reads video
    - runs YOLO on each frame
    - sets bird_count / pothole_count as 0 or 1 (detected or not)
    - saves final flags to RunwayAlert IF user did NOT press Stop
    """
    global detection_results, stop_requested, current_video_filename

    with processing_lock:
        detection_results["processing"] = True
        detection_results["bird_count"] = 0
        detection_results["pothole_count"] = 0
        local_filename = current_video_filename  # snapshot for this run
        local_stop = stop_requested = False      # reset stop flag at start

    cap = cv2.VideoCapture(video_path)

    bird_detected = False
    pothole_detected = False

    while True:
        # Check if user pressed Stop
        with processing_lock:
            local_stop = stop_requested
        if local_stop:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO models
        results_pothole = model_pothole(frame, conf=CONF_THRESH, verbose=False)[0]
        results_bird = model_bird(frame, conf=CONF_THRESH, verbose=False)[0]

        # If we see any potholes in any frame -> mark as detected
        if len(results_pothole.boxes) > 0:
            pothole_detected = True

        # If we see any birds/planes in any frame -> mark as detected
        if len(results_bird.boxes) > 0:
            bird_detected = True

        # Update global results (0 or 1)
        with processing_lock:
            detection_results["bird_count"] = 1 if bird_detected else 0
            detection_results["pothole_count"] = 1 if pothole_detected else 0

    cap.release()

    # Final flags from what we saw so far
    with processing_lock:
        bird_flag = 1 if bird_detected else detection_results["bird_count"]
        pothole_flag = 1 if pothole_detected else detection_results["pothole_count"]
        local_stop = stop_requested
        filename = local_filename

    # Save to DB ONLY if user did NOT press Stop (normal completion)
    if not local_stop and filename is not None:
        with current_app.app_context():
            video_filename = filename
            runway_alert = RunwayAlert(
                user_id=user_id,
                video_filename=video_filename,
                bird_count=bird_flag,
                pothole_count=pothole_flag,
                created_at=datetime.now(),
            )
            db.session.add(runway_alert)
            db.session.commit()

    with processing_lock:
        detection_results["processing"] = False


# ================== ROUTES ==================


@module1_bp.route('/module1/runway_monitor')
def runway_monitor():
    """Page to upload a new runway video."""
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('auth.login'))

    return render_template('module1_runway.html')


@module1_bp.route('/module1/video_results')
def video_results():
    """Page that shows the video feed + detection stats."""
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('auth.login'))

    return render_template('module1_video_results.html')


@module1_bp.route('/module1/upload_video', methods=['POST'])
def upload_video():
    """Handle runway video upload and start YOLO processing in background."""
    global current_video_path, current_video_filename, detection_results, stop_requested

    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('module1.runway_monitor'))

    if 'video' not in request.files:
        flash('No video file uploaded', 'danger')
        return redirect(url_for('module1.runway_monitor'))

    video = request.files['video']
    if video.filename == '':
        flash('No video file selected', 'danger')
        return redirect(url_for('module1.runway_monitor'))

    if video:
        # Save the video file
        filename = secure_filename(video.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        video_path = os.path.join(upload_folder, filename)
        video.save(video_path)

        # Update global state in this module
        with processing_lock:
            current_video_path = video_path
            current_video_filename = filename
            stop_requested = False
            detection_results["bird_count"] = 0
            detection_results["pothole_count"] = 0
            detection_results["processing"] = True

        # Start processing in a separate background thread
        user_id = session['user_id']
        processing_thread = threading.Thread(
            target=process_video,
            args=(video_path, user_id),
            daemon=True
        )
        processing_thread.start()

        flash('Video uploaded successfully. Processing started.', 'success')
        # Go to page that shows <img src="{{ url_for("module1.video_feed") }}">
        return redirect(url_for('module1.video_results'))

    return redirect(url_for('module1.runway_monitor'))


@module1_bp.route('/module1/video_feed')
def video_feed():
    """MJPEG stream of the processed video."""
    global current_video_path

    print("DEBUG /module1/video_feed current_video_path =", current_video_path)

    if not current_video_path:
        return Response("No video selected", status=404)

    def generate():
        cap = cv2.VideoCapture(current_video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run both models, draw boxes
            results_pothole = model_pothole(frame, conf=CONF_THRESH, verbose=False)[0]
            results_bird = model_bird(frame, conf=CONF_THRESH, verbose=False)[0]

            # Draw pothole detections (red)
            if len(results_pothole.boxes) > 0:
                boxes = results_pothole.boxes.xyxy.cpu().numpy()
                confs = results_pothole.boxes.conf.cpu().numpy()
                classes = results_pothole.boxes.cls.cpu().numpy()
                frame = draw_boxes(
                    frame, boxes, confs, classes, classnames_pothole, color=(0, 0, 255)
                )

            # Draw bird/plane detections (green)
            if len(results_bird.boxes) > 0:
                boxes = results_bird.boxes.xyxy.cpu().numpy()
                confs = results_bird.boxes.conf.cpu().numpy()
                classes = results_bird.boxes.cls.cpu().numpy()
                frame = draw_boxes(
                    frame, boxes, confs, classes, classnames_bird, color=(0, 255, 0)
                )

            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@module1_bp.route('/module1/video_status')
def video_status():
    """Return JSON with video/detection status."""
    global current_video_path, detection_results
    print("DEBUG /module1/video_status current_video_path =", current_video_path,
          "processing =", detection_results.get('processing'))
    return json.dumps({
        'video_loaded': current_video_path is not None,
        'processing': detection_results.get('processing', False),
        'bird_count': detection_results.get('bird_count', 0),
        'pothole_count': detection_results.get('pothole_count', 0)
    })


@module1_bp.route('/module1/stop_video')
def stop_video():
    """
    Stop video processing and clear current video.
    ALSO: save a RunwayAlert based on detections so far.
    """
    global current_video_path, current_video_filename, detection_results, stop_requested

    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('auth.login'))

    user_id = session['user_id']

    with processing_lock:
        # Tell background thread to stop
        stop_requested = True

        # Snapshot current detection flags and filename
        bird_flag = 1 if detection_results.get("bird_count", 0) >= 1 else 0
        pothole_flag = 1 if detection_results.get("pothole_count", 0) >= 1 else 0
        filename = current_video_filename

        # Clear current video for UI
        current_video_path = None
        current_video_filename = None
        detection_results["processing"] = False

    # Save alert in DB for this partial run (if we had a video)
    if filename is not None:
        with current_app.app_context():
            runway_alert = RunwayAlert(
                user_id=user_id,
                video_filename=filename,
                bird_count=bird_flag,
                pothole_count=pothole_flag,
                created_at=datetime.now(),
            )
            db.session.add(runway_alert)
            db.session.commit()

    return json.dumps({'status': 'success'})


@module1_bp.route('/module1/runway_alerts')
def runway_alerts():
    """Show runway alerts for the logged-in company."""
    if 'user_id' not in session:
        flash('Please log in to access this module', 'warning')
        return redirect(url_for('auth.login'))

    user_id = session['user_id']
    alerts = (
        RunwayAlert.query
        .filter_by(user_id=user_id)
        .order_by(RunwayAlert.created_at.desc())
        .all()
    )

    return render_template('module1_runway_alerts.html', alerts=alerts)


@module1_bp.route('/module1/download_video/<path:filename>')
def download_video(filename):
    """Download an uploaded runway video."""
    upload_folder = current_app.config['UPLOAD_FOLDER']
    return send_from_directory(upload_folder, filename, as_attachment=True)
