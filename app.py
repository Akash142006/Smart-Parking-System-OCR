from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from io import BytesIO
from werkzeug.datastructures import FileStorage
import os
from datetime import datetime
from database import init_db, check_plate, get_all_plates, add_plate, add_plates_bulk, delete_plate
from ocr import find_plate_number

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Avoid showing stale hardware image after restart.
latest_image_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
if os.path.exists(latest_image_path):
    try:
        os.remove(latest_image_path)
    except OSError:
        pass

latest_detection = {
    "plate": None,
    "ocr_success": False,
    "status": "UNKNOWN",
    "owner": None,
    "note": "No detection yet",
    "candidates": [],
    "image_url": None,
    "updated_at": None,
}


def _set_latest_detection(payload):
    latest_detection.update(payload)
    latest_detection["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_detection["image_url"] = "/uploads/latest.jpg"

# ─────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search")
def search_page():
    return render_template("search.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ─────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Receive image from hardware or website, run OCR, check database"""
    file = None
    is_live_mode = request.form.get("mode") == "live"

    if "image" in request.files:
        file = request.files["image"]
    elif request.data:
        file = FileStorage(
            stream=BytesIO(request.data),
            filename=request.headers.get("X-Filename", "capture.jpg"),
            content_type=request.headers.get("Content-Type", "application/octet-stream"),
        )
    else:
        return jsonify({"error": "No image received"}), 400

    if not file.filename:
        file.filename = "capture.jpg"

    # Save the uploaded image
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], "latest.jpg")
    file.save(image_path)

    # Run OCR on the image
    ocr_result = find_plate_number(image_path, fast=is_live_mode)

    best_candidate = ocr_result.get("plate") or "UNREADABLE"
    candidate_list = ocr_result.get("candidates", []) or []

    # In live mode, also try the best recognized candidate even if strict format matching failed.
    should_try_candidate_check = is_live_mode and best_candidate and best_candidate != "UNREADABLE"

    # If OCR is not confident about a valid plate format, keep status UNKNOWN
    if not ocr_result.get("success"):
        if should_try_candidate_check:
            try:
                db_result = check_plate(best_candidate)
                payload = {
                    "plate": best_candidate,
                    "ocr_success": False,
                    "status": "ALLOWED" if db_result["found"] else "DENIED",
                    "owner": db_result["owner"],
                    "note": ocr_result.get("note", "Live OCR used a best-effort candidate"),
                    "candidates": candidate_list,
                    "checked_candidate": True,
                }
                _set_latest_detection(payload)
                return jsonify(payload), 200
            except Exception as exc:
                payload = {
                    "plate": best_candidate,
                    "status": "PENDING_DB",
                    "owner": None,
                    "candidates": candidate_list,
                    "note": "Plate detected, but DB check is temporarily unavailable",
                    "error": str(exc),
                    "checked_candidate": True,
                }
                _set_latest_detection(payload)
                return jsonify(payload), 200

        payload = {
            "plate": ocr_result.get("plate", "UNREADABLE"),
            "ocr_success": False,
            "status": "UNKNOWN",
            "owner": None,
            "note": ocr_result.get("note", "OCR could not confidently detect a valid plate"),
            "candidates": candidate_list,
        }
        _set_latest_detection(payload)
        return jsonify(payload), 200

    # Check plate in database
    plate = ocr_result["plate"]
    try:
        db_result = check_plate(plate)
    except Exception as exc:
        payload = {
            "plate": plate,
            "status": "PENDING_DB",
            "owner": None,
            "ocr_success": ocr_result["success"],
            "candidates": ocr_result.get("candidates", []),
            "note": "Plate detected, but DB check is temporarily unavailable",
            "error": str(exc),
        }
        _set_latest_detection(payload)
        return jsonify(payload), 200

    payload = {
        "plate": plate,
        "ocr_success": ocr_result["success"],
        "status": "ALLOWED" if db_result["found"] else "DENIED",
        "owner": db_result["owner"],
        "candidates": ocr_result.get("candidates", [])
    }
    _set_latest_detection(payload)
    return jsonify(payload), 200


@app.route("/api/latest-detection", methods=["GET"])
def latest_detection_api():
    return jsonify(latest_detection), 200


@app.route("/api/search", methods=["POST"])
def search_plate():
    """NLP search — user types natural language, we extract plate and check"""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    query = data["query"].upper()

    # Extract plate number from natural language query
    import re
    plate_pattern = re.compile(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}')
    match = plate_pattern.search(query)

    if not match:
        return jsonify({
            "error": "No valid plate number found in your query",
            "tip": "Try something like: Is KA01AB1234 allowed?"
        }), 200

    plate = match.group()
    try:
        db_result = check_plate(plate)
    except Exception as exc:
        return jsonify(
            {
                "plate": plate,
                "status": "PENDING_DB",
                "owner": None,
                "error": str(exc),
                "note": "Plate extracted, but DB check is temporarily unavailable",
            }
        ), 200

    return jsonify({
        "plate": plate,
        "status": "ALLOWED" if db_result["found"] else "DENIED",
        "owner": db_result["owner"]
    }), 200


@app.route("/api/plates", methods=["GET"])
def get_plates():
    """Get all registered plates for admin page"""
    try:
        return jsonify(get_all_plates()), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.route("/api/plates/add", methods=["POST"])
def add_new_plate():
    """Add a new plate from admin page"""
    data = request.get_json()
    if not data or "plate" not in data or "owner" not in data:
        return jsonify({"error": "Plate and owner required"}), 400
    try:
        result = add_plate(data["plate"], data["owner"])
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 503


@app.route("/api/plates/bulk", methods=["POST"])
def bulk_upload_plates():
    """Bulk upload number plates from admin page to remote DB"""
    data = request.get_json()
    if not data or "plates" not in data:
        return jsonify({"error": "plates field is required"}), 400

    owner = data.get("owner", "Bulk Upload")
    try:
        result = add_plates_bulk(data["plates"], owner)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 503


@app.route("/api/plates/delete", methods=["POST"])
def delete_existing_plate():
    """Delete a plate from admin page"""
    data = request.get_json()
    if not data or "plate" not in data:
        return jsonify({"error": "Plate required"}), 400
    try:
        result = delete_plate(data["plate"])
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 503


# ─────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────

if __name__ == "__main__":
    try:
        init_db()
    except Exception as exc:
        print(f"Startup warning: {exc}")
        print("Server will run, but remote DB operations may fail until credentials/connectivity are fixed.")
    print("Starting Smart Parking Server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)