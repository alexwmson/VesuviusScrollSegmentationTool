from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pathlib import Path
from workers.tasks import run_segmentation
from config import VOLUME_MAP, OUTPUT_ROOT
from functions.send_png import send_png
from celery.result import AsyncResult
from celery_app import celery_app
from uuid import uuid4
import redis
import zipfile
import io
import json
import hashlib

app = Flask(__name__)
CORS(app, origins = ["http://localhost:5173"])

limiter = Limiter(get_remote_address, app = app, storage_uri = "redis://vesuvius_redis:6379/1", default_limits = ["2000 per day", "500 per hour"])
red = redis.Redis(host = "vesuvius_redis", port = 6379, db = 1, decode_responses=True)

@app.route("/")
def sanity_check():
    return jsonify({"status": "ok", "message": "Hello! This works"})

#generate segment stuff
@app.route("/generate_segment", methods=["POST"])
def generate_segment():
    payload = request.get_json(force=True)
    params = payload.get("params", {})

    errors = validate_params(params)
    if errors:
        return {
            "error": "Invalid parameters",
            "details": errors
        }, 400
    
    volume_key = payload["volume"]
    seed = payload.get("seed", [])
    signature = create_signature(volume_key, params, seed)

    dedupe_key = f"vesuvius:dedupe:{signature}"
    lock_key = f"vesuvius:dedupe_lock:{signature}"

    existing_uuid = red.get(dedupe_key)
    if existing_uuid:
        return jsonify({"uuid": existing_uuid, "deduped": True}), 202

    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    cur = red.incr(f"vesuvius:segmentation:in_progress:{ip}")
    glbl = red.incr("vesuvius:segmentation:in_progress")

    red.expire(f"vesuvius:segmentation:in_progress:{ip}", 1800)
    red.expire("vesuvius:segmentation:in_progress", 1800)

    if cur > 5:
        red.decr(f"vesuvius:segmentation:in_progress:{ip}")
        red.decr("vesuvius:segmentation:in_progress")
        return {
            "error": "Too many segmentation jobs in progress for this ip",
            "limit": 5,
        }, 429

    if glbl > 10:
        red.decr(f"vesuvius:segmentation:in_progress:{ip}")
        red.decr("vesuvius:segmentation:in_progress")
        return {
            "error": "Too many segmentation jobs in progress globally, please wait a bit",
            "limit": 10,
        }, 429

    job_uuid = str(uuid4())

    # lock to make sure we dont have the same job runnig twice
    got_lock = red.set(lock_key, "1", nx = True, ex = 60)
    if not got_lock:
        existing_uuid = red.get(dedupe_key)
        if existing_uuid:
            return jsonify({"uuid": existing_uuid, "deduped": True}), 202
        return jsonify({"error": "Job already being enqueued, try again"}), 409

    task = run_segmentation.apply_async(
        args=[
            payload["volume"],
            payload["params"],
            payload["seed"],
            ip,
            signature
        ],
        task_id=job_uuid
    )

    red.set(dedupe_key, job_uuid, ex = 86400) # a day
    red.delete(lock_key)

    return jsonify({
        "uuid": job_uuid
    }), 202

@app.route("/jobs/<uuid>")
@limiter.limit("10 per second")
def job_status(uuid):
    result = AsyncResult(uuid, app=celery_app)

    response = {
        "state": result.state,
    }

    if result.state == "PROGRESS":
        response.update(result.info)

    if result.state == "SUCCESS":
        response["result"] = result.result

    if result.state == "FAILURE":
        response["error"] = str(result.info)

    return jsonify(response)

#grayscale stuff
@app.route("/jobs/<uuid>/grayscale", methods=["GET"])
def get_grayscale(uuid):
    path = Path(OUTPUT_ROOT) / uuid / "grayscale.png"
    if not path.exists():
        return {"error": "not found"}, 404
    return send_png(path)

#render stuff
@app.route("/jobs/<uuid>/pngrenders", methods=["GET"])
def get_render_info(uuid):
    render_dir = Path(OUTPUT_ROOT) / uuid / "pngrenders"
    if not render_dir.exists():
        return {"error": "not found"}, 404

    files = sorted(render_dir.glob("*.png"))
    return {
        "count": len(files),
        "min_index": 1,
        "max_index": len(files)
    }

@app.route("/jobs/<uuid>/pngrenders/<int:index>", methods=["GET"])
@limiter.limit("120 per minute")
def get_render_slice(uuid, index):
    path = Path(OUTPUT_ROOT) / uuid / "pngrenders" / f"{index:02d}.png"
    if not path.exists():
        return {"error": "slice not found"}, 404
    return send_png(path)

#meta / download stuff
@app.route("/jobs/<uuid>/meta")
def get_meta(uuid):
    meta_path = Path(OUTPUT_ROOT) / uuid / "meta.json"
    if not meta_path.exists():
        return {"error": "meta.json not found"}, 404

    with open(meta_path, 'r') as f:
        meta_data = json.load(f)

    return jsonify(meta_data)

@app.route("/jobs/<uuid>/download", methods=["GET"])
def download_segment(uuid):
    segment_dir = Path(OUTPUT_ROOT) / uuid
    if not segment_dir.exists():
        return {"error": "segment not found"}, 404

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in segment_dir.rglob("*"):
            if path.is_file():
                zipf.write(path, arcname = path.relative_to(segment_dir))
    
    zip_buffer.seek(0) # resets pointer to 0 so we don't return nothing
    return send_file(
        zip_buffer,
        mimetype = "application/zip",
        as_attachment = True,
        download_name = f"segment_{uuid}.zip"
    )


#volume stuff
@app.route("/volumes/<volume_key>/<axis>", methods=["GET"])
def volume_axis_info(volume_key, axis):
    volume = Path(VOLUME_MAP[volume_key][0])
    axis_dir = volume / "pngrenders" / axis
    if axis not in {"x", "y", "z"} or not axis_dir.exists():
        return {"error": "not found"}, 404

    count = len(list(axis_dir.glob("*.png")))
    return {
        "axis": axis,
        "count": count,
        "min_index": 1,
        "max_index": count
    }

@app.route("/volumes/<volume_key>/<axis>/<int:index>", methods=["GET"])
def get_volume_slice(volume_key, axis, index):
    volume = Path(VOLUME_MAP[volume_key][0])
    axis_dir = volume / "pngrenders" / axis
    if axis not in {"x", "y", "z"} or not axis_dir.exists():
        return {"error": "not found"}, 404

    path = axis_dir / f"{index:04d}.png"
    if not path.exists():
        return {"error": "slice not found"}, 404
    return send_png(path)


#other functions/variables

RULES = {
        "min_size": {
            "type": int,
            "min": 1,
            "max": 1_000_000,
        },
        "max_size": {
            "type": int,
            "min": 1,
            "max": 1_000_000,
        },
        "steps": {
            "type": int,
            "min": 1,
            "max": 40,
        },
        "global_threshold": {
            "type": int,
            "min": 0,
            "max": 255,
        },
        "allowed_difference": {
            "type": (int, float),
            "min": -1.0,
            "max": 1.0,
        },
        "max_patience": {
            "type": int,
            "min": 1,
            "max": 50,
        },
    }

def validate_params(params: dict):
    errors = []

    for name, rules in RULES.items():
        if name not in params:
            continue

        value = params[name]

        if not isinstance(value, rules["type"]):
            errors.append(f"{name} must be {rules['type']}")
            continue

        if "min" in rules and value < rules["min"]:
            errors.append(f"{name} must be >= {rules['min']}")

        if "max" in rules and value > rules["max"]:
            errors.append(f"{name} must be <= {rules['max']}")

    #make sure min size is less or equal to max size
    if "min_size" in params and "max_size" in params:
        if params["min_size"] >= params["max_size"]:
            errors.append("max_size must be greater than min_size")
    if "min_size" in params and not "max_size" in params:
        if params["min_size"] > 250000:
            errors.append("min_size must be <= default max_size: 250000")
    if not "min_size" in params and "max_size" in params:
        if params["max_size"] < 50000:
            errors.append("max_size must be >= default min_size: 50000")

    return errors

def normalize_payload(volume: str, params: dict, seed) -> dict:
    if seed is None:
        seed_norm = []
    else:
        seed_norm = list(seed)

    params_norm = {}
    for k in sorted(params.keys()):
        v = params[k]
        if v is None:
            continue

        if isinstance(v, float):
            v = round(v, 6)

        params_norm[k] = v

    return {"volume": volume, "params": params_norm, "seed": seed_norm}

def create_signature(volume: str, params: dict, seed) -> str:
    canonical = normalize_payload(volume, params, seed)
    payload_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()