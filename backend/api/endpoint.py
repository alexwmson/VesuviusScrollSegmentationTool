from flask import Flask, request, jsonify, send_file
from pathlib import Path
from workers.tasks import run_segmentation
from config import VOLUME_MAP, OUTPUT_ROOT
from functions.send_png import send_png
from celery.result import AsyncResult
from celery_app import celery_app

app = Flask(__name__)

@app.route("/")
def sanity_check():
    return jsonify({"status": "ok", "message": "Hello! This works"})

#generate segment stuff
@app.route("/generate_segment", methods=["POST"])
def generate_segment():
    payload = request.get_json(force=True)

    task = run_segmentation.delay(
        payload["volume"],
        payload["params"],
        payload["seed"]
    )

    return jsonify({
        "task_id": task.id
    }), 202

@app.route("/jobs/<uuid>")
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
def get_render_slice(uuid, index):
    path = Path(OUTPUT_ROOT) / uuid / "pngrenders" / f"{index:02d}.png"
    if not path.exists():
        return {"error": "slice not found"}, 404
    return send_png(path)

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

    path = axis_dir / f"{index:02d}.png"
    if not path.exists():
        return {"error": "slice not found"}, 404
    return send_png(path)