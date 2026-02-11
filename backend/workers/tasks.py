import json
import uuid
import docker
import redis
from celery import shared_task
from celery_app import celery_app
from config import VOLUME_MAP, OUTPUT_ROOT, AW_BIN, VC_BIN, PNG_BIN, ALGO_IMAGE, PNGRENDER_IMAGE
from pathlib import Path
from typing import Optional, Sequence

red = redis.Redis(host = "vesuvius_redis", port = 6379, db = 1, decode_responses = True)

client = docker.from_env()

@celery_app.task(bind=True)
def run_segmentation(self, volume_key: str, params: dict, seed: Optional[Sequence[int]] = None, ip: str = "0"):
    job_uuid = self.request.id
    
    if volume_key not in VOLUME_MAP:
        raise ValueError(f"Unknown volume '{volume_key}'")

    print("OUTPUT_ROOT type:", type(OUTPUT_ROOT))
    print("job_uuid type:", type(job_uuid))

    output_root = Path(OUTPUT_ROOT)
    job_dir = output_root / job_uuid

    params_with_uuid = params.copy()
    params_with_uuid["uuid"] = job_uuid
    params_path = output_root / f"{job_uuid}_params.json"
    with open(params_path, "w") as f:
        json.dump(params_with_uuid, f, indent=2)

    volume, scale = VOLUME_MAP[volume_key]
    volume_path = Path(volume)
    try:
        self.update_state(state="PROGRESS", meta={"stage": "segmenting", "uuid": job_uuid})
        # AW segmentation command
        aw_command = [
            AW_BIN,
            "-v", f"/data/{volume_path.name}",
            "-t", "/output",
            "-p", f"/output/{params_path.name}"
        ]
        if seed:
            if len(seed) != 3:
                raise ValueError("seed must contain exactly 3 numbers")
            aw_command += ["-s", str(seed[0]), str(seed[1]), str(seed[2])]

        client.containers.run(
            ALGO_IMAGE,
            command=aw_command,
            volumes={
                str(volume_path.parent): {'bind': '/data', 'mode': 'ro'},
                str(output_root): {'bind': '/output', 'mode': 'rw'}
            },
            remove=True
        )

        self.update_state(state="PROGRESS", meta={"stage": "rendering", "uuid": job_uuid})
        # VC render command
        vc_command = [
            VC_BIN,
            "-v", f"/data/{volume_path.name}",
            "-o", f"/output/{job_uuid}/render",
            "--scale", scale,
            "-g", "0",
            "-s", f"/output/{job_uuid}",
            "-n", "16"
        ]
        client.containers.run(
            ALGO_IMAGE,
            command=vc_command,
            volumes={
                str(volume_path.parent): {'bind': '/data', 'mode': 'ro'},
                str(output_root): {'bind': '/output', 'mode': 'rw'}
            },
            remove=True
        )

        self.update_state(state="PROGRESS", meta={"stage": "tiftopng", "uuid": job_uuid})
        # PNG render command
        png_command = [
            PNG_BIN,
            f"/output/{job_uuid}/render",
            f"/output/{job_uuid}/pngrenders"
        ]
        client.containers.run(
            ALGO_IMAGE,
            command=png_command,
            volumes={
                str(output_root): {'bind': '/output', 'mode': 'rw'}
            },
            remove=True
        )

        return {
            "uuid": job_uuid,
            "output_dir": str(job_dir)
        }

    finally:
        params_path.unlink(missing_ok=True)
        red.decr(f"vesuvius:segmentation:in_progress:{ip}")
        red.decr("vesuvius:segmentation:in_progress")