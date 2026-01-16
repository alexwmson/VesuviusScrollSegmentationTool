from flask import send_file

def send_png(path):
    response = send_file(path, mimetype="image/png")
    response.headers["Cache-Control"] = "public, max-age=86400, immutable" #1 day
    return response