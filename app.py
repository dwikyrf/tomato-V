from flask import Flask, render_template, request
import os
from PIL import Image
import torch

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load YOLOv5 from torch hub
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="models/best.pt",
    source="github",
    force_reload=False
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filename = file.filename
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # YOLO detect
        results = model(save_path)

        # Save result
        results.render()
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename)
        img = Image.fromarray(results.ims[0])
        img.save(result_path)

        return render_template(
            "result.html",
            original=save_path,
            result=result_path
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
