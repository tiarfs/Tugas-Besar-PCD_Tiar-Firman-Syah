import os, json, uuid
from pathlib import Path

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from preprocess import preprocess_pil

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "mobilenetv3_card.pth"
LABELS_PATH = MODEL_DIR / "labels.json"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)

# -------- Load labels & model (kalau file ada) --------
device = "cuda" if torch.cuda.is_available() else "cpu"
labels = None
model = None

if LABELS_PATH.exists():
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

if MODEL_PATH.exists() and labels is not None:
    num_classes = len(labels)
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXT

@app.get("/")
def index():
    return render_template("index.html", model_ready=(model is not None))

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return render_template("result.html", error="Tidak ada file yang diupload.", model_ready=(model is not None))

    f = request.files["image"]
    if f.filename == "":
        return render_template("result.html", error="Nama file kosong.", model_ready=(model is not None))

    if not allowed_file(f.filename):
        return render_template("result.html", error="Format file tidak didukung (jpg/png/webp).", model_ready=(model is not None))

    # save upload
    safe_name = secure_filename(f.filename)
    uid = uuid.uuid4().hex
    ext = Path(safe_name).suffix.lower()
    in_name = f"{uid}_orig{ext}"
    in_path = UPLOAD_DIR / in_name
    f.save(in_path)

    # load + preprocess
    img = Image.open(in_path).convert("RGB")
    img_pp = preprocess_pil(img)

    # save preprocessed for display
    out_name = f"{uid}_pre.png"
    out_path = UPLOAD_DIR / out_name
    img_pp.save(out_path)

    orig_url = url_for("uploaded_file", filename=in_name)
    pre_url = url_for("uploaded_file", filename=out_name)

    if model is None or labels is None:
        return render_template(
            "result.html",
            error="Model belum siap. Pastikan models/mobilenetv3_card.pth dan models/labels.json sudah ada.",
            orig_url=orig_url,
            pre_url=pre_url,
            model_ready=False
        )

    # inference
    x = to_tensor(img_pp).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top3_idx = probs.argsort()[-3:][::-1]
    pred_label = labels[int(top3_idx[0])]
    confidence = float(probs[int(top3_idx[0])])

    top3 = [(labels[int(i)], float(probs[int(i)])) for i in top3_idx]

    return render_template(
        "result.html",
        orig_url=orig_url,
        pre_url=pre_url,
        pred_label=pred_label,
        confidence=confidence,
        top3=top3,
        model_ready=True
    )

@app.get("/uploads/<filename>")
def uploaded_file(filename):
    # serve uploaded images (simple way)
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
