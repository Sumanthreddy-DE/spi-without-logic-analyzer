#!/usr/bin/env python3
"""
Food recognition from the phone camera (IP Webcam app) - runs locally on the laptop.

Recognising the food does NOT give the waste type (origin) - buttons do that.
The camera adds a "food" field so the insight can name the item: "62 % of prep waste is pasta".

Setup:  pip install torch transformers pillow numpy
        Phone: IP Webcam app -> Start server, phone looks straight down at the tray only.

Test:   python vision.py --phone http://192.168.43.1:8080
        Empty tray -> Enter (before photo), put item -> Enter (after photo) -> prints the food.
Privacy: only the cropped tray region is saved (vision/crops/), never the full frame.
"""
import argparse
import io
import os
import time
import urllib.request

import numpy as np
from PIL import Image

# Edit freely: label -> text prompt. Add what the mentors bring.
FOODS = {
    "pasta": "a photo of dry pasta",
    "apple": "a photo of an apple or apple pieces",
    "bread": "a photo of bread or bread slices",
    "orange": "a photo of an orange or orange peel",
    "banana": "a photo of a banana or banana peel",
    "vegetable_trimmings": "a photo of vegetable peels and trimmings",
    "rice": "a photo of cooked rice",
    "empty": "a photo of an empty white tray",
}
MIN_CONF = 0.45  # below this -> "unknown" (honest beats wrong)

_model = _proc = None


def load_model():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor  # first run downloads ~600 MB
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        _proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _model, _proc


def snapshot(phone_url):
    with urllib.request.urlopen(phone_url.rstrip("/") + "/shot.jpg", timeout=3) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def changed_region(before, after, thresh=35, pad=20):
    """Bounding box of what changed between two frames (the new item). None if nothing changed."""
    a = np.asarray(before.convert("L").resize(after.size), dtype=np.int16)
    b = np.asarray(after.convert("L"), dtype=np.int16)
    mask = np.abs(a - b) > thresh
    if mask.mean() < 0.002:          # < 0.2 % of pixels changed
        return None
    ys, xs = np.where(mask)
    x0, x1 = np.percentile(xs, [2, 98]).astype(int)   # percentiles ignore stray noise pixels
    y0, y1 = np.percentile(ys, [2, 98]).astype(int)
    w, h = after.size
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def classify(img):
    import torch
    model, proc = load_model()
    labels = list(FOODS)
    inputs = proc(text=[FOODS[l] for l in labels], images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0].tolist()
    ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
    best, conf = ranked[0]
    if conf < MIN_CONF or best == "empty":
        best = "unknown"
    return best, conf, ranked[:3]


def recognise(before, after, save_dir=None, tag=None):
    box = changed_region(before, after)
    crop = after.crop(box) if box else after
    food, conf, top3 = classify(crop)
    image_id = ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        image_id = f"{tag or int(time.time())}.jpg"
        crop.save(os.path.join(save_dir, image_id), quality=80)
    return {"food": food, "food_conf": round(conf, 2), "image_id": image_id, "top3": top3, "box": box}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", required=True, help="e.g. http://192.168.43.1:8080")
    a = ap.parse_args()
    print("loading model (first time downloads it)...")
    load_model()
    crops = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crops")
    while True:
        input("\nEmpty/current tray - press Enter for BEFORE photo")
        before = snapshot(a.phone)
        input("Put the item on the tray - press Enter for AFTER photo")
        after = snapshot(a.phone)
        t = time.time()
        r = recognise(before, after, save_dir=crops)
        print(f"-> {r['food']}  ({r['food_conf']:.0%})   top3: " +
              ", ".join(f"{l} {p:.0%}" for l, p in r["top3"]) +
              f"   box={r['box']}  {1000*(time.time()-t):.0f} ms  saved {r['image_id']}")


if __name__ == "__main__":
    main()
