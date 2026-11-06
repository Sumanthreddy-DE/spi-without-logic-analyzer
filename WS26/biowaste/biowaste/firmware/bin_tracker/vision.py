#!/usr/bin/env python3
"""
Food recognition from a camera inside the bin - runs locally on the laptop.

Flow: lid opens -> BEFORE photo, food goes in -> AFTER photo -> recognise(before, after):
  1. change_mask(): what is new? Aligns the two photos (bin bumped), evens out the light, compares
     colour + texture, and returns each new piece separately (food lands on old food, so a plain
     brightness difference does not work).
  2. cutout(): each piece on its own, old waste greyed out.
  3. classify() each piece with SigLIP 2; the disposal's foods = union of the pieces.
Tested on tools/vision_bench/binpairs (bin_test.py): finds the new food with 0.79 pixel IoU (old
grey-level box: 0.17) and names it with F1 93 in a lit bin vs 5 when classifying the whole photo.
Dark bins are much worse (F1 27): put a small white LED in the bin.
Mount the camera on the bin body (rim / wall), NOT on the lid - a lid camera moves between the photos.

Recognising the food does NOT give the waste type (origin) - buttons do that.
The camera adds a "foods" field so the insight can name the items: "62 % of prep waste is pasta".

Model: SigLIP 2 (chosen with tools/vision_bench - about 4x the F1 of the old CLIP rule on real
plate leftovers). It scores every food separately, so one tip can come back as "pasta + sauce".
  * GPU with >= 3 GB free VRAM -> SigLIP 2 so400m in fp16 (best results, ~2.7 GB VRAM)
  * less VRAM, or no GPU        -> SigLIP 2 base (~0.9 GB VRAM, ~0.4 s per photo on a CPU)
PyTorch is capped at the VRAM that is free, so a full GPU fails loudly and falls back to base
instead of silently spilling into system RAM (which makes every photo take seconds).

Setup:  pip install "transformers>=4.49" pillow numpy
        pip install torch --index-url https://download.pytorch.org/whl/cu126   (NVIDIA GPU; plain
        `pip install torch` gives the CPU version). First run downloads the model (4.5 GB / 1.5 GB).
        Phone: IP Webcam app -> Start server, phone looks straight down at the tray only.
        Steady light (phone flashlight: <phone>/enabletorch) helps more than any model.

Test:   python vision.py --phone http://192.168.43.1:8080            (auto model choice)
        python vision.py --phone http://192.168.43.1:8080 --model base
        Empty tray -> Enter (before photo), put item -> Enter (after photo) -> prints the foods.
Privacy: only the cropped tray region is saved (vision/crops/), never the full frame.
"""
import argparse
import io
import os
import time
import urllib.request

import numpy as np
from PIL import Image

# Edit freely: label -> what it looks like (used in the prompts). Add what the mentors bring.
# tools/vision_bench/own/vocab.txt should list the same foods, so the benchmark tests this list.
FOODS = {
    "pasta": "pasta",
    "rice": "rice",
    "bread": "bread",
    "potatoes": "potatoes",
    "potato peels": "potato peels",
    "apple": "an apple or apple pieces",
    "apple peel or core": "apple peel or an apple core",
    "banana peel": "a banana peel",
    "orange peel": "orange peel",
    "onion skin": "onion skin",
    "carrot peels": "carrot peels",
    "vegetable trimmings": "vegetable trimmings",
    "lettuce leaves": "lettuce leaves",
    "cucumber": "cucumber",
    "tomato": "tomato",
    "egg shells": "egg shells",
    "cooked meat": "cooked meat",
    "chicken bones": "chicken bones",
    "fish": "fish",
    "cheese": "cheese",
    "sauce": "sauce",
    "salad": "salad",
    "coffee grounds": "coffee grounds",
    "tea bag": "a tea bag",
}
# Same prompt templates as the benchmark, so the thresholds below carry over.
TEMPLATES = ["a photo of {}.", "a photo of {}, food waste.", "leftover {} on a tray."]

# name -> (Hugging Face id, score threshold, free VRAM needed to pick it automatically)
# Thresholds = best F1 on the benchmark's real plate leftovers (so400m is flat from 0.03 to 0.12).
# Re-tune them on your own tray photos: tools/vision_bench, `python bench.py run --datasets own`.
MODELS = {
    "so400m": ("google/siglip2-so400m-patch14-384", 0.05, 3.0e9),
    "base": ("google/siglip2-base-patch16-384", 0.02, 1.0e9),
}

_m = None  # loaded model state


def _vram_cap(torch, margin=0.3e9):
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(0.95, (free - margin) / total)))
    return free


def _load(name, device):
    import torch
    from transformers import AutoModel, AutoProcessor
    hf_id, thr, _ = MODELS[name]
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(hf_id, dtype=dtype).to(device).eval()  # first run downloads it
    proc = AutoProcessor.from_pretrained(hf_id)
    labels = list(FOODS)
    with torch.no_grad():
        embs = []
        for lab in labels:  # one food at a time keeps the VRAM peak low
            inp = proc(text=[t.format(FOODS[lab]).lower() for t in TEMPLATES], return_tensors="pt",
                       padding="max_length", max_length=64).to(device)
            e = model.get_text_features(**inp).float()
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.mean(0))
        text = torch.stack(embs)
        text = text / text.norm(dim=-1, keepdim=True)
    state = {"name": name, "device": device, "model": model, "proc": proc, "text": text,
             "labels": labels, "threshold": thr}
    _classify(state, Image.new("RGB", (384, 384), "white"))  # warm-up: proves the forward pass fits
    return state


def load_model(prefer=None):
    """prefer: None (auto), "so400m" or "base"."""
    global _m
    if _m is not None:
        return _m
    import torch
    if not torch.cuda.is_available():
        _m = _load(prefer or "base", "cpu")
        print(f"vision: {_m['name']} on CPU (no CUDA GPU found)")
        return _m
    free = _vram_cap(torch)
    order = [prefer] if prefer else [n for n, (_, _, need) in MODELS.items() if free >= need] or ["base"]
    if "base" not in order:
        order.append("base")
    for name in order:
        try:
            _m = _load(name, "cuda")
            print(f"vision: {name} on {torch.cuda.get_device_name(0)} ({free / 1e9:.1f} GB VRAM was free, "
                  f"peak use {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)")
            return _m
        except torch.OutOfMemoryError:
            print(f"vision: {name} does not fit in the free VRAM ({free / 1e9:.1f} GB) - trying the next model")
            _m = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    _m = _load("base", "cpu")
    print("vision: base on CPU (GPU too full)")
    return _m


def snapshot(phone_url):
    with urllib.request.urlopen(phone_url.rstrip("/") + "/shot.jpg", timeout=3) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


# ---------------------------------------------------------------- what changed in the bin?
WORK = 640          # change detection runs at this width (fast, and ignores tiny jitter)
MIN_PART = 0.004    # a new piece must cover >= 0.4 % of the image
DIFF_FLOOR = 14.0   # colour difference (Lab units) below this is never "new"


def _align(gb, ga):
    """Warp `gb` (before) onto `ga` (after): the bin may have been bumped between the photos.
    ECC with a rigid (shift + turn) model; identity if it does not converge."""
    import cv2
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        pass
    return warp


def change_mask(before, after):
    """Pixels of `after` that are new waste. Returns (mask HxW bool at after's size, list of parts),
    each part = (x0, y0, x1, y1) box of one connected new piece, largest first.

    Inside a bin the new food lands on old food, so a plain grey-level difference fails. Steps:
      1. align the before photo to the after photo (bin bumped, camera shifted a few pixels)
      2. brighten dark photos the same way, then match the before photo's colours to the after
         photo on the unchanged area (light changed between the photos)
      3. difference in Lab colour + texture, each pixel compared to its best match in a 5x5
         neighbourhood of the before photo (forgives the last pixel of misalignment)
      4. threshold (Otsu, never below DIFF_FLOOR), clean up, keep pieces >= MIN_PART
    """
    import cv2
    A0 = np.asarray(after.convert("RGB"))
    H, Wd = A0.shape[:2]
    s = WORK / Wd
    size = (WORK, max(1, int(H * s)))
    a = cv2.resize(A0, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(np.asarray(before.convert("RGB").resize(after.size)), size,
                   interpolation=cv2.INTER_AREA).astype(np.float32)

    # 1. alignment
    ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) / 255.0, (0, 0), 1.5)
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) / 255.0, (0, 0), 1.5)
    warp = _align(gb.astype(np.float32), ga.astype(np.float32))
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    b = cv2.warpAffine(b, warp, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones(size[::-1], np.float32), warp, size, flags=flags) > 0.99

    # near-black in both photos (corners the lamp does not reach): only noise there, never "new"
    lum_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    lum_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    valid &= ~((cv2.GaussianBlur(lum_a, (0, 0), 3) < 22) & (cv2.GaussianBlur(lum_b, (0, 0), 3) < 22))

    # 2. light: same gamma for both if dark, then fit before -> after per channel on calm pixels
    mean = a.mean()
    if mean < 90:
        g = np.log(110 / 255) / np.log(max(mean, 5) / 255)
        a = 255 * (a / 255) ** g
        b = 255 * (b / 255) ** g
    calm = valid.copy()
    for _ in range(2):
        for ch in range(3):
            x, y = b[..., ch][calm], a[..., ch][calm]
            gain = np.clip(np.std(y) / max(np.std(x), 1e-3), 0.6, 1.6)
            b[..., ch] = (b[..., ch] - x.mean()) * gain + y.mean()
        d0 = np.abs(a - b).sum(2)
        calm = valid & (d0 < np.percentile(d0[valid], 60))

    # 3. difference: Lab colour (lightness half weight) + texture, best match in 5x5 window
    la = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lb = cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    la, lb = cv2.GaussianBlur(la, (0, 0), 1.2), cv2.GaussianBlur(lb, (0, 0), 1.2)
    w = np.array([0.5, 1.0, 1.0], np.float32)
    ta = cv2.Laplacian(la[..., 0], cv2.CV_32F, ksize=3)
    tb = cv2.Laplacian(lb[..., 0], cv2.CV_32F, ksize=3)
    D = np.full(size[::-1], np.inf, np.float32)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            sh = np.float32([[1, 0, dx], [0, 1, dy]])
            lbs = cv2.warpAffine(lb, sh, size, borderMode=cv2.BORDER_REPLICATE)
            tbs = cv2.warpAffine(tb, sh, size, borderMode=cv2.BORDER_REPLICATE)
            d = np.sqrt((((la - lbs) * w) ** 2).sum(2)) + 0.25 * np.abs(cv2.GaussianBlur(
                np.abs(ta) - np.abs(tbs), (0, 0), 2))
            D = np.minimum(D, d)
    D = cv2.GaussianBlur(D, (0, 0), 2)
    D[~valid] = 0

    # 4. threshold + clean up
    d8 = np.clip(D * 3, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (D > max(otsu / 3, DIFF_FLOOR)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    parts = []
    for k in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        if stats[k, cv2.CC_STAT_AREA] < MIN_PART * m.size:
            break
        piece = (comp == k).astype(np.uint8)
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(piece, cnts, -1, 1, -1)          # fill holes inside a piece
        keep |= piece
        x, y, bw, bh = stats[k, :4]
        parts.append(tuple(int(v / s) for v in (x, y, x + bw, y + bh)))
    mask = cv2.resize(keep, (Wd, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, parts


def changed_region(before, after, pad=20):
    """Bounding box around everything new (all pieces). None if nothing changed."""
    _, parts = change_mask(before, after)
    if not parts:
        return None
    w, h = after.size
    x0, y0 = min(p[0] for p in parts), min(p[1] for p in parts)
    x1, y1 = max(p[2] for p in parts), max(p[3] for p in parts)
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def cutout(after, mask, box, pad=16, fill=(128, 128, 128), normalize=False):
    """Crop of one new piece with the old waste around it greyed out, so the classifier only sees
    what was just thrown in. normalize: undo a warm lamp's colour cast and lift a dark piece to normal
    brightness. Off by default: on the dark test pairs it lowered F1 from 27 to 13 - a lamp in the bin
    works, fixing the photo afterwards does not."""
    import cv2
    w, h = after.size
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)
    full = np.asarray(after.convert("RGB")).astype(np.float32)
    img = full[y0:y1, x0:x1]
    m = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((9, 9), np.uint8))
    if normalize and full.mean() < 90:                            # only dark photos (bin lamp)
        lit = full.mean(2) > 20                                   # ignore black corners
        means = full[lit].mean(0) if lit.any() else full.reshape(-1, 3).mean(0)
        img = img * np.clip(means.mean() / np.maximum(means, 1), 0.6, 1.8)   # grey-world white balance
        piece = img[m > 0].mean() if (m > 0).any() else img.mean()
        if piece < 110:                                           # dark piece -> gamma up to ~120
            g = np.log(120 / 255) / np.log(max(piece, 4) / 255)
            img = 255 * (np.clip(img, 0, 255) / 255) ** g
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3)[..., None]
    return Image.fromarray(np.clip(img * mf + np.array(fill) * (1 - mf), 0, 255).astype(np.uint8))


def _classify(m, img):
    import torch
    with torch.no_grad():
        inp = m["proc"](images=img, return_tensors="pt").to(m["device"])
        inp["pixel_values"] = inp["pixel_values"].to(m["model"].dtype)
        e = m["model"].get_image_features(**inp).float()
        e = e / e.norm(dim=-1, keepdim=True)
        logits = e @ m["text"].T * m["model"].logit_scale.exp().float() + m["model"].logit_bias.float()
        return torch.sigmoid(logits)[0].cpu().tolist()


def classify(img):
    """-> (foods above the threshold, best first; {food: score}; top-3 [(food, score)])."""
    m = load_model()
    probs = _classify(m, img)
    ranked = sorted(zip(m["labels"], probs), key=lambda x: -x[1])
    foods = [l for l, p in ranked if p >= m["threshold"]]   # several foods per tip are normal
    return foods, {l: round(p, 3) for l, p in ranked if p >= m["threshold"]}, ranked[:3]


MAX_PARTS = 3     # classify at most this many separate new pieces per disposal
PIECE_REL = 0.5   # a piece is usually one food: keep labels scoring >= half of its best label...
PIECE_MAX = 2     # ...and at most this many per piece


def recognise(before, after, save_dir=None, tag=None):
    """before = photo when the lid opened, after = photo once the food is in.
    Each new piece is cut out on its own and classified; the result is the union of the pieces."""
    mask, parts = change_mask(before, after)
    if not parts:
        return {"foods": [], "food": "unknown", "food_conf": {}, "top3": [], "image_id": "",
                "parts": [], "box": None, "model": load_model()["name"]}
    pieces, conf, top = [], {}, {}
    for k, box in enumerate(parts[:MAX_PARTS]):
        crop = cutout(after, mask, box)
        foods, c, top3 = classify(crop)
        if foods:
            best = c[foods[0]]
            foods = [f for f in foods if c[f] >= PIECE_REL * best][:PIECE_MAX]
        pieces.append({"box": box, "foods": foods, "top3": top3})
        for f in foods:
            conf[f] = max(conf.get(f, 0), c[f])
        for f, p in top3:
            top[f] = max(top.get(f, 0), p)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            crop.save(os.path.join(save_dir, f"{tag or int(time.time())}_{k}.jpg"), quality=80)
    foods = sorted(conf, key=lambda f: -conf[f])
    x0, y0 = min(p[0] for p in parts), min(p[1] for p in parts)
    x1, y1 = max(p[2] for p in parts), max(p[3] for p in parts)
    return {"foods": foods,                                   # [] -> "unknown" (honest beats wrong)
            "food": foods[0] if foods else "unknown",         # single best, for simple displays
            "food_conf": {f: conf[f] for f in foods},
            "top3": sorted(top.items(), key=lambda x: -x[1])[:3],
            "image_id": f"{tag}_0.jpg" if save_dir and tag else "",
            "parts": pieces, "box": (x0, y0, x1, y1), "model": _m["name"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", required=True, help="e.g. http://192.168.43.1:8080")
    ap.add_argument("--model", choices=list(MODELS), help="default: so400m if the GPU has room, else base")
    a = ap.parse_args()
    print("loading model (first time downloads it)...")
    load_model(a.model)
    crops = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crops")
    while True:
        input("\nEmpty/current tray - press Enter for BEFORE photo")
        before = snapshot(a.phone)
        input("Put the item on the tray - press Enter for AFTER photo")
        after = snapshot(a.phone)
        t = time.time()
        r = recognise(before, after, save_dir=crops)
        found = ", ".join(f"{f} ({r['food_conf'][f]:.0%})" for f in r["foods"]) or "unknown"
        print(f"-> {found}   top3: " + ", ".join(f"{l} {p:.0%}" for l, p in r["top3"]) +
              f"   box={r['box']}  {1000 * (time.time() - t):.0f} ms  saved {r['image_id']}")


if __name__ == "__main__":
    main()
