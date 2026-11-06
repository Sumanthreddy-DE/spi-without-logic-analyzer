# food_cam – what food got thrown away?

Camera over the bin opening. A hand comes in with food → motion starts a rolling
frame buffer → motion stops → the 3 sharpest in-hand frames + a before/after crop
of the bin go to Claude in one request → food class, confidence, description.

```
pip install opencv-python numpy
export ANTHROPIC_API_KEY=sk-ant-...            # Windows: set ANTHROPIC_API_KEY=...
python3 hand_cam.py --camera 0                 # USB/laptop webcam
python3 hand_cam.py --camera http://PHONE_IP:8080/video   # phone "IP Webcam" app
python3 hand_cam.py --camera 0 --offline       # test motion/frames without API
```

Keys: `r` draw bin-opening region (do first) · `SPACE` force snap · `b` new before-frame · `d` motion debug · `q` quit

**Dashboard: http://localhost:8001/** (live camera, last throw with food name + photos, totals, history)

HTTP: `:8001/history` · `:8001/last` (latest JSON) · `:8001/status` · `:8001/img/<file>` · `:8001/snap?weight_g=123`

Output: `captures/throws.csv` (with `food` = specific name like apple/banana, `item` = category) + `captures/throw_XXX_{hand1..3,after,collage}.jpg`

Tuning (top of `hand_cam.py`): `MOTION_START`, `MOTION_STOP`, `QUIET_S`, `MODEL`
(`claude-sonnet-5-5` default, `claude-haiku-5-5` faster/cheaper).

`food_cam.py` = simpler before/after-only version (SPACE or HTTP trigger).
