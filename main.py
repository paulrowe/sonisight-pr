import numpy as np, base64
import io
import re
import os
import json
# using  Gemini API for the AI (API key in .env)
import google.generativeai as genai
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2


from dotenv import load_dotenv
load_dotenv()  # loads environment variables (mainly for the Gemini API key)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# choosing Gemini vers
_GEMINI_MODEL = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0, 
        "response_mime_type": "application/json"  # asking for JSON to not parse long text
    },
)

app = FastAPI(title="Ultrasound Analyzer Prototype")

from fastapi.staticfiles import StaticFiles
# static sample images for the frontend so testing without uploading own
app.mount("/samples-static", StaticFiles(directory="samples"), name="samples")

# built in sample images for those who don't want to download their own
SAMPLES = {
    "normal": [
        "samples/normal/normal_01.png",
        "samples/normal/normal_02.png",
        "samples/normal/normal_03.png",
        "samples/normal/normal_04.png",
    ],
    "suspicious": [
        "samples/suspicious/suspicious_01.png",
        "samples/suspicious/suspicious_02.png",
        "samples/suspicious/suspicious_03.png",
        "samples/suspicious/suspicious_04.png",
    ],
}
# so the frontend can just send "normal_01.png"
SAMPLE_NAME_TO_PATH = {
    os.path.basename(p): p
    for cat in SAMPLES.values()
    for p in cat
}

# CORS setup so React or other frontends can call this API
#  Vercel preview URLs and the main deployed domain
origins = [
    "http://localhost:5173",
    "https://sonisight-pr.vercel.app",
    "https://www.sonisight.app",  # for custom domains
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],
    allow_origin_regex=r"^https://sonisight-pr(-[a-z0-9-]+)?\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenCV helpers

# step 0: preprocessing
def preprocess(bgr: np.ndarray) -> np.ndarray:
    # converts BGR image to grayscale version thats more cleaned up
    # grayscale, median blur for speckle, CLAHE for contrast
    h, w = bgr.shape[:2]
    m = int(0.05 * min(h, w))  # 5% margin 
    bgr = bgr[m:h-m, m:w-m] if h > 2*m and w > 2*m else bgr

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    return g

# helper for converting a PIL image to PNG bytes (for sending to Gemini)
def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# step 1: ROI (region of interest)
# uses Gemini to find the main mass/lesion in the image by asking for a bounding box
# if it can't find anything falls back to local saliency detector
def get_roi_from_gemini(pil_img: Image.Image) -> list | None:
    # ask Gem for bounding box [x,y,w,h] of the most prominent lesion
    # returns None if no mass is found or on any API error
    prompt = (
        "You are a medical imaging specialist. "
        "Find the single most prominent mass/lesion in the ultrasound.\n\n"
        "If you find a clear mass:\n"
        "Respond ONLY with a bounding box like: [x, y, w, h]\n"
        "If you do not find any clear mass: respond ONLY with: None\n"
        "No extra text.\n"
        "Coordinates should refer to the pixel grid of the provided image."
    )

    try:
        img_bytes = _pil_to_png_bytes(pil_img)
        # reusing  already-configured Gem model (gemini-2.5-flash) but request text back
        resp = _GEMINI_MODEL.generate_content(
            [
                prompt,
                {"mime_type": "image/png", "data": img_bytes},
            ],
            generation_config={"response_mime_type": "text/plain", "temperature": 0},
        )
        text = (resp.text or "").strip().replace("`", "")
        # fast path for "None"
        if text.lower() == "none":
            print("AI-ROI: None")
            return None

        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
        if not m:
            print(f"AI-ROI: Could not parse ROI from: {text!r}")
            return None

        x, y, w, h = map(int, m.groups())
        if w <= 0 or h <= 0:
            return None
        print(f"AI-ROI Found: {[x,y,w,h]}")
        return [x, y, w, h]

    except Exception as e:
        # typical: 404 model name, 401 key missing, or quota errors
        print(f"Error in get_roi_from_gemini: {e}")
        return None
    
# step 1 cont: local fallback ROI using saliency 
def local_roi_from_saliency(bgr: np.ndarray) -> list | None:
    # if Gem can't fiind a mass, thos tries to pick out a likely lesion region using a saliency map
    # it's looking for dark, blob-like areas that stand out from the background
    gray = preprocess(bgr)
    if gray is None or gray.size == 0:
        return None

    # saliency: combine darkness and "blobness" (edges)
    dark = 1.0 - _normalize01(gray.astype(np.float32))
    g_blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    log = _normalize01(np.abs(cv2.Laplacian(g_blur, cv2.CV_32F, ksize=3)))
    S = _normalize01(0.7 * dark + 0.3 * log)

    # adaptive threshold: if the whole image is super smooth, gets stricter (less likely to call something a mass)
    global_std = float(gray.std())
    p = 94 if global_std >= 22.0 else 96
    T = float(np.percentile(S, p))
    mask = (S >= T).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H, W = gray.shape
    A_img = H * W

    def ok(c):
        x, y, w, h = cv2.boundingRect(c)
        # ignore anything touching the image borders (probes, labels, etc.)
        if x <= 2 or y <= 2 or (x + w) >= W - 2 or (y + h) >= H - 2:
            return False
        # ignore tiny regions (probably noise)
        if (w * h) < 0.003 * A_img:
            return False
        # require at least a little contrast (so there's not always find “some blob”)
        tmp = np.zeros((H, W), np.uint8)
        cv2.drawContours(tmp, [c], -1, 255, -1)
        ring = cv2.dilate(tmp, np.ones((7, 7), np.uint8))
        ring = cv2.subtract(ring, tmp)
        inside = gray[tmp > 0]
        outside = gray[ring > 0]
        if inside.size == 0 or outside.size == 0:
            return False
        mean_in = float(inside.mean())
        mean_out = float(outside.mean())
        contrast_out_in = max(0.0, (mean_out - mean_in) / (mean_out + 1e-6))
        if contrast_out_in < 0.06:  # require at least modest hypoechogenicity
            return False
        return True

    cnts = [c for c in cnts if ok(c)]
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return [int(x), int(y), int(w), int(h)]

# step 2: segement likely lesion candidate from ROI 
def segment_candidate(gray: np.ndarray):
    # try to find best candidate lesion in a grayscale ROI
    # uses Otsu thresholding (since most lesion are darker), then picks the best contour using a hand-tuned scoring
    # Otsu works well for these images to keep inverse because lesions tend to be darker
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # better morphology to clean up speckle and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    H, W = gray.shape
    A_img = H * W

    def contour_score(c):
        # hand-tuned scoring function that rewards compact, dark, solid blobs with clean margins
        area = cv2.contourArea(c)
        if area <= 0:
            return -1e9, None  # invalid
        x, y, w, h = cv2.boundingRect(c)
        touches_border = (x <= 0 or y <= 0 or (x + w) >= W - 1 or (y + h) >= H - 1)
        if touches_border:
            return -1e9, None
        area_frac = area / A_img
        # allow a wide range of areas, but ignore tiny or enormous blobs
        if area_frac < 0.005 or area_frac > 0.45:
            return -1e9, None
        # solidity and aspect ratio filters (avoid weirdly stringy or hollow shapes)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        ar = max(w / (h + 1e-6), h / (w + 1e-6))
        if ar > 3.5 or solidity < 0.30:
            return -1e9, None
        # margin analysis: draw a ring just outside the contour and look for edge density
        tmp_mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(tmp_mask, [c], -1, 255, -1)
        ring = cv2.dilate(tmp_mask, np.ones((3, 3), np.uint8))
        ring = cv2.subtract(ring, tmp_mask)
        edges = cv2.Canny(gray, 30, 100)
        ring_edges = (edges[ring > 0] > 0).mean() if ring.sum() > 0 else 0.0
        # contrast: mean intensity outside ring vs inside mask (lesions darker = higher contrast_out_in)
        inside_vals = gray[tmp_mask > 0]
        if inside_vals.size == 0:
            return -1e9, None
        outside_band = cv2.dilate(ring, np.ones((7, 7), np.uint8))  # a bit wider context
        outside_vals = gray[outside_band > 0]
        if outside_vals.size == 0:
            return -1e9, None
        mean_in = float(inside_vals.mean())
        mean_out = float(outside_vals.mean())
        contrast_out_in = max(0.0, (mean_out - mean_in) / (mean_out + 1e-6))  # 0..1 approx
        # circularity is basically: how round is it?
        perim = cv2.arcLength(c, True)
        circularity = (4.0 * np.pi * area) / (perim * perim + 1e-6)
        # score: prefer round, solid, dark, and smooth. penalize noisy edges
        score = (
            1.0 * circularity +
            1.2 * contrast_out_in +
            0.6 * solidity -
            0.8 * max(0.0, ring_edges - 0.25)  # penalize only when very high noise
        )
        return score, (tmp_mask, ring, ring_edges, circularity, contrast_out_in, solidity, area, area_frac)

    best = (-1e9, None)
    for c in cnts:
        s, aux = contour_score(c)
        if s > best[0]:
            best = (s, (c, aux))

    if best[0] <= -1e8:
        return None, None

    c, aux = best[1]
    tmp_mask, ring, ring_edges, circularity, contrast_out_in, solidity, area, area_frac = aux

    mask = np.zeros_like(th)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return c, mask

#
# step 2b: saliency-based fallback segmentation 
#
def saliency_candidate(gray: np.ndarray, p: int = 90):
    # build a saliency map (same as overlay) and return the best contour + mask
    # returns (cnt, mask) or (none, none) if nothing reliable is found
    if gray is None or gray.size == 0:
        return None, None

    # re-using the same saliency recipe as in the ROI finder
    dark = 1.0 - _normalize01(gray.astype(np.float32))
    g_blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    log = _normalize01(np.abs(cv2.Laplacian(g_blur, cv2.CV_32F, ksize=3)))
    edges = cv2.Canny(gray, 30, 100).astype(np.float32)
    edges = _normalize01(cv2.GaussianBlur(edges, (5, 5), 0))
    S = _normalize01(0.6 * dark + 0.3 * log + 0.1 * edges)

    # threshold by percentile
    T = float(np.percentile(S, p))
    th = (S >= T).astype(np.uint8) * 255

    # clean up and remove tiny speckle
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    H, W = gray.shape
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    # filter candidates: ignore tiny or border-touching regions
    A_img = H * W
    kept = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or (x + w) >= W - 1 or (y + h) >= H - 1:
            continue
        area = cv2.contourArea(c)
        if area < 0.003 * A_img:
            continue
        kept.append(c)
    if not kept:
        return None, None

    c = max(kept, key=cv2.contourArea)
    mask = np.zeros_like(gray, np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return c, mask


#
# step 3: extract descriptors from a candidate contour 
#
def descriptors_from_cnt(gray: np.ndarray, cnt, mask) -> Dict:
    # compute same features used in extract_descriptors but from a given contour/mask
    # returns a descriptor dict with mass_present=True (unless it failes no-mass gate)
    # shape
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularity = (4.0 * np.pi * area) / (perim * perim + 1e-6)
    shape = "round" if circularity >= 0.73 else ("oval" if circularity >= 0.60 else "irregular")

    # margins
    edges = cv2.Canny(gray, 30, 100)
    dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    ring = cv2.subtract(dil, mask)
    ring_edges = (edges[ring > 0] > 0).mean() if ring.sum() > 0 else 0.0
    margins = "smooth" if ring_edges < 0.18 else ("lobulated" if ring_edges < 0.38 else "spiculated")

    # texture
    vals = gray[mask > 0]
    std = float(vals.std()) if vals.size else 0.0
    texture = "homogeneous" if std < 18 else ("mixed" if std < 33 else "heterogeneous")

    # contrast
    outside_band = cv2.dilate(ring, np.ones((7, 7), np.uint8))
    mean_in = float(vals.mean()) if vals.size else 0.0
    mean_out = float(gray[outside_band > 0].mean()) if outside_band.sum() > 0 else mean_in
    contrast_out_in = max(0.0, (mean_out - mean_in) / (mean_out + 1e-6))

    H, W = gray.shape
    area_frac = area / (H * W)

    # use  same “no-mass” gate as in the main extractor: if it’s tiny, low-contrast, or super noisy, don’t call it a mass
    if ((area_frac < 0.006 and contrast_out_in < 0.08) or (ring_edges > 0.36)):
        return {
            "mass_present": False,
            "img_quality": "ok",
            "shape": "none",
            "margins": "none",
            "texture": "homogeneous",
        }

    # for cysts: if it’s very round, smooth, dark, and homogeneous, call it cyst-like
    looks_cystic = (circularity >= 0.78) and (ring_edges < 0.10) and (contrast_out_in >= 0.22) and (std < 16.0)

    return {
        "mass_present": True,
        "img_quality": "ok",
        "shape": "round" if looks_cystic else shape,
        "margins": "smooth" if looks_cystic else margins,
        "texture": texture,
        "area": area,
        "circularity": circularity,
        "edge_density": float(ring_edges),
        "intensity_std": std,
        "contrast_out_in": round(contrast_out_in, 3),
        "cyst_like": bool(looks_cystic),
        "detection_method": "saliency-fallback"
    }

def _normalize01(x):
    # normalizes an array to 0..1. used everywhere for contrast/saliency stuff
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def extract_descriptors(bgr: np.ndarray) -> Dict:
    gray = preprocess(bgr)
    cnt, mask = segment_candidate(gray)

    if cnt is None:
        # if no candidate found, just say “no mass” and fill in defaults
        return {
            "mass_present": False,
            "img_quality": "unknown",
            "shape": "none",
            "margins": "none",
            "texture": "homogeneous",
        }

    # shape 
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    circularity = (4.0 * np.pi * area) / (perim * perim + 1e-6)
    shape = "round" if circularity >= 0.73 else ("oval" if circularity >= 0.60 else "irregular")

    # margins / edges 
    edges = cv2.Canny(gray, 30, 100)
    dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    ring = cv2.subtract(dil, mask)
    ring_edges = (edges[ring > 0] > 0).mean() if ring.sum() > 0 else 0.0
    margins = "smooth" if ring_edges < 0.16 else ("lobulated" if ring_edges < 0.26 else "spiculated")

    # texture 
    vals = gray[mask > 0]
    std = float(vals.std()) if vals.size else 0.0
    texture = "homogeneous" if std < 18 else ("mixed" if std < 33 else "heterogeneous")

    # contrast inside vs outside
    outside_band = cv2.dilate(ring, np.ones((7, 7), np.uint8))
    mean_in = float(vals.mean()) if vals.size else 0.0
    mean_out = float(gray[outside_band > 0].mean()) if outside_band.sum() > 0 else mean_in
    contrast_out_in = max(0.0, (mean_out - mean_in) / (mean_out + 1e-6))

    H, W = gray.shape
    area_frac = area / (H * W)

    # “no mass” gate: if it’s tiny & low contrast or the edge is super noisy, don’t call it a mass
    if ((area_frac < 0.006 and contrast_out_in < 0.08) or (ring_edges > 0.36)):
        return {
            "mass_present": False,
            "img_quality": "ok",
            "shape": "none",
            "margins": "none",
            "texture": "homogeneous",
        }

    # benign cyst: very round, smooth, and dark masses often look like cysts
    looks_cystic = (circularity >= 0.75) and (ring_edges < 0.10) and (contrast_out_in >= 0.25)

    return {
        "mass_present": True,
        "img_quality": "ok",
        "shape": "round" if looks_cystic else shape,
        "margins": "smooth" if looks_cystic else margins,
        "texture": texture,
        "area": area,
        "circularity": circularity,
        "edge_density": float(ring_edges),
        "intensity_std": std,
        "contrast_out_in": round(contrast_out_in, 3),
        "cyst_like": bool(looks_cystic),
    }

#
# step 4: build prompt for Gemini to get probabilities 
#
def build_gemini_prompt(descriptors: Dict) -> str:
    # ask for normal vs sus probabilites (sus = any mass likely needing follow-up)
    # irregular/spiculated/heterogeneous or unclear
    mass_present = descriptors.get("mass_present", False)
    shape = descriptors.get("shape", "none")
    margins = descriptors.get("margins", "none")
    texture = descriptors.get("texture", "homogeneous")
    circ = float(descriptors.get("circularity", 0.0))
    edged = float(descriptors.get("edge_density", 0.0))
    istd = float(descriptors.get("intensity_std", 0.0))
    imgq = descriptors.get("img_quality", "unknown")
    contrast = float(descriptors.get("contrast_out_in", 0.0))
    cyst_like = bool(descriptors.get("cyst_like", False))
    indet = bool(descriptors.get("indeterminate_roi", False))

    # explicit prompt tells Gem how to triage the descriptors
    return f"""
You are a triage assistant for breast ultrasound images.
OUTPUT JSON ONLY with this schema:
{{
  "normal": <float 0..1>,
  "suspicious": <float 0..1>,
  "rationale": "<<= 2 concise sentences>"
}}
Rules:
- normal + suspicious must sum to 1.0 (±1e-6).
- “Suspicious” means features likely warranting further evaluation (e.g., irregular/spiculated/heterogeneous or unclear).
- Prioritize sensitivity (err slightly toward suspicious when descriptors are unclear).
- Keep wording non-diagnostic. If indeterminate_roi is true, avoid claiming "no mass"; prefer "no clear region" or "subtle."

DESCRIPTORS:
mass_present: {str(mass_present).lower()}
shape: {shape}
margins: {margins}
texture: {texture}
circularity: {circ:.3f}
edge_density: {edged:.3f}
intensity_std: {istd:.3f}
contrast_out_in: {contrast:.3f}
cyst_like: {str(cyst_like).lower()}
indeterminate_roi: {str(indet).lower()}
image_quality: {imgq}

EXAMPLES:
# clearly normal background
{{"normal":0.96,"suspicious":0.04,"rationale":"No discrete mass; background appears homogeneous."}}

# irregular + spiculated
{{"normal":0.03,"suspicious":0.97,"rationale":"Irregular shape with spiculated margins suggests follow-up."}}

# round, smooth, cyst-like
{{"normal":0.70,"suspicious":0.30,"rationale":"Round and smooth with high internal homogeneity suggests benign appearance."}}

JSON:
""".strip()

#
# step 5: query Gemini for normal/suspicious triage 
#
def gemini_infer(descriptors: Dict) -> Dict:
    prompt = build_gemini_prompt(descriptors)
    try:
        resp = _GEMINI_MODEL.generate_content(prompt)
        data = json.loads(resp.text.strip())
        n = float(data.get("normal", 0.0))
        s = float(data.get("suspicious", 0.0))
        r = data.get("rationale", "")
        total = n + s
        if total <= 0:
            raise ValueError("bad probs")
        n, s = n/total, s/total
        return {"probabilities": {"normal": round(n, 3), "suspicious": round(s, 3)}, "rationale": r}
    except Exception:
        # if Gemini fails for any reason fall back to a simple rules-based triage
        if descriptors.get("indeterminate_roi"):
            return {"probabilities": {"normal": 0.75, "suspicious": 0.25},
                    "rationale": "No clearly delineated region, but features are subtle; recommend routine review."}
        if not descriptors.get("mass_present", False):
            return {"probabilities": {"normal": 0.95, "suspicious": 0.05},
                    "rationale": "No discrete region detected; background appears homogeneous."}
        score = 0.0
        if descriptors.get("shape") == "irregular": score += 0.4
        if descriptors.get("margins") == "spiculated": score += 0.4
        if descriptors.get("texture") == "heterogeneous": score += 0.2
        if descriptors.get("indeterminate_roi"): score += 0.15
        score = max(0.05, min(0.98, score))
        return {"probabilities": {"normal": round(1 - score, 3), "suspicious": round(score, 3)},
                "rationale": "Triage based on irregularity/margins/texture."}

# API endpoints

# health check endpoint (for uptime monitoring or just to see if the server is alive) 
@app.get("/health")
def health():
    return {"ok": True}


# list available sample images for the frontend to display 
@app.get("/samples")
def list_samples():
    # return the available built-in sample images grouped by label
    return {
        "normal": [os.path.basename(p) for p in SAMPLES["normal"]],
        "suspicious": [os.path.basename(p) for p in SAMPLES["suspicious"]],
    }

from fastapi import HTTPException, Query
from typing import Optional, Literal

# main prediction endpoint 
# takes an uploaded image or a sample, runs the whole pipeline, and returns descriptors, triage, and overlay
@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    source: Literal["live","sample"] = Query("live"),
    name: Optional[str] = Query(None),
):
    # step 0: get the image
    # either from a live upload or from the built-in samples
    if source == "live":
        if file is None:
            raise HTTPException(status_code=400, detail="Upload a file when source=live.")
        data = await file.read()
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=415, detail="Unsupported image format. Please upload PNG or JPG.")
    else:  # source == "sample"
        if not name:
            raise HTTPException(status_code=400, detail="Provide ?name=<sample_file.png> when source=sample.")
        path = SAMPLE_NAME_TO_PATH.get(name)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Sample '{name}' not found.")
        with open(path, "rb") as f:
            data = f.read()
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to load sample image.")

    # convert PIL image to OpenCV BGR (since all the OpenCV  expects BGR)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # step 1: try to find the ROI using Gemini 
    box = get_roi_from_gemini(pil)

    # if Gemini can’t find anything, fall back to local saliency
    if box is None:
        box = local_roi_from_saliency(bgr)

    # step 2: if still nothing, use conservative descriptors (no mass found)
    if box is None:
        descriptors = {
            "mass_present": False,
            "img_quality": "ok",
            "shape": "none",
            "margins": "none",
            "texture": "homogeneous",
        }
    else:
        x, y, w, h = box
        H, W = bgr.shape[:2]
        # clamp to image bounds (sometimes AI or saliency gives slightly out-of-bounds boxes)
        if w <= 0 or h <= 0:
            box = None
        else:
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = min(w, W - x)
            h = min(h, H - y)
            if w <= 2 or h <= 2:
                box = None

        if box is None:
            descriptors = {
                "mass_present": False,
                "img_quality": "ok",
                "shape": "none",
                "margins": "none",
                "texture": "homogeneous",
            }
        else:
            # adds a little padding to the ROI for safety ( masses are right up against the box edge sometimes)
            pad_w = int(w * 0.10); pad_h = int(h * 0.10)
            x1 = max(0, x - pad_w); y1 = max(0, y - pad_h)
            x2 = min(W, x + w + pad_w); y2 = min(H, y + h + pad_h)
            roi = bgr[y1:y2, x1:x2]

            if roi.size == 0:
                descriptors = {
                    "mass_present": False, "img_quality": "ok",
                    "shape": "none", "margins": "none", "texture": "homogeneous"
                }
            else:
                descriptors = extract_descriptors(roi)

                if not descriptors.get("mass_present", False):
                    # if  main segmentation didn’t find anything, try a more aggressive saliency-based segmentation inside the ROI
                    gray_roi = preprocess(roi)
                    cnt_s, mask_s = saliency_candidate(gray_roi, p=90)  # start with 90th percentile
                    if cnt_s is None:
                        # try  lower threshold if nothing found 
                        cnt_s, mask_s = saliency_candidate(gray_roi, p=85)

                    if cnt_s is not None and mask_s is not None:
                        # compute features from the saliency contour
                        desc2 = descriptors_from_cnt(gray_roi, cnt_s, mask_s)
                        if desc2.get("mass_present", False):
                            descriptors = desc2
                            descriptors["detection_method"] = "saliency-fallback"
                        else:
                            descriptors["indeterminate_roi"] = True
                    else:
                        # still nothing: keep it as no-mass, but mark indeterminate
                        descriptors["indeterminate_roi"] = True

    # step 4: run Gemini inference on descriptors to get probabilities and rationale 
    try:
        result = gemini_infer(descriptors)
        probs = result["probabilities"]
        rationale = result["rationale"]
    except Exception:
        # if all else fails just show fallback probabilities
        probs = {"normal": 0.33, "benign": 0.33, "malignant": 0.34}
        rationale = "AI rationale unavailable; showing fallback probabilities."

    # step 5: draw overlays for the frontend 
    overlay = bgr.copy()
    # only show a box if Gemini classifies as “suspicious” (otherwise hide to avoid false positives)
    if box is not None and probs.get("suspicious", 0) >= 0.35:
        x, y, w, h = box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # hide ROI if classified as normal
        box = None

    ok, png = cv2.imencode(".png", overlay)
    overlay_b64 = base64.b64encode(png.tobytes()).decode("ascii") if ok else None

    try:
        gray2 = preprocess(bgr)
        cnt, _mask = segment_candidate(gray2)
        if cnt is not None and descriptors.get("mass_present"):
            # You could draw the contour here if you want!
            pass
    except Exception:
        pass

    return {
        "descriptors": descriptors,
        "probabilities": probs,
        "rationale": rationale,
        "overlay_png_base64": overlay_b64
    }