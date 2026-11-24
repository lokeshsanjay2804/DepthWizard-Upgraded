import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the pretrained MiDaS Small model and its matching transform."""
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(DEVICE).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return model, transforms.small_transform

def open_uploaded_image(uploaded_file):
    """Read common images; use rasterio for georeferenced TIFF when available."""
    name = getattr(uploaded_file, "name", "image.png").lower()
    if name.endswith((".tif", ".tiff")):
        try:
            import rasterio
            uploaded_file.seek(0)
            with rasterio.open(uploaded_file) as src:
                if src.count >= 3:
                    arr = np.stack([src.read(i) for i in (1,2,3)], axis=-1)
                else:
                    band = src.read(1)
                    arr = np.repeat(band[...,None],3,axis=-1)
                arr = _to_uint8_rgb(arr)
                meta = {"transform":src.transform,"crs":str(src.crs) if src.crs else None,"width":src.width,"height":src.height}
                return Image.fromarray(arr,"RGB"), meta
        except ImportError:
            pass
        except Exception:
            # Fall back to Pillow for ordinary TIFFs without readable rasterio metadata.
            uploaded_file.seek(0)
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    return image, None

def _to_uint8_rgb(array):
    array = np.asarray(array, dtype=np.float32)
    out = np.zeros_like(array, dtype=np.uint8)
    for band in range(3):
        layer = array[...,band]
        valid = np.isfinite(layer)
        if not valid.any():
            continue
        lo, hi = np.percentile(layer[valid], [2,98])
        scaled = (layer-lo)/max(float(hi-lo),1e-8)
        out[...,band] = (np.clip(scaled,0,1)*255).astype(np.uint8)
    return out

def predict_depth(image, model, transform):
    image = image.convert("RGB")
    input_batch = transform(np.asarray(image)).to(DEVICE)
    with torch.inference_mode():
        prediction = model(input_batch)
        prediction = F.interpolate(prediction.unsqueeze(1), size=(image.height,image.width), mode="bicubic", align_corners=False).squeeze()
    depth = prediction.cpu().numpy().astype(np.float32)
    low, high = np.percentile(depth,[2,98])
    depth = np.clip(depth,low,high)
    return ((depth-low)/max(float(high-low),1e-8)).astype(np.float32)
