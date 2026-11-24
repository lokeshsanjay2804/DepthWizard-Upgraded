import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from depth_engine import load_model, predict_depth, open_uploaded_image
from visualization import create_3d_surface, colorize_depth
from analysis import region_statistics, depth_summary
from geospatial import read_dem, calibrate_to_dem, export_geotiff
from reporting import build_pdf_report

def _npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()

st.set_page_config(page_title="DepthWizard Upgraded", page_icon="🌍", layout="wide")
st.title("🌍 DepthWizard | AI Depth & 3D Explorer")
st.caption("Single-image relative depth estimation • 3D exploration • region analysis • optional DEM comparison")

@st.cache_resource(show_spinner="Loading pretrained MiDaS Small model...")
def get_model():
    return load_model()

with st.sidebar:
    st.header("Processing settings")
    color_mode = st.selectbox("Depth-map color mode", ["Viridis", "Grayscale", "Turbo", "Inferno", "Plasma", "Magma"])
    height_scale = st.slider("3D vertical exaggeration", 0.2, 4.0, 1.0, 0.1)
    max_points = st.slider("3D surface resolution", 60, 240, 150, 10)
    st.divider()
    st.caption("MiDaS outputs relative depth only. It does not directly measure metres above sea level or building height.")

files = st.file_uploader("Upload one or more RGB images", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)
if files:
    if len(files) > 12:
        st.error("Please upload 12 images or fewer per batch to avoid exhausting memory.")
        st.stop()
    names = [f.name for f in files]
    selected = st.selectbox("Choose image to inspect", names)
    uploaded = next(f for f in files if f.name == selected)
    try:
        rgb, geo_meta = open_uploaded_image(uploaded)
    except Exception as exc:
        st.error(f"Could not read {selected}: {exc}")
        st.stop()
    st.subheader("Input image")
    st.image(rgb, caption=selected, width="stretch")
    if geo_meta and geo_meta.get("crs"):
        st.info(f"Georeferenced image detected • CRS: {geo_meta['crs']}")
    elif selected.lower().endswith((".tif", ".tiff")):
        st.info("TIFF opened as an image. No usable georeferencing metadata was detected.")

    run = st.button("Generate / update depth maps", type="primary")
    if run:
        model, transform = get_model()
        results = {}
        progress = st.progress(0)
        status = st.empty()
        for idx, item in enumerate(files):
            try:
                img, meta = open_uploaded_image(item)
                status.write(f"Estimating depth: {item.name}")
                depth = predict_depth(img, model, transform)
                results[item.name] = {"image": img, "depth": depth, "geo": meta}
            except Exception as exc:
                st.warning(f"Skipped {item.name}: {exc}")
            progress.progress((idx + 1) / len(files))
        st.session_state["results"] = results
        st.success(f"Completed {len(results)} image(s).")

if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    current_name = selected if 'selected' in locals() and selected in results else next(iter(results))
    item = results[current_name]
    image, depth = item["image"], item["depth"]
    st.divider()
    st.header("Depth analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Original RGB image", width="stretch")
    with c2:
        st.image(colorize_depth(depth, color_mode), caption=f"Relative depth • {color_mode}", width="stretch")
    summary = depth_summary(depth)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Image width", f"{image.width}px")
    m2.metric("Image height", f"{image.height}px")
    m3.metric("Relative depth min", f"{summary['min']:.3f}")
    m4.metric("Relative depth max", f"{summary['max']:.3f}")

    st.subheader("3D Explorer")
    st.plotly_chart(create_3d_surface(depth, color_mode, height_scale, max_points), width="stretch", key="depth_surface")
    st.caption("Use the Plotly toolbar to rotate, pan, zoom, reset the camera, or save a snapshot. The vertical axis is normalized relative depth, not metres.")

    st.subheader("Region-based depth analysis")
    h,w = depth.shape
    a,b = st.columns(2)
    with a:
        x1 = st.slider("Left column", 0, max(0,w-2), 0)
        x2 = st.slider("Right column", 1, w, w)
    with b:
        y1 = st.slider("Top row", 0, max(0,h-2), 0)
        y2 = st.slider("Bottom row", 1, h, h)
    if x2 <= x1 or y2 <= y1:
        st.warning("Choose a region with positive width and height.")
    else:
        stats = region_statistics(depth, x1, y1, x2, y2)
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Mean relative depth", f"{stats['mean']:.3f}")
        r2.metric("Minimum", f"{stats['min']:.3f}")
        r3.metric("Maximum", f"{stats['max']:.3f}")
        r4.metric("Pixels", f"{stats['pixels']:,}")
        st.caption("Region coordinates are pixel coordinates in the predicted depth map.")

    st.subheader("DEM integration / reference comparison")
    st.write("Upload a reference DEM GeoTIFF covering the same scene. For a non-georeferenced RGB image, the DEM must already be aligned to the same pixel grid.")
    dem_file = st.file_uploader("Reference DEM (.tif / .tiff)", type=["tif", "tiff"], key="dem_upload")
    aligned = st.checkbox("I confirm the DEM and selected RGB image cover the same area and are pixel-aligned.")
    if dem_file and aligned:
        try:
            dem, dem_meta = read_dem(dem_file)
            if dem.shape != depth.shape:
                st.error(f"Grid mismatch: depth map is {depth.shape}, DEM is {dem.shape}. Reproject/resample both to the same grid before comparison.")
            else:
                calibrated, metrics = calibrate_to_dem(depth, dem)
                st.info("Affine calibration fitted to this reference DEM. The displayed residual metrics describe this fitted comparison, not independent validation accuracy.")
                q1,q2,q3 = st.columns(3)
                q1.metric("Fit MAE", f"{metrics['mae']:.2f} m")
                q2.metric("Fit RMSE", f"{metrics['rmse']:.2f} m")
                q3.metric("Valid DEM pixels", f"{metrics['count']:,}")
                st.image(colorize_depth(calibrated, "Turbo"), caption="DEM-calibrated relative-depth surface (metre scale fitted to this reference)", width="stretch")
                if st.button("Export calibrated GeoTIFF"):
                    if not dem_meta.get("transform") or not dem_meta.get("crs"):
                        st.error("The uploaded DEM has no valid georeferencing metadata; GeoTIFF export is unavailable.")
                    else:
                        data = export_geotiff(calibrated, dem_meta)
                        st.download_button("Download calibrated GeoTIFF", data, "depthwizard_calibrated_surface.tif", "image/tiff")
        except Exception as exc:
            st.error(f"Could not process DEM: {exc}. Install rasterio and use a valid single-band GeoTIFF.")
    elif dem_file and not aligned:
        st.warning("Confirm pixel alignment before comparing the arrays. Matching dimensions alone do not prove matching geography.")

    st.subheader("Downloads and reports")
    d1,d2 = st.columns(2)
    with d1:
        st.download_button("Download raw normalized depth (NPY)", _npy_bytes(depth), f"{Path(current_name).stem}_relative_depth.npy", "application/octet-stream")
    with d2:
        st.download_button("Download color depth map (PNG)", colorize_depth(depth, color_mode), f"{Path(current_name).stem}_depth_map.png", "image/png")
    pdf = build_pdf_report(current_name, image, depth, summary, stats if x2>x1 and y2>y1 else None)
    st.download_button("Download analysis report (PDF)", pdf, f"{Path(current_name).stem}_DepthWizard_report.pdf", "application/pdf")

    if len(results) > 1:
        st.subheader("Batch processing summary")
        rows = []
        for name, result in results.items():
            s = depth_summary(result['depth'])
            rows.append({"Image":name,"Width":result['image'].width,"Height":result['image'].height,"Mean relative depth":round(s['mean'],4)})
        st.dataframe(rows, width="stretch", hide_index=True)
        if st.button("Download all depth maps as ZIP"):
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
                for name, result in results.items():
                    buf=io.BytesIO(); Image.fromarray((np.clip(result['depth'],0,1)*255).astype(np.uint8)).save(buf,format="PNG")
                    z.writestr(f"{Path(name).stem}_depth.png",buf.getvalue())
            st.download_button("Save batch ZIP",bundle.getvalue(),"depthwizard_batch_outputs.zip","application/zip")

st.divider()
st.caption("DepthWizard Upgraded • MiDaS relative depth is not a survey-grade DEM/DSM. Use properly co-registered reference data and independent checkpoints for scientific accuracy assessment.")

