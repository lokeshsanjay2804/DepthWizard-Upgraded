import io
import numpy as np

def read_dem(uploaded_file):
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is not installed. Install the optional geospatial dependencies from requirements.txt") from exc
    uploaded_file.seek(0)
    with rasterio.open(uploaded_file) as src:
        dem=src.read(1).astype(np.float32)
        nodata=src.nodata
        if nodata is not None: dem[dem==nodata]=np.nan
        dem[~np.isfinite(dem)]=np.nan
        meta={"transform":src.transform,"crs":str(src.crs) if src.crs else None,"width":src.width,"height":src.height,"nodata":nodata,"profile":src.profile.copy()}
    if not np.isfinite(dem).any(): raise ValueError("DEM contains no valid elevation pixels")
    return dem,meta

def calibrate_to_dem(relative_depth, dem):
    """Fit z = a*relative_depth+b to valid DEM cells; metrics are in-sample fit residuals."""
    x=np.asarray(relative_depth,dtype=np.float64); y=np.asarray(dem,dtype=np.float64)
    mask=np.isfinite(x)&np.isfinite(y)
    if mask.sum()<3: raise ValueError("At least three overlapping valid pixels are required")
    A=np.column_stack([x[mask],np.ones(mask.sum())])
    coeff,_,_,_=np.linalg.lstsq(A,y[mask],rcond=None)
    calibrated=(coeff[0]*x+coeff[1]).astype(np.float32)
    residual=calibrated[mask]-y[mask]
    return calibrated,{"slope":float(coeff[0]),"intercept":float(coeff[1]),"mae":float(np.mean(np.abs(residual))),"rmse":float(np.sqrt(np.mean(residual**2))),"count":int(mask.sum())}

def export_geotiff(array, dem_meta):
    import rasterio
    profile=dem_meta["profile"].copy()
    profile.update(driver="GTiff",count=1,dtype="float32",compress="deflate",nodata=np.nan)
    buf=io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as dst: dst.write(np.asarray(array,dtype=np.float32),1)
        buf.write(mem.read())
    return buf.getvalue()
