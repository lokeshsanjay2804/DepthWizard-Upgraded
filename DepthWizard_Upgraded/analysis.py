import numpy as np

def depth_summary(depth):
    a=np.asarray(depth,dtype=np.float32); a=a[np.isfinite(a)]
    if not a.size: raise ValueError("Depth map contains no finite values")
    return {"min":float(a.min()),"max":float(a.max()),"mean":float(a.mean()),"std":float(a.std()),"median":float(np.median(a))}

def region_statistics(depth,x1,y1,x2,y2):
    region=np.asarray(depth,dtype=np.float32)[y1:y2,x1:x2]
    region=region[np.isfinite(region)]
    if not region.size: raise ValueError("Selected region has no valid depth values")
    return {"min":float(region.min()),"max":float(region.max()),"mean":float(region.mean()),"std":float(region.std()),"pixels":int(region.size)}
