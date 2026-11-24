import numpy as np
import plotly.graph_objects as go
from PIL import Image

COLOR_MAPS = {"Viridis":"Viridis","Grayscale":"Gray","Turbo":"Turbo","Inferno":"Inferno","Plasma":"Plasma","Magma":"Magma"}

def colorize_depth(depth, color_mode="Viridis"):
    """Return PNG bytes for the selected depth color map."""
    import io
    arr=np.clip(np.asarray(depth,dtype=np.float32),0,1)
    try:
        import matplotlib.cm as cm
        cmap_name={"Grayscale":"gray","Turbo":"turbo"}.get(color_mode,color_mode.lower())
        rgb=(cm.get_cmap(cmap_name)(arr)[...,:3]*255).astype(np.uint8)
    except Exception:
        gray=(arr*255).astype(np.uint8); rgb=np.stack([gray]*3,axis=-1)
    buf=io.BytesIO(); Image.fromarray(rgb).save(buf,format="PNG"); return buf.getvalue()

def create_3d_surface(depth, color_mode="Viridis", height_scale=1.0, max_points=150):
    arr=np.asarray(depth,dtype=np.float32)
    step=max(1,int(max(arr.shape)/max_points))
    sampled=arr[::step,::step]
    h,w=sampled.shape
    fig=go.Figure(data=[go.Surface(x=np.arange(w),y=np.arange(h),z=sampled*height_scale,colorscale=COLOR_MAPS.get(color_mode,"Viridis"),colorbar={"title":"Relative depth"},hovertemplate="x=%{x}<br>y=%{y}<br>relative depth=%{z:.3f}<extra></extra>")])
    fig.update_layout(title="DepthWizard — Interactive 3D Relative-Depth Surface",scene={"xaxis_title":"Image column","yaxis_title":"Image row","zaxis_title":"Scaled relative depth","aspectmode":"auto"},height=680,margin={"l":0,"r":0,"t":55,"b":0})
    return fig
