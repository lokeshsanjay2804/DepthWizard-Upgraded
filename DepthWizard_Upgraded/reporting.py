import io
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

def build_pdf_report(filename, image, depth, summary, region=None):
    out=io.BytesIO(); c=canvas.Canvas(out,pagesize=A4); width,height=A4
    c.setTitle("DepthWizard Analysis Report")
    c.setFont("Helvetica-Bold",18); c.drawString(42,height-48,"DepthWizard — Analysis Report")
    c.setFont("Helvetica",10); c.drawString(42,height-68,f"Input image: {filename}")
    c.drawString(42,height-84,f"Image dimensions: {image.width} × {image.height} pixels")
    c.drawString(42,height-100,"Model: pretrained MiDaS Small • Output: normalized relative depth")
    c.drawString(42,height-116,"This report does not represent survey-grade elevation or metric height.")
    c.setFont("Helvetica-Bold",12); c.drawString(42,height-145,"Input image")
    img=image.copy(); img.thumbnail((240,180)); c.drawImage(ImageReader(img),42,height-340,width=240,height=180,preserveAspectRatio=True,anchor='c')
    depth_img=Image.fromarray((np.clip(depth,0,1)*255).astype(np.uint8)).convert("RGB"); depth_img.thumbnail((240,180))
    c.drawString(310,height-145,"Relative depth map")
    c.drawImage(ImageReader(depth_img),310,height-340,width=240,height=180,preserveAspectRatio=True,anchor='c')
    c.setFont("Helvetica-Bold",12); c.drawString(42,height-375,"Depth statistics (normalized scale)")
    c.setFont("Helvetica",10)
    y=height-395
    for key in ("min","max","mean","median","std"):
        c.drawString(52,y,f"{key.title()}: {summary[key]:.5f}"); y-=16
    if region:
        y-=10; c.setFont("Helvetica-Bold",12); c.drawString(42,y,"Selected region statistics"); y-=18; c.setFont("Helvetica",10)
        for key in ("min","max","mean","std","pixels"):
            c.drawString(52,y,f"{key.title()}: {region[key]}"); y-=16
    c.setFont("Helvetica-Oblique",8); c.drawString(42,28,"MiDaS relative depth is not absolute elevation. DEM calibration requires spatial co-registration and independent validation.")
    c.save(); return out.getvalue()
