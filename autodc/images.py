from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageFilter
import numpy as np
import json

VALID_EXT = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def process_image_zip(uploaded_zip, threshold=100.0):
    out_buffer = BytesIO()
    report = {"total_images":0, "accepted_images":0, "flagged_blurry":0, "items":[]}
    with ZipFile(uploaded_zip) as zin, ZipFile(out_buffer, "w", ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.is_dir():
                continue
            from pathlib import PurePosixPath
            path = PurePosixPath(info.filename)
            if path.suffix.lower() not in VALID_EXT:
                continue
            report["total_images"] += 1
            try:
                image = Image.open(zin.open(info)).convert("RGB")
                gray = image.convert("L")
                arr = np.asarray(gray, dtype=np.float32)
                lap = gray.filter(ImageFilter.FIND_EDGES)
                variance = float(np.asarray(lap, dtype=np.float32).var())
                blurry = variance < threshold
                item = {"file":info.filename, "width":image.width, "height":image.height,
                        "laplacian_variance_proxy":variance, "flagged_blurry":blurry}
                report["items"].append(item)
                if blurry:
                    report["flagged_blurry"] += 1
                    continue
                image = image.resize((224,224))
                buff = BytesIO()
                image.save(buff, format="JPEG", quality=92)
                name = str(path.with_suffix(".jpg"))
                zout.writestr(name, buff.getvalue())
                report["accepted_images"] += 1
            except Exception as e:
                report["items"].append({"file":info.filename, "error":str(e), "flagged_blurry":True})
                report["flagged_blurry"] += 1
    out_buffer.seek(0)
    return out_buffer.getvalue(), report
