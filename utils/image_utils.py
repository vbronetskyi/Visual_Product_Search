import io
from PIL import Image, UnidentifiedImageError, ImageOps

def preprocess_image(uploaded_file):
    try:
        img = Image.open(io.BytesIO(uploaded_file.read()))
        #EXIF орієнтація
        try:
            exif = img._getexif()
            orient = 274
            if exif and orient in exif:
                o = exif[orient]
                if o == 3: img = img.rotate(180, expand=True)
                elif o == 6: img = img.rotate(270, expand=True)
                elif o == 8: img = img.rotate(90, expand=True)
        
        except: pass
        if img.mode != "RGB": img = img.convert("RGB")
        target = (224,224)
        img.thumbnail(target, Image.ANTIALIAS)

        bg = Image.new("RGB", target, (255,255,255))
        off = ((target[0]-img.width)//2, (target[1]-img.height)//2)
        bg.paste(img, off)
        
        return ImageOps.autocontrast(bg) # aвтоконтраст img
    except UnidentifiedImageError:
        raise ValueError("Unidentified image")
    
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")
