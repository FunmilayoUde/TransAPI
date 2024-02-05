from fastapi import FastAPI,UploadFile, File, Response, Body
from bisenetv2 import BiSeNetV2
from skinsegmentation import load_skin_segmentation_model,create_skin_mask, refine_mask
from colordetection import extractSkin, extractDominantColor, adjust_hsv_dominance
from transfer import applyAdjustedColorToSkinRegion,blendSkinWithTexture
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
from PIL import Image
import cv2


app = FastAPI(debug= True)
modelweight = 'model_segmentation_realtime_skin_30.pth'
model = load_skin_segmentation_model(modelweight)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Adjust as needed
    allow_headers=["*"],  # Adjust as needed
)
# Model loading and function definitions will go here
@app.get('/')
def home():

    return{'text':'Ahoy! API coming through'}


@app.post("/createmask")
async def create_mask(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))  # Load image using PIL
    image_array = np.array(image)  # Convert to NumPy array

    mask, resized_image = create_skin_mask(model, image_array)

    # Return mask as bytes (adjust format if needed)
    mask_bytes = mask.tobytes()
    resized_image_bytes = cv2.imencode(".jpg", resized_image)[1].tobytes() 
     # Create a response with both image bytes
    response = Response(
        content=b"".join([mask_bytes, resized_image_bytes]),  # Combine bytes
        media_type="multipart/form-data"  # Indicate multiple parts
    )

    # Set appropriate content-disposition headers for each image
    response.headers["Content-Disposition"] = f"attachment; filename=mask.png"
    response.headers.append("Content-Disposition", f"attachment; filename=resized.jpg")

    return response

@app.post("/refinemask")
async def refine_mask(image: UploadFile = File(...), mask: UploadFile = File(...)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))  # Load image using PIL
    image_array = np.array(image)  # Convert to NumPy array

    mask_bytes = await mask.read()
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)  # Load mask as NumPy array
    mask_array = mask_array.reshape(image.shape[:2])  # Reshape to match image dimensions

    refined_mask = refine_mask(image, model, mask_array)

    # Return refined mask as bytes (adjust format if needed)
    refined_mask_bytes = refined_mask.tobytes()
    return Response(content=refined_mask_bytes, media_type="image/png")

@app.post("/extractskin")
async def extract_skin(image: UploadFile = File(...), segmentation_mask: UploadFile = File(None)):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))  # Load image using PIL
    image_array = np.array(image)  # Convert to NumPy array

    if segmentation_mask is not None:
        # Load segmentation mask if provided
        mask_bytes = await segmentation_mask.read()
        mask = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = mask.reshape(image_array.shape[:2])  # Reshape to match image dimensions
    else:
        mask = None

    extracted_skin = extractSkin(image_array, segmentation_mask=mask)  # Call your function with appropriate mask
    # Return extracted skin as bytes (adjust format if needed)
    skin_bytes = cv2.imencode(".jpg", extracted_skin)[1].tobytes()  # Example using JPEG format
    return Response(content=skin_bytes, media_type="image/jpeg")


@app.post("/extractdominantcolor")
async def extract_dominant_color(image: UploadFile = File(...), number_of_colors: int = 5, has_thresholding: bool = False):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))  # Load image using PIL
    image_array = np.array(image)  # Convert to NumPy array

    dominant_color = extractDominantColor(image_array, number_of_colors=number_of_colors, hasThresholding=has_thresholding)  # Call your function with thresholding

    # Return dominant color as a NumPy array (no need for .tolist())
    return {"dominant_color": dominant_color.tolist()}

@app.post("/adjustdominantcolor")
async def adjust_dominant_color(dominant_color: str = Body(...), hsv_adjust: float = 0.2):
    # Parse dominant color from string input
    dominant_color_array = np.array(list(map(int, dominant_color.split(",")))).astype('uint8')

    adjusted_color = adjust_hsv_dominance(dominant_color_array, hsv_adjust)  # Call your function

    # Return adjusted color as RGB values in a JSON-compatible format
    return {"adjusted_color": adjusted_color.tolist()}

@app.post("/applyadjustedcolor")
async def apply_adjusted_color(target_skin: UploadFile = File(...),
                               target_skin_mask: UploadFile = File(...),
                               adjusted_color: str = Body(...)):
    target_skin_bytes = await target_skin.read()
    target_skin_array = np.array(Image.open(io.BytesIO(target_skin_bytes)))

    mask_bytes = await target_skin_mask.read()
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_array = mask_array.reshape(target_skin_array.shape[:2])  # Reshape mask

    adjusted_color_array = np.array(list(map(int, adjusted_color.split(",")))).astype('uint8')

    applied_skin = applyAdjustedColorToSkinRegion(target_skin_array, mask_array, adjusted_color_array)

    # Return adjusted image as bytes (adjust format if needed)
    applied_skin_bytes = cv2.imencode(".jpg", applied_skin)[1].tobytes()  # Example using JPEG format
    return Response(content=applied_skin_bytes, media_type="image/jpeg")

@app.post("/blendwithtexture")
async def blend_with_texture(target_skin: UploadFile = File(...),
                             target_skin_mask: UploadFile = File(...),
                             target_skin_result: UploadFile = File(...),
                             resized_target_image: UploadFile = File(...),
                             alpha: float = 0.5):
    # Load all images and convert to NumPy arrays
    target_skin_array = np.array(Image.open(io.BytesIO(await target_skin.read())))
    mask_array = np.frombuffer(await target_skin_mask.read(), dtype=np.uint8).reshape(target_skin_array.shape[:2])
    target_skin_result_array = np.array(Image.open(io.BytesIO(await target_skin_result.read())))
    resized_target_image_array = np.array(Image.open(io.BytesIO(await resized_target_image.read())))

    blended_image = blendSkinWithTexture(target_skin_array, mask_array, target_skin_result_array, resized_target_image_array, alpha)

    # Return blended image as bytes (adjust format if needed)
    blended_image_bytes = cv2.imencode(".jpg", blended_image)[1].tobytes()  # Example using JPEG format
    return Response(content=blended_image_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app)

