import os
import torch
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from torch.autograd import Variable
from models import resnet as resnet
from opts import parse_opts
from model import generate_model
import ssl
import logging
from fastapi import Request, HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model
modelYolo = YOLO("best.pt")  # Update with your trained model's path


# Model loading class for ResNet
class ModelLoad:
    def __init__(self, opt):
        self.model = resnet.resnet34(pretrained=True, num_classes=opt.n_classes)
        self.model = self.model.to(device)
        pretrain = torch.load(
            "resnet34/save_10.pth", map_location=device, weights_only=True
        )

        saved_state_dict = pretrain["state_dict"]
        new_params = self.model.state_dict().copy()

        for name, param in new_params.items():
            if (
                "module." + name in saved_state_dict
                and param.size() == saved_state_dict["module." + name].size()
            ):
                new_params[name].copy_(saved_state_dict["module." + name])

        self.model.load_state_dict(new_params)
        self.model.eval()

    def model_pre(self, img):
        predict = self.model(img)
        return predict


@app.on_event("startup")
async def load_model():
    global modelM
    opt = parse_opts()
    modelM = ModelLoad(opt)


def process_image(file):
    image = Image.open(BytesIO(file)).convert("RGB")
    img = np.array(image)
    ori = img.copy()

    height, width, channels = img.shape
    base_region_size = 40
    color_m = [(0, 255, 0), (255, 255, 0)]

    # Resize the image to 224x224 for initial classification
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    sour_img = torch.Tensor(torch.from_numpy(img_resized).float().div(255))
    sour_img = sour_img.permute(2, 0, 1).unsqueeze(0).to(device)

    # Run initial classification on the entire image
    predict = modelM.model_pre(sour_img)
    _, predicted = torch.max(predict, 1)
    initial_class = predicted.data.cpu().numpy()[0]

    # Step 2: For small images, mask the entire image based on classification
    if height < 224 or width < 224:
        if initial_class != 0:
            # If cracked, apply a mask to the entire image
            color = color_m[initial_class - 1]
            cv2.rectangle(img, (0, 0), (width, height), color, -1)
        cv2.addWeighted(ori, 0.6, img, 0.4, 0, img)
        return Image.fromarray(img), initial_class, (width, height)

    # Step 3: For larger images, apply sliding window approach
    cracked_regions_found = False
    widthLevel = max(1, int(width / base_region_size))
    heightLevel = max(1, int(height / base_region_size))

    for ii in range(1, widthLevel - 1):
        for jj in range(1, heightLevel - 1):
            rangeXForCut = [
                (ii - 1) * base_region_size + 1,
                (ii - 1) * base_region_size + base_region_size * 2,
            ]
            rangeYForCut = [
                (jj - 1) * base_region_size + 1,
                (jj - 1) * base_region_size + base_region_size * 2,
            ]

            # Ensure the window doesn't exceed image boundaries
            if rangeXForCut[1] > width or rangeYForCut[1] > height:
                continue

            tempImg = img[
                rangeYForCut[0] : rangeYForCut[1], rangeXForCut[0] : rangeXForCut[1], :
            ]
            height1, width1, channels1 = tempImg.shape
            if height1 != 0 and width1 != 0:
                tempImg = cv2.resize(tempImg, (224, 224), interpolation=cv2.INTER_CUBIC)
                sour_img = torch.Tensor(torch.from_numpy(tempImg).float().div(255))
                sour_img = sour_img.permute(2, 0, 1).unsqueeze(0).to(device)

                predict = modelM.model_pre(sour_img)
                _, predicted = torch.max(predict, 1)
                window_class = predicted.data.cpu().numpy()[0]

                # If the window is classified as cracked, mask the region
                if window_class == 1 or window_class == 2:
                    color = color_m[window_class - 1]
                    cv2.rectangle(
                        img,
                        (int(rangeXForCut[1]), int(rangeYForCut[1])),
                        (int(rangeXForCut[0]), int(rangeYForCut[0])),
                        color,
                        -1,
                    )
                    cracked_regions_found = True

    # If any cracked regions were found, save the masked image
    if cracked_regions_found or initial_class != 0:
        cv2.addWeighted(ori, 0.6, img, 0.4, 0, img)
        return Image.fromarray(img), initial_class, (width, height)
    else:
        return (
            Image.fromarray(img),
            0,
            (width, height),
        )  # Return not cracked if no cracks were found


@app.post("/process_image/")
async def process_upload_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()

        processed_image, initial_class, image_size = process_image(contents)

        img_byte_arr = BytesIO()
        processed_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Prepare headers
        headers = {
            "Class": f"Class {initial_class}",
            "Image-Size": f"{image_size[0]}x{image_size[1]}",
        }

        logger.info("Image processed successfully")

        return StreamingResponse(img_byte_arr, headers=headers, media_type="image/png")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}


@app.post("/segment/image")
async def segment_image(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Run YOLO segmentation on the image
    results = modelYolo.predict(source=img, save=False, show=False)

    # Extract the result from the prediction (e.g., the first result if multiple)
    segmented_img = results[0].plot()  # Get image with overlays

    # Encode image to JPEG format to send back as response
    _, im_encoded = cv2.imencode(".jpg", segmented_img)
    image_io = BytesIO(im_encoded.tobytes())
    image_io.seek(0)  # Reset stream position

    return StreamingResponse(image_io, media_type="image/jpeg")


@app.post("/segment/video")
async def segment_video(file: UploadFile = File(...)):
    # Read uploaded video
    video_bytes = await file.read()
    np_video = np.frombuffer(video_bytes, np.uint8)

    # Decode video from buffer
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(np_video)

    # Run YOLO segmentation on the video
    results = modelYolo.predict(source=video_path, save=False)

    # Create a video stream in memory
    output_video = BytesIO()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
    height, width = results[0].orig_shape[:2]  # Get original shape of the first frame
    video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    for result in results:
        frame = result.plot()  # Get the segmented frame
        video_writer.write(frame)  # Write each segmented frame

    video_writer.release()
    output_video.seek(0)  # Reset stream position

    return StreamingResponse(output_video, media_type="video/mp4")


@app.post("/labelstudio/submit/")
async def labelstudio_submit(request: Request):
    try:
        # Get the payload from Label Studio
        payload = await request.json()
        print("Received labeled data:", json.dumps(payload, indent=4))

        # Process the labeled data here (e.g., save to database, retrain model, etc.)

        return {"status": "success", "message": "Data received"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
import json

async def labelstudio_submit(request: Request):
    try:
        # Get the payload from Label Studio
        payload = await request.json()
        print("Received labeled data:", json.dumps(payload, indent=4))

        # Process the labeled data here (e.g., save to database, retrain model, etc.)

        return {"status": "success", "message": "Data received"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Start FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
