import os
import torch
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from torch.autograd import Variable
from models import resnet as resnet
from opts import parse_opts
from model import generate_model
import ssl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()


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
