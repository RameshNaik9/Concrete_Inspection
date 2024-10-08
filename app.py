import os
import cv2
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
from io import BytesIO
from PIL import Image
from torch.autograd import Variable
from models import resnet as resnet
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from opts import parse_opts
from model import generate_model
from utils import Logger
from train import train_epoch
from validation import val_epoch
from torch.autograd import Variable
from models import resnet as resnet
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
import ssl
from fastapi.responses import StreamingResponse
from io import BytesIO
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import UnidentifiedImageError
import logging
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssl._create_default_https_context = ssl._create_unverified_context


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

    regionSize = 40
    color_m = [(0, 255, 0), (255, 255, 0)]
    height, width, _ = img.shape
    widthLevel = int(width / regionSize)
    heightLevel = int(height / regionSize)

    for ii in range(1, widthLevel - 1):
        for jj in range(1, heightLevel - 1):
            rangeXForCut = [
                (ii - 1) * regionSize + 1,
                (ii - 1) * regionSize + regionSize * 2,
            ]
            rangeYForCut = [
                (jj - 1) * regionSize + 1,
                (jj - 1) * regionSize + regionSize * 2,
            ]
            if rangeXForCut[1] > width or rangeYForCut[1] > height:
                continue
            tempImg = img[
                int(rangeYForCut[0]) : int(rangeYForCut[1]),
                int(rangeXForCut[0]) : int(rangeXForCut[1]),
                :,
            ]
            if tempImg.shape[0] != 0 and tempImg.shape[1] != 0:
                tempImg = cv2.resize(tempImg, (224, 224), interpolation=cv2.INTER_CUBIC)
                sour_img = torch.Tensor(torch.from_numpy(tempImg).float().div(255))
                sour_img = sour_img.permute(2, 0, 1)
                sour_img = sour_img.unsqueeze(0)
                sour_img = Variable(sour_img).to(device)
                predict = modelM.model_pre(sour_img)
                _, predicted = torch.max(predict, 1)
                if (
                    predicted.data.cpu().numpy()[0] == 1
                    or predicted.data.cpu().numpy()[0] == 2
                ):
                    cv2.rectangle(
                        img,
                        (int(rangeXForCut[1]), int(rangeYForCut[1])),
                        (int(rangeXForCut[0]), int(rangeYForCut[0])),
                        color_m[predicted.data.cpu().numpy()[0] - 1],
                        -1,
                    )

    cv2.addWeighted(ori, 0.6, img, 0.4, 0, img)
    return Image.fromarray(img)


@app.post("/process_image/")
async def process_upload_image(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")

        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")

        processed_image = process_image(contents)

        img_byte_arr = BytesIO()
        processed_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        logger.info("Image processed successfully")

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except UnidentifiedImageError:
        logger.error("Unsupported file type or corrupt image")
        return {"error": "Unsupported file type or corrupt image"}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}
