# # -*- coding: utf-8 -*-
# # demo.py

# # Copyright (c) 2018, Eric Liang Yang @chiyangliang@gmail.com
# # Produced at the Robotics Laboratory of the City College of New York
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # * Redistributions of source code must retain the above copyright
# #   notice, this list of conditions and the following disclaimer.
# # * Redistributions in binary form must reproduce the above copyright
# #   notice, this list of conditions and the following disclaimer in the
# #   documentation and/or other materials provided with the distribution.
# # * Neither the name of the copyright holders nor the names of any
# #   contributors may be used to endorse or promote products derived
# #   from this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# # ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# # POSSIBILITY OF SUCH DAMAGE.

# import os
# import sys
# import json
# import numpy as np
# import torch
# from torch import nn
# from torch import optim
# from torch.optim import lr_scheduler
# from opts import parse_opts
# from model import generate_model
# from utils import Logger
# from train import train_epoch
# from validation import val_epoch
# from torch.autograd import Variable
# from models import resnet as resnet
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from PIL import Image
# import torch
# import cv2
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ModelLoad():

#     def __init__(self, opt):
#         self.model = resnet.resnet34(pretrained = True, num_classes=opt.n_classes)
#         self.model = self.model.to(device)
#         pretrain = torch.load(
#             "resnet34/save_10.pth", map_location=device, weights_only=True
#         )

#         print('--------------------        load the pretrained model        --------------------------------------')
#         saved_state_dict = pretrain['state_dict']
#         # saved_state_dict = pretrain.state_dict()
#         print('----------------------------------------------------------')
#         new_params = self.model.state_dict().copy()
#         for name, param in new_params.items():
#             if 'module.'+name in saved_state_dict and param.size() == saved_state_dict['module.'+name].size():
#                 new_params[name].copy_(saved_state_dict['module.'+name])

#         self.model.load_state_dict(new_params)
#         self.model.eval()

#     def model_pre(self, img):
#         predict = self.model(img)
#         return predict

# if __name__ == '__main__':

#     opt = parse_opts()

#     modelM = ModelLoad(opt)
#     regionSize = 40
#     color_m = [(0,255,0), (255,255,0)]

#     # load the prepared data
#     with torch.no_grad():
#         all_path = 'image_and_results/image/'
#         output   = 'image_and_results/output/'
#         for file in os.listdir(all_path):
#            if file.endswith('.png'):
#                img_path = all_path + file
#                img = cv2.imread(img_path)
#                img = np.asarray(img)
#                ori = img.copy()
#                print('img', file, img.shape)

#                height, width, channels = img.shape
#                widthLevel = int(width/regionSize);
#                heightLevel = int(height/regionSize);

#                for ii in range(1, widthLevel-1):
#                    for jj in range(1, heightLevel-1):
#                        rangeXForCut = [(ii-1)*regionSize +1, (ii-1)*regionSize + regionSize*2]
#                        rangeYForCut = [(jj-1)*regionSize +1, (jj-1)*regionSize + regionSize*2]
#                        if rangeXForCut[1] > width or rangeYForCut[1] > height:
#                            continue
#                        tempImg = img[int(rangeYForCut[0]):int(rangeYForCut[1]), int(rangeXForCut[0]):int(rangeXForCut[1]), :]

#                        #cv2.imwrite('ou.jpg',tempImg)
#                        height1, width1, channels1 = tempImg.shape
#                        if height1 != 0 and width1 != 0:
#                            tempImg = cv2.resize(tempImg,(224, 224), interpolation = cv2.INTER_CUBIC)

#                            tempImg = np.asarray(tempImg)
#                            sour_img = torch.Tensor(torch.from_numpy(tempImg).float().div(255))
#                            sour_img = sour_img.permute(2, 0, 1)
#                            sour_img     = sour_img.unsqueeze(0)
#                            sour_img = Variable(sour_img).to(device)
#                            predict =  modelM.model_pre(sour_img)
#                            #result = predict.data.cpu().numpy()
#                            _, predicted = torch.max(predict, 1)
#                            #print(predicted.data.cpu().numpy()[0])
#                            if predicted.data.cpu().numpy()[0] == 1 or predicted.data.cpu().numpy()[0] == 2:
#                                try:
#                                   cv2.rectangle(img,(int(rangeXForCut[1]), int(rangeYForCut[1])), (int(rangeXForCut[0]), int(rangeYForCut[0])), color_m[predicted.data.cpu().numpy()[0] - 1], -1)
#                                except:
#                                   print(rangeXForCut, rangeYForCut, predicted.data.cpu().numpy()[0])
#                                   print(img.shape)

#                cv2.addWeighted(ori, 0.6, img, 0.4,0, img)
#                cv2.imwrite(output + file, img)

#         '''
#         pre_first = modelM.model_first(sour_img)

#         featuremap = pre_first.data.cpu().numpy()
#         print(featuremap.shape)

#         cmap = plt.get_cmap('jet')
#         all_image = np.empty([56,56])
#         all_image.fill(0)
#         for i in range(featuremap.shape[1]):
#             cur_img = featuremap[0, i, :, :]
#             all_image = all_image + featuremap[0, i, :, :]
#             cur_img = (cur_img - np.amin(cur_img))/(np.amax(cur_img) - np.amin(cur_img))
#             #rgba_img = cmap(cur_img)
#             cur_img = cv2.resize(cur_img,(200,200))
#             cur_img =  np.uint8(cm.jet(1 - cur_img)*255)
#             cv2.imwrite('feature_res/img1/features/%06d.jpg'%i, cur_img)

#         all_image = (all_image - np.amin(all_image))/(np.amax(all_image) - np.amin(all_image))
#         #rgba_img = cmap(cur_img)
#         all_image = cv2.resize(all_image,(200,200))
#         all_image =  np.uint8(cm.jet(1 - all_image)*255)
#         cv2.imwrite('feature_res/img1/features/%06d.jpg'%featuremap.shape[1], all_image)

#         ###########################second Group################################

#         pre_first = modelM.model_sec(sour_img)

#         featuremap = pre_first.data.cpu().numpy()
#         print(featuremap.shape)

#         cmap = plt.get_cmap('jet')
#         all_image = np.empty([28,28])
#         all_image.fill(0)
#         for i in range(featuremap.shape[1]):
#             cur_img = featuremap[0, i, :, :]
#             all_image = all_image + featuremap[0, i, :, :]
#             cur_img = (cur_img - np.amin(cur_img))/(np.amax(cur_img) - np.amin(cur_img))
#             #rgba_img = cmap(cur_img)
#             cur_img = cv2.resize(cur_img,(200,200))
#             cur_img =  np.uint8(cm.jet(1 - cur_img)*255)
#             cv2.imwrite('feature_res/img1/feature2/%06d.jpg'%i, cur_img)

#         all_image = (all_image - np.amin(all_image))/(np.amax(all_image) - np.amin(all_image))
#         #rgba_img = cmap(cur_img)
#         all_image = cv2.resize(all_image,(200,200))
#         all_image =  np.uint8(cm.jet(1 - all_image)*255)
#         cv2.imwrite('feature_res/img1/feature2/%06d.jpg'%featuremap.shape[1], all_image)
#         '''
import os
import torch
from torch.autograd import Variable
import cv2
import numpy as np
from opts import parse_opts
from models import resnet as resnet
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the class to load and initialize the model
class ModelLoad:
    def __init__(self, opt):
        self.model = resnet.resnet34(pretrained=True, num_classes=opt.n_classes)
        self.model = self.model.to(device)
        pretrain = torch.load(
            "resnet34/save_9.pth", map_location=device, weights_only=True
        )

        print("Loading pretrained model...")
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


if __name__ == "__main__":
    opt = parse_opts()

    modelM = ModelLoad(opt)
    base_region_size = 40  # Define the base region size
    color_m = [
        (0, 255, 0),
        (255, 255, 0),
    ]  # Define colors for classification (green for class 1, yellow for class 2)

    # Load the prepared data
    with torch.no_grad():
        all_path = "image_and_results/image/"  # Input directory
        output = "image_and_results/output/"  # Output directory

        for file in os.listdir(all_path):
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):  # Accept multiple image formats
                img_path = os.path.join(all_path, file)
                img = cv2.imread(img_path)
                ori = img.copy()

                height, width, channels = img.shape

                # Step 1: Print the image size and classify the entire image first
                print(f"Processing image '{file}' with size {width}x{height}...")

                # Resize the image to 224x224 (ResNet input size)
                img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                sour_img = torch.Tensor(torch.from_numpy(img_resized).float().div(255))
                sour_img = sour_img.permute(2, 0, 1).unsqueeze(0).to(device)

                # Run the model prediction on the entire image
                predict = modelM.model_pre(sour_img)
                _, predicted = torch.max(predict, 1)

                # Print the initial classification result
                if predicted.data.cpu().numpy()[0] == 0:
                    print(f"Initial classification: The image '{file}' is NOT CRACKED.")
                else:
                    print(
                        f"Initial classification: The image '{file}' is CRACKED (Class {predicted.data.cpu().numpy()[0]})."
                    )

                # Step 2: For small images, classify and potentially mask the entire image
                if height < 224 or width < 224:
                    if predicted.data.cpu().numpy()[0] == 0:
                        # The entire image is classified as NOT CRACKED (class 0)
                        print(f"The image '{file}' does not contain any cracks.")
                        # Save the original image (no need for masking)
                        cv2.imwrite(os.path.join(output, file), img)
                    else:
                        # The image contains cracks (class 1 or 2)
                        print(
                            f"The image '{file}' contains cracks (class {predicted.data.cpu().numpy()[0]})."
                        )
                        # Mask the entire image
                        color = color_m[predicted.data.cpu().numpy()[0] - 1]
                        cv2.rectangle(img, (0, 0), (width, height), color, -1)
                        cv2.addWeighted(ori, 0.6, img, 0.4, 0, img)
                        output_file = os.path.join(output, file)
                        cv2.imwrite(output_file, img)
                        print(f"Processed and saved: {output_file}")

                # Step 3: For larger images, ALWAYS apply sliding window approach, regardless of classification
                if height >= 224 or width >= 224:
                    print(
                        f"The image '{file}' is large. Applying sliding window approach."
                    )

                    regionSize = base_region_size  # Use the base window size of 40x40

                    widthLevel = max(1, int(width / regionSize))
                    heightLevel = max(1, int(height / regionSize))

                    cracked_regions_found = False

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

                            # Ensure the window doesn't exceed the image boundaries
                            if rangeXForCut[1] > width or rangeYForCut[1] > height:
                                continue

                            tempImg = img[
                                rangeYForCut[0] : rangeYForCut[1],
                                rangeXForCut[0] : rangeXForCut[1],
                                :,
                            ]

                            height1, width1, channels1 = tempImg.shape
                            if height1 != 0 and width1 != 0:
                                tempImg = cv2.resize(
                                    tempImg, (224, 224), interpolation=cv2.INTER_CUBIC
                                )
                                tempImg = np.asarray(tempImg)

                                # Preprocess the image
                                sour_img = torch.Tensor(
                                    torch.from_numpy(tempImg).float().div(255)
                                )
                                sour_img = (
                                    sour_img.permute(2, 0, 1).unsqueeze(0).to(device)
                                )

                                # Run the model prediction on the windowed region
                                predict = modelM.model_pre(sour_img)
                                _, predicted = torch.max(predict, 1)

                                # If the predicted class is 1 or 2, highlight the region
                                if (
                                    predicted.data.cpu().numpy()[0] == 1
                                    or predicted.data.cpu().numpy()[0] == 2
                                ):
                                    color = color_m[predicted.data.cpu().numpy()[0] - 1]
                                    cv2.rectangle(
                                        img,
                                        (int(rangeXForCut[1]), int(rangeYForCut[1])),
                                        (int(rangeXForCut[0]), int(rangeYForCut[0])),
                                        color,
                                        -1,
                                    )
                                    cracked_regions_found = True

                    # If any cracks were found, save the masked image
                    if cracked_regions_found:
                        cv2.addWeighted(ori, 0.6, img, 0.4, 0, img)
                        output_file = os.path.join(output, file)
                        cv2.imwrite(output_file, img)
                        print(f"Processed and saved: {output_file}")
                    else:
                        print(f"No cracked regions found for large image '{file}'.")
