from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("/concrete-defect-segmentation/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="/image_and_results/image/175.png", show=True,save=True)