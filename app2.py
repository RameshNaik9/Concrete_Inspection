
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from io import BytesIO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")  # Update with your trained model's path


@app.post("/segment/image")
async def segment_image(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Run YOLO segmentation on the image
    results = model.predict(source=img, save=False, show=False)

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
    results = model.predict(source=video_path, save=False)

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


# Start FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
