from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import io
import cv2
import numpy as np
import base64
import supervision as sv
from ultralytics import YOLOv10
from PIL import Image

model = YOLOv10(f'./last.pt')
app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image from the uploaded file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Perform YOLO inference
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Annotate the image with bounding boxes and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Convert annotated image to bytes for streaming response
    _, img_encoded = cv2.imencode('.png', annotated_image)
    io_buf = io.BytesIO(img_encoded)

    return StreamingResponse(io_buf, media_type="image/png")


@app.post("/predictwithclasses/")
async def predict(file: UploadFile = File(...)):
    # Read image from the uploaded file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Perform YOLO inference
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    # print(results)

    # Prepare textual results (class IDs and confidence scores only)
    detection_data = []
    for detection in detections:
        detection_data.append({
            "class": detection[5],
            "confidence": str(detection[2])
        })

    # Annotate the image with bounding boxes and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Convert annotated image to PNG and then to base64
    _, img_encoded = cv2.imencode('.png', annotated_image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    print(detection_data)

    # Return JSON response with detection results (without bounding boxes) and the base64-encoded image
    return JSONResponse(content={
        "detections": detection_data,
        "annotated_image": img_base64
    })