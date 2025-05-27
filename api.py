from fastapi import FastAPI, Form
from pydantic import BaseModel
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import requests
import time

app = FastAPI()

# Load models once
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

class DetectRequest(BaseModel):
    video_path: str
    result_api: str

def detect_license(video_path):
    vid = cv2.VideoCapture(video_path)
    results = set()
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        plates = yolo_LP_detect(frame, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        for plate in list_plates:
            x = int(plate[0])
            y = int(plate[1])
            w = int(plate[2] - plate[0])
            h = int(plate[3] - plate[1])
            crop_img = frame[y:y+h, x:x+w]
            lp = ""
            for cc in range(0,2):
                for ct in range(0,2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        results.add(lp)
                        break
                if lp != "unknown":
                    break
    vid.release()
    return list(results)

@app.post("/detect")
def detect_api(req: DetectRequest):
    plates = detect_license(req.video_path)
    # Send result to another API
    try:
        resp = requests.post(req.result_api, json={"plates": plates})
        return {"plates": plates, "sent_status": resp.status_code}
    except Exception as e:
        return {"plates": plates, "error": str(e)}

# To run: uvicorn api:app --reload
# Start server:
# uvicorn api:app --reload
# POST to /detect with JSON:

#**
# {
#   "video_path": "0310.mp4",
#   "result_api": "http://your-other-api/receive"
# }