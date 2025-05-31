import threading
import json
import time
from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import requests

app = FastAPI()

# Load models once
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

destinationResult = "http://your-other-api/receive"
CAMERA_CONFIG_FILE = "cameras.json"
camera_list_lock = threading.Lock()
camera_list = []

def load_camera_list():
    global camera_list
    with camera_list_lock:
        try:
            with open(CAMERA_CONFIG_FILE, "r") as f:
                camera_list = json.load(f)
        except Exception:
            camera_list = []

def save_camera_list():
    with camera_list_lock:
        with open(CAMERA_CONFIG_FILE, "w") as f:
            json.dump(camera_list, f, indent=2)

# Load camera list at startup
load_camera_list()

class CameraConfig(BaseModel):
    video_path: str
    laneId: int

def send_plate_to_api(laneId, plate):
    try:
        resp = requests.post(destinationResult, json={"plate": plate, "timestamp": time.time(), "laneId": laneId})
        return {"plate": plate, "sent_status": resp.status_code}
    except Exception as e:
        return {"plate": plate, "error": str(e)}

def detect_license(video_path, laneId):
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
                        if lp not in results:
                            results.add(lp)
                            send_plate_to_api(laneId, lp)
                        break
                if lp != "unknown":
                    break
    vid.release()
    return list(results)

def camera_worker(camera):
    while True:
        detect_license(camera["video_path"], camera["laneId"])
        time.sleep(1)  # Adjust as needed

def start_all_cameras():
    load_camera_list()
    for camera in camera_list:
        t = threading.Thread(target=camera_worker, args=(camera,), daemon=True) 
        t.start()

@app.post("/add_camera")
def add_camera(camera: CameraConfig):
    load_camera_list()
    with camera_list_lock:
        camera_list.append({"video_path": camera.video_path, "laneId": camera.laneId})
        save_camera_list()
    # Start a new thread for this camera
    t = threading.Thread(target=camera_worker, args=({"video_path": camera.video_path, "laneId": camera.laneId},), daemon=True)
    t.start()
    return {"status": "added", "camera": camera}

@app.get("/cameras")
def get_cameras():
    load_camera_list()
    return {"cameras": camera_list}

# Start all camera threads when app starts
@app.on_event("startup")
def on_startup():
    start_all_cameras()
