# Recognize Vehicles and Human Face by YOLOv8
## Introduction
This project is built to recognize vehicles (buses, cars, trucks, motorcycles) and human faces in real-world environments using [YOLOv8](https://docs.ultralytics.com/models/yolov8/) with the details is YOLOv8m and YOLOv8n-face models. By identifying and numbering the IDs of the appearing objects, the model will track the disappearance time of the object and save it to a json file.

<p align="center">
  <img src="https://github.com/SaikySu/Yolov8-Vehicles-and-Human-face-Detection/blob/main/Example/Face_before.gif" width="400">
  <img src="https://github.com/SaikySu/Yolov8-Vehicles-and-Human-face-Detection/blob/main/Example/Face_after.gif" width="400"><br/>
  <img src="https://github.com/SaikySu/Yolov8-Vehicles-and-Human-face-Detection/blob/main/Example/Vehicles_before.gif" width="400">
  <img src="https://github.com/SaikySu/Yolov8-Vehicles-and-Human-face-Detection/blob/main/Example/Vehicles_after.gif" width="400"><br/>
  <i>Sample results</i>
</p>

## How to use
1. Download the source code in the repository.
2. Install all requirements libraries
3. Run this code in: 

```python
    python Object_Detection.py --input Your_video/mp4
```

## Features
**Vehicales Detection**: This model focuses on detecting vehicles such as motorbikes, cars, buses, trucks and numbering the IDs of the appearing objects.

**Human Face Detection**: Detect human face and assign IDs separate from vehicle IDs, and save data to json file.

**Bounding Boxes**: When detecting a vehicle or human face, the model will Bounding Boxes and assign IDs. Parameters displayed on the output include: ID of the vehicle/face in the video and the accompanying accuracy.

**Time appeared and disappeared**: The time of appearance of the vehicle/face is recorded in the json file exported with the output video.

**Output**: The output video has a quality corresponding to the quality of the input video and the json file has a name corresponding to the name of the input video.

## Requirements
* CV2
* Ultralytics.YOLO
* Json
* Collections
* Os

## Some notes:
* This model recognizes the car using the YOLOv8m model to detection vehicale, if you want to change the model, you can change at:

```python
    parser.add_argument('--vehicle_model', type=str, default='models/yolov8m.pt', help="Path to vehicle detection model")
```

* Similar, model recognizes the car using the YOLOv8n-face model to detection Human face, if you want to change the model, you can change at:

```python
    parser.add_argument('--face_model', type=str, default='models/yolov8n-face.pt', help="Path to face detection model")
```
