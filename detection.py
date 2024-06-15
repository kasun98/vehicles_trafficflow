import os
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics import YOLO, solutions


# Load a model
model = YOLO("vehicles/runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("vehicles/vids/vid1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# Video writer
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
out = cv2.VideoWriter("bar_chart.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (900, h))
out2 = cv2.VideoWriter("multiple_line_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

line_pts = [(0, 100), (720, 250)]
classes_to_count = [0, 1, 2, 3, 4, 6]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=model.names,
    view_img=True,
    line_thickness=1,
    region_thickness=2,
)

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=line_pts,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=1,
    track_thickness=1,
    region_thickness=2,
)

analytics = solutions.Analytics(
    type="bar",
    writer=out,
    title = "Classwise Total Vehicle Flow",
    im0_shape=(900, h),
    view_img=True,
)

analytics2 = solutions.Analytics(
    type="line",
    writer=out2,
    line_width=1,
    im0_shape=(w, h),
    view_img=True,
    max_points=200,
    
)

def transform_class_wise_count(counter):
    result = {}
    for vehicle, counts in counter.items():
        for direction, count in counts.items():
            result[f"{vehicle}_{direction}"] = count
    return result

def aggregate_counts(counter):
    result = {'IN': 0, 'OUT': 0}
    for vehicle, counts in counter.items():
        for direction, count in counts.items():
            result[direction] += count
    return result

frame_count = 0
data = {}
labels = ['IN','OUT']

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    frame_count += 1
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
    im0 = counter.start_counting(im0, tracks)
    im0 = speed_obj.estimate_speed(im0, tracks)
    if counter.class_wise_count is not None:
        
        analytics.update_bar(transform_class_wise_count(counter.class_wise_count))
        data = aggregate_counts(counter.class_wise_count)
        

    analytics2.update_multiple_lines(data, labels, frame_count)
    data = {}
    video_writer.write(im0)


cap.release()
video_writer.release()
cv2.destroyAllWindows()
