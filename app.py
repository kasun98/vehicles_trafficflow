import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics import YOLO, solutions
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import mpld3
import io

# Initialize the YOLO model
model = YOLO('vehicles/runs/detect/train/weights/best.pt')

cap = cv2.VideoCapture("vehicles/vids/vid1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter("bar_chart.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (900, h))
# out2 = cv2.VideoWriter("multiple_line_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Coorinates for lines
line_pts = [(0, 100), (720, 250)]
classes_to_count = [0, 1, 2, 3, 4, 6]
line_pts_spd_out = [(0, 100), (340, 170)]
line_pts_spd_in = [(340, 170), (720, 250)]

# Initialize speed-estimation objects
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts_spd_in,
    names=model.names,
    view_img=False,
    line_thickness=1,
    region_thickness=2,
)

speed_obj2 = solutions.SpeedEstimator(
    reg_pts=line_pts_spd_out,
    names=model.names,
    view_img=False,
    line_thickness=1,
    region_thickness=2,
)

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=line_pts,
    classes_names=model.names,
    view_in_counts = False,
    view_out_counts = False,
    draw_tracks=True,
    line_thickness=1,
    track_thickness=1,
    region_thickness=2,
)


# analytics = solutions.Analytics(
#     type="bar",
#     writer=out,
#     title = "Classwise Total Vehicle Flow",
#     im0_shape=(900, h),
#     view_img=True,)

# analytics2 = solutions.Analytics(
#     type="line",
#     writer=out2,
#     line_width=1,
#     im0_shape=(w, h),
#     view_img=True,
#     max_points=200,)


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

def mean_spead(obj):
    dist_data_values = list(obj.dist_data.values())
    if dist_data_values:
        mean_value = np.mean(dist_data_values)
    else:
        mean_value = 0
    return mean_value

frame_count = 0
data = {}
labels = ['IN','OUT']

# Tkinter setup
root = tk.Tk()
root.title("Traffic Flow Management")

# Set fixed size for the video frame
video_width = 720
video_height = 480

# Video frame
video_canvas = tk.Canvas(root, width=video_width, height=video_height)
video_canvas.grid(row=0, column=0, padx=10, pady=10)

# Bar chart frame
fig_bar, ax_bar = plt.subplots(figsize=(11.5, 4.7))
canvas_bar = FigureCanvasTkAgg(fig_bar, master=root)
canvas_bar.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

# Line chart frame with specific size
fig_line, ax_line = plt.subplots(figsize=(11.5, 4.7))
canvas_line = FigureCanvasTkAgg(fig_line, master=root)
canvas_line.get_tk_widget().grid(row=1, column=1, padx=10, pady=10)

# Average speeds of lanes
# Set fixed size for the gvideo frame
gvideo_width = 720
gvideo_height = 480

# gVideo frame using Canvas
gvideo_canvas = tk.Canvas(root, width=gvideo_width, height=gvideo_height)
gvideo_canvas.grid(row=1, column=0, padx=10, pady=10)

# Initialize Plotly figure for gauge charts
fig = go.Figure()

# Initialize gauge 1
gauge1 = fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=0,  # Initial value, will be updated dynamically
    domain={'x': [0, 0.43], 'y': [0, 1]},
    title={'text': "Average Lane Speed (OUT) Km/h"},
    gauge = {'axis': {'range': [None, 20]},
            'bar': {'color': "darkred"},
             'steps' : [
                 {'range': [15, 20], 'color': "lightgray"},]
            }
))

# Initialize gauge 2
gauge2 = fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=0,  # Initial value, will be updated dynamically
    domain={'x': [0.52, 0.95], 'y': [0, 1]},
    title={'text': "Average Lane Speed (IN) Km/h"},
    gauge = {'axis': {'range': [None, 20]},
            'bar': {'color': "darkblue"},
             'steps' : [
                 {'range': [15, 20], 'color': "lightgray"},]
            }
))

# Update layout properties
fig.update_layout(
    height=600,
    width=800,
    margin=dict(l=20, r=20, t=50, b=20),
    grid={'rows': 2, 'columns': 1, 'pattern': "independent"},
)

# Convert Plotly figure to image function
def plotly_fig_to_image(fig):
    img_bytes = pio.to_image(fig, format='png')
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((gvideo_width, gvideo_height), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk, img

# Display initial image
initial_imgtk, initial_img = plotly_fig_to_image(fig)
gvideo_canvas.create_image(0, 0, anchor=tk.NW, image=initial_imgtk)
gvideo_canvas.imgtk = initial_imgtk

# Line chart data initialization
time_series = []
in_counts = []
out_counts = []

def update_video():
    global frame_count, time_series, in_counts, out_counts
    
    cap = cv2.VideoCapture('vehicles/vids/vid1.mp4')

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        frame_count += 1
        tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
        im0 = counter.start_counting(im0, tracks)
        im0 = speed_obj.estimate_speed(im0, tracks)
        im0 = speed_obj2.estimate_speed(im0, tracks)

        if counter.class_wise_count is not None:
            #analytics.update_bar(transform_class_wise_count(counter.class_wise_count))
            data = aggregate_counts(counter.class_wise_count)

        #analytics2.update_multiple_lines(data, list(counter.class_wise_count.keys()), frame_count)
        #data = {}

        # Average speed of lanes
        speed_in = mean_spead(speed_obj2)
        speed_out = mean_spead(speed_obj)

        im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(im0_rgb)

        # Resize the image to fit the video frame
        img = img.resize((video_width, video_height), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        video_canvas.imgtk = imgtk
        
        # Update the bar chart
        ax_bar.clear()
        classes = list(transform_class_wise_count(counter.class_wise_count).keys())
        counts = list(transform_class_wise_count(counter.class_wise_count).values())
        bars = ax_bar.bar(classes, counts, color=plt.cm.Paired(np.arange(len(classes))))
        plt.rcParams['font.size'] = 8
        
        # Add labels on top of the bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2, height, str(count), ha='center', va='bottom')
       
        ax_bar.set_title('Class-wise Total Vehicle Flow', fontsize=12)
        ax_bar.set_xlabel('Vehicle Class and Direction', fontsize=9)
        ax_bar.set_ylabel('Count', fontsize=9)
        canvas_bar.draw()

        # Update the line chart with IN and OUT counts
        time_series.append(frame_count)
        in_counts.append(data['IN'])
        out_counts.append(data['OUT'])

        ax_line.clear()
        ax_line.plot(time_series, in_counts, label='IN', color='blue')
        ax_line.plot(time_series, out_counts, label='OUT', color='red')
        ax_line.legend(loc='upper left')
        ax_line.set_title('Lane-wise Total Vehicle Count', fontsize=12)
        ax_line.set_xlabel('Frame Count', fontsize=9)
        ax_line.set_ylabel('Count', fontsize=9)
        canvas_line.draw()

        # Gauge chart
        # Update the gauge chart
        value_gauge1 = speed_in
        value_gauge2 = speed_out

        # Update gauge values
        fig.data[0].value = value_gauge1
        fig.data[1].value = value_gauge2

        # Convert updated Plotly figure to image
        imgtk, img = plotly_fig_to_image(fig)

        # Update video canvas with new image
        gvideo_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        gvideo_canvas.imgtk = imgtk


        root.update_idletasks()
        root.update()

    cap.release()

# Run the video update in a separate thread
threading.Thread(target=update_video, daemon=True).start()

# Start the Tkinter event loop
root.mainloop()
