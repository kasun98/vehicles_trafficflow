# Vehicle Traffic Flow Management

This project is a vehicle traffic flow management system using YOLOv8 for object detection and tracking. The system processes a video to detect vehicles, estimate their speeds using ultralytics speed estimation, and count them as they pass through predefined lines. It also provides a graphical interface to display the video feed, bar charts of vehicle counts, line charts of vehicle flow, and gauge charts of average speeds.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)


## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/kasun98/vehicles_trafficflow.git
    cd vehicles_trafficflow
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download YOLOv8 weights or train on custom data**: Ensure you have the YOLOv8 weights (`best.pt`) saved in `vehicles/runs/detect/train/weights/`.

5. **Ensure you have the video file**: Place your video file in the `vehicles/vids/` directory.

## Usage

Run the application:
```sh
python app.py
```

This will open a graphical interface where the video feed, vehicle counts, flow charts, and speed gauges will be displayed.

## Project Structure

- **app.py**: Main application script.
- **vehicles/**: Directory containing video files and YOLOv8 weights.
  - **vids/**: Directory for video files.
  - **runs/**: Directory for YOLOv8 runs.
    - **detect/**: Directory for detection runs.
      - **train/**: Directory for training runs.
        - **weights/**: Directory for YOLOv8 weights.
- **requirements.txt**: List of dependencies.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

