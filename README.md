# GNSS-Denied Drone Localization Pipeline
This project implements a hierarchical visual localization pipeline to estimate a drone's trajectory in environments where GNSS data is unavailable or unreliable. It fuses visual odometry (via hloc and COLMAP) with raw onboard telemetry to generate a high-precision flight path.

## Prerequisites
Python 3.10+

Git

System Dependencies: Ensure you have ffmpeg and colmap installed on your system path.

## Setup Instructions
### 1. Initialize the Repository
Clone the repository and initialize the submodules required for the localization engine:

```bash
git clone <repository-url>
cd <project-folder>
git submodule update --init --recursive
```
### 2. Environment Configuration
Set up a virtual environment to isolate the project dependencies:

#### Create and activate virtual environment
```bash
python -m venv venv
```

Windows:
```powershell
venv\Scripts\activate
```
Linux/macOS:
```bash
source venv/bin/activate
```
### 3. Install Dependencies
Install all required Python libraries:

```bash
pip install -r requirements.txt
```
### 4. Data Preparation
Place your test flight files inside the data/testing/ directory:

test_flight.mp4: The raw drone video stream.

test_flight.srt: The corresponding DJI telemetry file.

Note: Ensure the filenames match the configuration paths defined in Maps.py.

Running the Pipeline
Build the Map (Optional):
If you are starting with a new environment, run the map-builder script first to generate the 3D reference database:

```bash
python build_map.py
```
Run Navigation:
Execute the localization pipeline to fuse visual data with telemetry:

```bash
python navigate.py
```
The script will output the processed trajectory as outputs/estimated_trajectory.srt, which can be visualized in any standard drone telemetry viewer.

```bash
python target_tracker.py
```
This finale script will use the former trajectory srt of the drone given by the hloc, and return the set of points we are looking at in the video.

### FINALE NOTE:
We only extracted 1 out of every 30 frames(1 second indexes since its 30fps) just for comnfort, we didnt want the hloc script to take a few hours each time and than the output to take too much space, in principal we get the results we intended.