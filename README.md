# Contactless Health Monitoring via Webcam

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project presents a real-time, non-contact system for monitoring physiological signals using a standard webcam. It uses remote photoplethysmography (rPPG) technology to estimate Heart Rate (HR), Heart Rate Variability (HRV), Breathing Rate (BR) and predict stress levels.

The system is built with Python and features a comprehensive processing pipeline including dynamic ROI selection, robust signal extraction, dynamic smoothing and an interactive GUI for live data visualization.

## Features

- **Real-Time Vital Signs:**
  - **Heart Rate (HR):** Beats per minute (BPM).
  - **Heart Rate Variability (HRV):** Time-domain metrics including SDNN and RMSSD.
  - **Breathing Rate (BR):** Breaths per minute from the rPPG signal.
- **Stress Level Inference:** A machine learning model predicts stress levels (Low, Medium, High) based on HRV features.
- **Dynamic ROI Selection:** Intelligently selects the best facial region (forehead, cheeks) for signal extraction based on head pose (yaw, pitch, roll) to maximize signal-to-noise ratio (SNR).
- **Cognitive Indicators:** Blink Detection, Blink Rate, Gaze (Z-Score), Attention Level using dynamic facial landmarks.
- **Robust Signal Processing:**
  - Implements multiple rPPG algorithms (**POS, CHROM, PBV**).
  - Automatically selects the best signal projection on a frame-by-frame basis based on SNR.
  - Utilizes Butterworth filtering and signal smoothing to handle noise and motion artifacts.
- **Interactive GUI:** A user-friendly interface built with PyQt6 and PyQtGraph displays the live webcam feed, vital signs, cognitive info, and signal plots.

## System Pipeline

The core logic of the application follows this pipeline:

1.  **Video Capture:** Reads video feed from a file or webcam.
2.  **Face & Landmark Detection:** Uses **MediaPipe Face Mesh** to detect the face and its 478 landmarks in real-time.
3.  **Dynamic ROI Extraction:**
    - Calculates head pose (yaw, pitch and roll).
    - Selects visible ROIs (forehead, left/right cheek).
    - Extracts pixel data from these regions using a convex hull mask.
4.  **rPPG Signal Extraction:**
    - Generates three different rPPG signals using **POS**, **CHROM**, and **PBV** methods.
    - Calculates the SNR for each signal.
    - Selects the signal with the highest quality for the current window.
5.  **Vital Sign Calculation:**
    - **HR/HRV:** The best signal is filtered using a Butterworth band-pass filter. Welch based FFT is detected to calculate HR and Peaks are detected to calculate SDNN, and RMSSD.
    - **BR:** The respiratory signal is derived from amplitude variations (AM) and inter-beat interval (IBI) changes extracted from the rPPG signal. These components are normalized, combined, and detrended to isolate a respiratory waveform. The breathing rate is then calculated by identifying the dominant frequency using Welch’s power spectral density (PSD) method.
6.  **Stress Prediction:** The calculated HR and HRV metrics are fed into a pre-trained Random Forest model to classify the user's stress level.
7.  **Visualization:** All metrics and plots are updated in real-time in the GUI.

## Installation

To set up the project, follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>/src
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To start the application, run the `main.py` script from the `src` directory:

```bash
python main.py
```

The application will launch, displaying the GUI. The system will use the `vid.avi` file as the default video source.

Modify `Capture_thread` parameter to your default camera id for realtime use.

### Command-Line Arguments

-   `--profile`: Run the application with cProfile to analyze performance.
    ```bash
    python main.py --profile
    ```

## Project Structure

```
.
├── UI/               # PyQt6 User Interface files
├── ROI/              # ROI extraction and stabilization logic
├── POS/              # rPPG signal processing, HR/HRV extraction
├── BR/               # Breathing rate extraction pipeline
├── STRESS/           # Stress detection model and assets
├── timeseries/       # Manages time-series data for signals
├── main.py           # Main application entry point
├── capture_thread.py # Handles video frame capture
├── config.py         # Configuration parameters
└── requirements.txt  # Project dependencies
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
