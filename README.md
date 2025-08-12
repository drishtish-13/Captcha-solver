# Audio Captcha Solver

Project Title

Audio Captcha Recognizer using Wav2Vec2 

Description

This project focuses on recognizing and decoding audio captchas using a deep learning approach. 



## Setup

1.  **Clone the repository (or extract the zip file):**
    ```bash
    git clone [https://github.com/yourusername/audio-captcha-solver.git](https://github.com/yourusername/audio-captcha-solver.git)  # Or unzip the archive
    cd audio-captcha-solver
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv  # Create a virtual environment
    source .venv/bin/activate  # Activate the environment (Linux/macOS)
    .venv\Scripts\activate  # Activate the environment (Windows)
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Preparation:**
    *   Place your audio captcha files (.wav) in the `data/audio` directory.

## Usage

### Training the model

```bash
python gen.py

python train.py

### for check the model prediction
python predict.py# Captcha-solver
