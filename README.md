# Multi-Numbers Detection

Multi-Numbers Detection is a computer vision project that detects and recognizes handwritten or displayed digits in real time using a webcam.

The project uses two separate deep learning models:

1. **Digit Detection Model**: detects where each digit is located in the camera frame.
2. **Digit Recognition Model**: recognizes the detected digit as a class from `0` to `9`.

The detection model was trained on a custom dataset created specifically for this project. Since the dataset currently contains a limited number of images, the detection model can still be improved by collecting and annotating more data.

---

## Features

- Real-time digit detection using a webcam.
- Detection of multiple digits in the same frame.
- Custom dataset support for digit detection.
- Bounding box prediction using a TinySSD model.
- Digit classification using a CNN model.
- Image preprocessing before recognition.
- Separate models for detection and recognition.
- Local training support for the model files ignored by Git.

---

## How the Project Works

The project follows this pipeline:

1. The webcam captures a frame.
2. The TinySSD detection model predicts bounding boxes around digits.
3. Each detected digit is cropped from the original frame.
4. The crop is converted to grayscale and binarized.
5. The cropped digit is resized to `28x28` and normalized.
6. The CNN recognition model predicts the digit class.
7. The result is displayed on the video feed with a bounding box and the predicted digit.

---

## Models Used

### 1. Digit Detection Model

The detection model is based on a TinySSD architecture. It is trained to detect one class: `digit`.

The model outputs bounding boxes around digits. These boxes are then used to crop each digit before sending it to the recognition model.

Expected local weight file:

```bash
ssd_digits.pth
```

### 2. Digit Recognition Model

The recognition model is a CNN used to classify cropped digit images into one of ten classes:

```bash
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Expected local weight file:

```bash
cnn_model.pth
```

---

## Important Note About Ignored Files

Some files and folders are intentionally not pushed to GitHub because they are generated locally, too large, or specific to the developer environment.

The following files/folders are ignored by Git:

```bash
/__pycache__
/.venv
/data
/dataset
/runs
cnn_model.pth
eval_iou.py
preprocessed_image.png
ssd_digits.pth
```

Because of this, a fresh clone of the repository will not include:

- the custom detection dataset,
- the MNIST data downloaded for CNN training,
- the trained CNN weights,
- the trained TinySSD weights,
- local experiment outputs,
- temporary preprocessing images,
- the local evaluation script.

To run the project after cloning it, you must either:

1. train the models locally, or
2. manually place the trained `.pth` files in the root folder of the project.

Required local model files before running `main.py`:

```bash
cnn_model.pth
ssd_digits.pth
```

---

## Dataset

The detection dataset is custom-made for this project and is not included in the GitHub repository.

You need to create the following local folder structure:

```bash
dataset/
├── images/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── annotations/
    ├── image_1.json
    ├── image_2.json
    └── ...
```

Each image must have a matching JSON annotation file with the same name.

Example:

```bash
dataset/images/number_01.jpg
dataset/annotations/number_01.json
```

The annotation files contain the bounding boxes of the digits.

### Dataset Limitation

The current dataset contains a small amount of data. This limits the robustness of the detection model, especially in real-time conditions.

To improve the model, it is recommended to add:

- more images,
- different backgrounds,
- different lighting conditions,
- different digit sizes,
- different writing styles,
- rotated or slightly distorted digits,
- examples with multiple digits in the same image.

A larger and more diverse dataset will help the model generalize better.

---

## Project Structure

The GitHub repository contains the source code only. Some generated files are ignored by Git.

```bash
MultiNumbersDetection/
├── camera.py              # Camera setup and frame capture
├── CNN.py                 # CNN training script for digit recognition
├── CNN_model.py           # CNN architecture and local weight loading
├── dataset.py             # Custom dataset class and data augmentation
├── main.py                # Real-time detection and recognition script
├── model.py               # TinySSD detection model architecture
├── preprocessing.py       # Image preprocessing before CNN prediction
├── train.py               # Training script for the TinySSD detection model
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files and folders
└── README.md              # Project documentation
```

Ignored/generated local files:

```bash
dataset/                 # Custom local dataset
data/                    # MNIST data downloaded by torchvision
runs/                    # Local experiment outputs
cnn_model.pth            # Trained CNN weights
ssd_digits.pth           # Trained TinySSD weights
preprocessed_image.png   # Temporary image generated during preprocessing
eval_iou.py              # Local evaluation script, not included in Git
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- Albumentations
- D2L
- PIL
- NumPy

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lvdoo/MultiNumbersDetection.git
cd MultiNumbersDetection
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Models

Since the trained `.pth` files are ignored by Git, you need to generate them locally before running the project.

### Train the digit recognition model

```bash
python CNN.py
```

This trains the CNN model and creates:

```bash
cnn_model.pth
```

The script uses MNIST through `torchvision.datasets.MNIST`. The downloaded MNIST data is stored locally in the ignored `data/` folder.

### Train the digit detection model

Before training the detection model, make sure your local dataset exists:

```bash
dataset/images/
dataset/annotations/
```

Then run:

```bash
python train.py
```

This trains the TinySSD detection model and creates:

```bash
ssd_digits.pth
```

---

## Running the Project

Before running the project, make sure these files exist locally in the root folder:

```bash
cnn_model.pth
ssd_digits.pth
```

Then start the real-time detection system:

```bash
python main.py
```

Press `q` to quit the camera window.

---

## Camera Configuration

By default, the camera is opened with:

```python
cv.VideoCapture(1)
```

If the camera does not open, change the camera index in `camera.py`:

```python
cv.VideoCapture(0)
```

This depends on your computer and connected webcam devices.

---

## Current Limitations

- The custom detection dataset is small.
- The dataset is not included in the GitHub repository.
- The trained model weights are not included in the GitHub repository.
- A fresh clone cannot run directly without training or manually adding the `.pth` files.
- The detection model may struggle with unseen backgrounds or lighting conditions.
- Recognition accuracy depends heavily on the quality of the detected crop.
- The project detects individual digits, not full multi-digit number sequences.

---

## Possible Improvements

- Add more images to the custom dataset.
- Improve annotation quality.
- Add more varied real-world conditions.
- Add more data augmentation.
- Tune the confidence threshold.
- Improve bounding box precision.
- Add support for reading complete multi-digit numbers.
- Add a public download link for trained weights.
- Add a small sample dataset for demonstration.
- Add a dedicated evaluation script to the repository if needed.

---

## Author

Project developed by Lvdoo
