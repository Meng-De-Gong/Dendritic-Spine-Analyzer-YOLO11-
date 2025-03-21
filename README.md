# Dendritic-Spine-Analyzer-YOLO11  

<h2 align="center">
    <a href="./README.md"><strong>English</strong></a> | 
    <a href="./README_CN.md"><strong>简体中文</strong></a>
</h2>

You can download the software from the **Releases** section on the right sidebar or **click the link** below:  

<p align="center">  
    👉 <a href="https://github.com/Meng-De-Gong/Dendritic-Spine-Analyzer-YOLO11-/releases/tag/v1.0.1"><strong>GUI-DendriticSpineClassification v1.0.1</strong></a>  
</p>

---

##  Project Overview
This project is developed based on the cutting-edge YOLO11 target detection framework, focusing on the automated recognition and classification of dendritic spine morphology.

##  Description of relevant documents
### 🗂️ **datasets**  
This folder contains the dataset files used for YOLO11 training. It includes:  
- **spines_train** – Training set  
- **spines_val** – Validation set  
- **spines_test** – Test set  

---

### 🏃‍ **runs**  
This folder serves as the directory for storing model weights and evaluation files during training and prediction. It includes:  
- **train** – Stores training-related files  
- **detect** – Stores prediction-related files  

👉 The trained weights are saved at:  
`runs/train/exp2/weights/best.pt`  

---

### 💾 **save_data**  
The save_data folder is used to store files generated while running the UI operation.  

---

### 📂 **Testdata**  
This folder contains dendritic spine image files used for testing in the UI.

---

### 🎨 **UIProgram**  
This folder contains the PyQt5-based UI program files.  

---

### ⚙ **Config.py**  
This file defines the configuration settings for dendritic spine detection.  

---

### 🔎 **detect.py**  
This file contains the YOLO11-based prediction code.  

---

### 🛠️ **detect_tools.py**  
This file handles the annotation and classification of detected dendritic spines.  

---

### 🚀 **MainProgram.py**  
This is the **main program**.  
👉 **Run this file to start the UI.**  

---

### 📋 **requirements.txt**  
This file lists the required dependencies and environment configurations.

👉 To install the required dependencies, run the following command:  
`pip install -r requirements.txt`

---

### 🎯 **train.py**  
This file contains the model training code.  
