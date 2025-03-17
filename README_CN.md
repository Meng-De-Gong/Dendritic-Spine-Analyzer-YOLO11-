# Dendritic-Spine-Analyzer-YOLO11  

<h2 align="center">
    <a href="./README.md"><strong>English</strong></a> | 
    <a href="./README_CN.md"><strong>简体中文</strong></a>
</h2>

您可以从右侧边栏的**releases**部分下载软件，或点击如下**链接**

<p align="center">  
    👉 <a href="https://github.com/Meng-De-Gong/Dendritic-Spine-Analyzer-YOLO11-/releases/tag/v1.0.1"><strong>GUI-DendriticSpineClassification v1.0.1</strong></a>  
</p>

---

##  项目概述
该项目基于最前沿的 YOLO11 目标检测框架开发，侧重于树突棘形态的自动识别和分类。

##  相关文档说明
### 🗂️ **datasets**  
此文件夹包含用于 YOLO11 训练的数据集文件。其中包括： 
- **spines_train** – **训练集**  
- **spines_val** – **验证集**  
- **spines_test** – **测试集**  

---

### 🏃‍ **runs**  
此文件夹用作在训练和预测期间存储模型权重和评估文件的目录。它包括：
- **train** – **存储与训练相关的文件**  
- **detect** – **存储与预测相关的文件**  

👉 训练好的权重保存至：
`runs/train/exp2/weights/best.pt`  

---

### 💾 **save_data**  
save_data 文件夹用于存储运行 UI 操作时生成的文件。  

---

### 📂 **Testdata**  
该文件夹包含用于用户界面测试的树突棘图像文件。

---

### 🎨 **UIProgram**  
此文件夹包含基于 PyQt5 的用户界面（UI）程序文件。

---

### ⚙ **Config.py**  
此文件定义了树突棘检测的配置设置。  

---

### 🔎 **detect.py**  
此文件包含基于 YOLO11 的预测代码。

---

### 🛠️ **detect_tools.py**  
此文件用于处理检测到的树突棘的标注和分类。

---

### 🚀 **MainProgram.py**  
这是**主程序**.  
👉 **运行此文件以启动用户界面。**  

---

### 📋 **requirements.txt**  
该文档列出了所需的依赖项和环境配置。

👉要安装所需的依赖项，请运行以下命令：  
`pip install -r requirements.txt`


---

### 🎯 **train.py**  
此文件为模型训练代码。
