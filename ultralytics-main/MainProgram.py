import time
from PyQt5.QtWidgets import QApplication,\
    QMessageBox, QWidget, QFileDialog, QHeaderView, QTableWidgetItem, QAbstractItemView, QFileSystemModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QDir, QModelIndex
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_Form
from PyQt5.QtCore import Qt, QCoreApplication
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
import numpy as np
import torch
import pandas as pd
from datetime import datetime

def resource_path(relative_path):
    """ 获取资源的绝对路径（兼容打包后的路径） """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
class Form(QWidget):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()

        # 当前文件夹路径
        self.folder_path = ""
        self.img_width = 600  # 显示宽度
        self.img_height = 400  # 显示高度

        # 初始化文件系统模型，并设置过滤器只显示目录
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setFilter(QDir.Dirs | QDir.NoDotAndDotDot)  # 只显示目录，不显示.和..
        self.file_system_model.setRootPath('')

        # 设置 treeView
        self.ui.treeView.setModel(self.file_system_model)
        self.ui.treeView.setRootIndex(self.file_system_model.index(''))
        self.ui.treeView.setHeaderHidden(True)  # 隐藏头部
        # 可选：隐藏其他列
        self.ui.treeView.setColumnHidden(1, True)
        self.ui.treeView.setColumnHidden(2, True)
        self.ui.treeView.setColumnHidden(3, True)

        # 加载 css 渲染效果
        style_file = resource_path('style.css')
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

        self.conf = 0.25
        self.iou = 0.7

        # 当前图片索引和图片列表
        self.current_index = -1
        self.image_list = []

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.SaveBtn.clicked.connect(self.save_detect_image)
        self.ui.SaveExcelBtn.clicked.connect(self.export_to_excel)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.treeView.clicked.connect(self.on_treeview_item_clicked)
        self.ui.listView.clicked.connect(self.on_listview_item_clicked)

        # 添加上一张和下一张按钮的点击事件
        self.ui.UpBtn.clicked.connect(self.show_previous_image)
        self.ui.DownBtn.clicked.connect(self.show_next_image)

    def initMain(self):
        self.show_width = 800
        self.show_height = 600
        self.org_path = None
        self.is_camera_open = False
        self.cap = None
        self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO(resource_path(Config.model_path), task='detect')
        self.model(np.zeros((48, 48, 3)), device=self.device)  # 预先加载推理模型
        self.fontC = ImageFont.truetype(resource_path("platech.ttf"), 16, 0)

        # 表格
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(45)
        self.ui.tableWidget.setColumnWidth(0, 90)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 160)  # 调整列宽
        self.ui.tableWidget.setColumnWidth(2, 100)
        self.ui.tableWidget.setColumnWidth(3, 240)
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

    def on_treeview_item_clicked(self, index: QModelIndex):
        folder_path = self.file_system_model.filePath(index)  # 获取当前选中文件夹路径
        self.folder_path = folder_path
        if os.path.isdir(folder_path):
            self.update_listview(folder_path)

    def update_listview(self, folder_path):
        if not os.path.isdir(folder_path):
            QMessageBox.warning(self, 'error', 'The specified path is not a valid directory.')
            return
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp', 'tif']
        self.image_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.split('.')[-1].lower() in img_suffix]
        model = QStandardItemModel()
        for img_file in self.image_list:
            item = QStandardItem(img_file)
            model.appendRow(item)
        self.ui.listView.setModel(model)

    def clear_list_view(self):
        model = self.ui.listView.model()
        if model:
            model.clear()

    def on_listview_item_clicked(self, index):
        img_file = index.data()
        img_path = os.path.join(self.folder_path, img_file)
        img_path = os.path.normpath(img_path)
        self.current_index = self.image_list.index(img_file)  # 更新当前索引
        self.org_path = img_path  # 更新当前图片路径
        self.display_image(img_path)
        self.perform_detection(img_path)
        self.ui.PiclineEdit.setText(img_path)

    def display_image(self, img_path):
        if not os.path.exists(img_path):
            print(f"Error: File {img_path} not found.")
            return
        img = cv2.imread(img_path)
        self.img_width, self.img_height = self.get_resize_size(img)
        resize_cvimg = cv2.resize(img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show_2.setPixmap(pix_img)
        self.ui.label_show_2.setAlignment(Qt.AlignCenter)

    def perform_detection(self, img_path):
        t1 = time.time()
        self.results = self.model(img_path, conf=self.conf, iou=self.iou)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)
        now_img = self.results.plot()
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        self.update_info_table()

    def update_info_table(self):
        location_list = self.results.boxes.xyxy.tolist()
        location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        cls_list = [int(i) for i in cls_list]
        conf_list = self.results.boxes.conf.tolist()
        conf_list = ['%.2f %%' % (each * 100) for each in conf_list]
        target_nums = len(cls_list)
        self.ui.label_nums.setText(str(target_nums))
        Mushroom_nums = cls_list.count(0)
        Stubby_nums = cls_list.count(1)
        Ball_nums = cls_list.count(2)
        Thin_nums = cls_list.count(3)
        Filopodia_nums = cls_list.count(4)
        self.ui.label_num3.setText(str(Mushroom_nums))
        self.ui.label_num2.setText(str(Stubby_nums))
        self.ui.label_num0.setText(str(Ball_nums))
        self.ui.label_num1.setText(str(Thin_nums))
        self.ui.label_num4.setText(str(Filopodia_nums))
        choose_list = ['ALL'] + [Config.names[id] + '_' + str(index) for index, id in enumerate(cls_list)]
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)
        if target_nums >= 1:
            self.ui.type_lb.setText(Config.EN_names[cls_list[0]])
            self.ui.label_conf.setText(conf_list[0])
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.populate_info_table(location_list, cls_list, conf_list)

    def open_img(self):
        file_path, _ = QFileDialog.getOpenFileName(None, 'Open image', './', "Image files (*.jpg *.jpeg *.png *.bmp *.tif)")
        if not file_path:
            return
        self.org_path = file_path
        dir_path = os.path.dirname(file_path)
        self.folder_path = dir_path
        index = self.file_system_model.index(dir_path)
        self.ui.treeView.setCurrentIndex(index)
        self.update_listview(dir_path)
        self.perform_detection(file_path)
        self.display_image(self.org_path)
        self.ui.PiclineEdit.setText(self.org_path)

    def show_image_by_index(self, index):
        if 0 <= index < len(self.image_list):
            img_file = self.image_list[index]
            img_path = os.path.join(self.folder_path, img_file)
            self.current_index = index
            self.org_path = img_path  # 更新当前图片路径
            self.display_image(img_path)
            self.perform_detection(img_path)
            self.ui.PiclineEdit.setText(img_path)
            # 更新 listView 的选中项
            self.ui.listView.setCurrentIndex(self.ui.listView.model().index(index, 0))

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image_by_index(self.current_index)

    def show_next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image_by_index(self.current_index)

    def combox_change(self):
        com_text = self.ui.comboBox.currentText()
        if com_text == '全部':
            if hasattr(self.results, 'boxes') and len(self.results.boxes.cls) > 0:
                cur_box = self.results.boxes.xyxy.tolist()
                cur_img = self.results.plot()
                self.ui.type_lb.setText(Config.EN_names[int(self.results.boxes.cls.tolist()[0])])
                self.ui.label_conf.setText('%.2f %%' % (self.results.boxes.conf.tolist()[0] * 100))
            else:
                self.ui.type_lb.setText('')
                self.ui.label_conf.setText('')
        else:
            try:
                index = int(com_text.split('_')[-1])
                if hasattr(self.results, 'boxes') and 0 <= index < len(self.results.boxes.cls):
                    cur_box = [self.results.boxes.xyxy.tolist()[index]]
                    cur_img = self.results[index].plot()
                    self.ui.type_lb.setText(Config.EN_names[int(self.results.boxes.cls.tolist()[index])])
                    self.ui.label_conf.setText('%.2f %%' % (self.results.boxes.conf.tolist()[index] * 100))
                else:
                    self.ui.type_lb.setText('')
                    self.ui.label_conf.setText('')
            except (ValueError, IndexError) as e:
                print(f"combox_change 出错: {e}")
                self.ui.type_lb.setText('')
                self.ui.label_conf.setText('')

        # 显示图像
        if hasattr(self, 'img_width') and hasattr(self, 'img_height'):
            resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

    def populate_info_table(self, locations, clses, confs):
        data = []
        for i, (location, cls, conf) in enumerate(zip(locations, clses, confs), start=1):
            data.append([
                str(i),
                Config.EN_names[cls],
                conf,
                str(location)
            ])
        self.ui.tableWidget.setRowCount(len(data))
        for row, row_data in enumerate(data):
            for col, item in enumerate(row_data):
                cell_item = QTableWidgetItem(item)
                cell_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.ui.tableWidget.setItem(row, col, cell_item)
        self.ui.tableWidget.scrollToBottom()

    def get_resize_size(self, img):
        height, width, _ = img.shape
        ratio = width / height
        if ratio >= self.show_width / self.show_height:
            img_width = self.show_width
            img_height = int(img_width / ratio)
        else:
            img_height = self.show_height
            img_width = int(img_height * ratio)
        return img_width, img_height

    def save_detect_image(self):
        if not self.org_path:
            QMessageBox.about(self, 'Tip', 'There is currently no information to save. Please open the image or folder first!')
            return
        if os.path.isfile(self.org_path):
            name, ext = os.path.splitext(os.path.basename(self.org_path))
            save_name = f"{name}_detect_result{ext}"
            save_img_path = os.path.join(Config.save_path, save_name)
            results = self.model(self.org_path, conf=self.conf, iou=self.iou)[0]
            now_img = results.plot()
            cv2.imwrite(save_img_path, now_img)
            QMessageBox.about(self, 'Tip', f'Image saved successfully!\nFile path:{save_img_path}')
        elif os.path.isdir(self.org_path):
            img_suffix = ['jpg', 'png', 'jpeg', 'bmp', 'tif']
            for file_name in os.listdir(self.org_path):
                full_path = os.path.join(self.org_path, file_name)
                if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                    name, ext = os.path.splitext(file_name)
                    save_name = f"{name}_detect_result{ext}"
                    save_img_path = os.path.join(Config.save_path, save_name)
                    results = self.model(full_path, conf=self.conf, iou=self.iou)[0]
                    now_img = results.plot()
                    cv2.imwrite(save_img_path, now_img)
            QMessageBox.about(self, 'Tip', f'All pictures have been saved successfully!\nFile save path: {Config.save_path}')
        else:
            QMessageBox.about(self, 'Tip', 'Invalid file or folder path!')

    def export_to_excel(self):
        if not self.org_path:
            QMessageBox.about(self, 'Tip', 'There is currently no information to save. Please open the image or folder first!')
            return
        save_dir = "save_data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.isdir(self.org_path):
            img_suffix = ['jpg', 'png', 'jpeg', 'bmp', 'tif']
            all_data = []
            summary_data_all = []
            category_counts_summary = {}
            for file_name in os.listdir(self.org_path):
                full_path = os.path.join(self.org_path, file_name)
                if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                    results = self.model(full_path, conf=self.conf, iou=self.iou)[0]
                    location_list = [list(map(int, box)) for box in results.boxes.xyxy.tolist()]
                    cls_list = [int(cls) for cls in results.boxes.cls.tolist()]
                    conf_list = ['%.2f %%' % (conf * 100) for conf in results.boxes.conf.tolist()]
                    data = {
                        "Name": [file_name] * len(cls_list),
                        "Category": [Config.EN_names[cls] for cls in cls_list],
                        "Confidence": conf_list,
                        "X_min": [loc[0] for loc in location_list],
                        "Y_min": [loc[1] for loc in location_list],
                        "X_max": [loc[2] for loc in location_list],
                        "Y_max": [loc[3] for loc in location_list]
                    }
                    all_data.append(pd.DataFrame(data))
                    total_count = len(cls_list)
                    category_counts = {Config.EN_names[i]: cls_list.count(i) for i in set(cls_list)}
                    summary_data_all.append([file_name, total_count])
                    for category, count in category_counts.items():
                        category_counts_summary[category] = category_counts_summary.get(category, 0) + count
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                summary_df = pd.DataFrame(summary_data_all, columns=["Image Name", "Dendritic spines total number"])
                category_counts_summary_df = pd.DataFrame([[category, count] for category, count in category_counts_summary.items()], columns=["类型", "值"])
                summary_file_path = os.path.join(save_dir, "Summary file.xlsx")
                with pd.ExcelWriter(summary_file_path) as writer:
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    category_counts_summary_df.to_excel(writer, sheet_name="Summary", index=False, startrow=len(summary_df) + 2)
                    combined_df.to_excel(writer, sheet_name="Details", index=False)
                QMessageBox.about(self, 'Tip', f'All data has been successfully exported to {save_dir}')
        else:
            results = self.model(self.org_path, conf=self.conf, iou=self.iou)[0]
            location_list = [list(map(int, box)) for box in results.boxes.xyxy.tolist()]
            cls_list = [int(cls) for cls in results.boxes.cls.tolist()]
            conf_list = ['%.2f %%' % (conf * 100) for conf in results.boxes.conf.tolist()]
            data = {
                "Image Name": [os.path.basename(self.org_path)] * len(cls_list),
                "Dendritic spine type": [Config.EN_names[cls] for cls in cls_list],
                "Confidence level": conf_list,
                "X_min": [loc[0] for loc in location_list],
                "Y_min": [loc[1] for loc in location_list],
                "X_max": [loc[2] for loc in location_list],
                "Y_max": [loc[3] for loc in location_list]
            }
            df = pd.DataFrame(data)
            base_name = os.path.splitext(os.path.basename(self.org_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d")
            file_path = os.path.join(save_dir, f"{base_name}_{timestamp}.xlsx")
            df.to_excel(file_path, index=False)
            QMessageBox.about(self, 'Tip', f'The data has been successfully exported to {file_path}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Form()
    win.show()
    sys.exit(app.exec_())