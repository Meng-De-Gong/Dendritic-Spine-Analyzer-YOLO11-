import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'C:\Users\96443\Desktop\YOLO11\ultralytics-main\datasets\yolo11l.yaml')
    model.train(data=r'C:\Users\96443\Desktop\YOLO11\ultralytics-main\datasets\spine.yaml',
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                 device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                line_width=1,
                patience=0,
                )