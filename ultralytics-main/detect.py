# -*- coding: utf-8 -*-


from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'C:\Users\96443\Desktop\YOLO11\ultralytics-main\runs\train\exp2\weights\best.pt')
    model.predict(source=r'C:\Users\96443\Desktop\YOLO11\ultralytics-main\spinesdata',
                  save=True,
                  show=True,
                  line_width=1,
                  )