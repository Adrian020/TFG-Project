from ultralytics import YOLO

def detectionModel():
    model = YOLO('yolov5su.pt')

    model.train(
        data="/home/amartinez/TFG/dataset/dataset.yaml",
        epochs=40,
        imgsz=640,
        batch=32,
        device=0,
        workers=2,
        
        #freeze = 24,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.1,
        weight_decay=0.0005,
        cos_lr=True,
        #conf=0.4,
        patience=15,
    )

    model.val(
        data="/home/amartinez/TFG/dataset/dataset.yaml",
        save=True,
        workers=2,
    )

    model.predict(
        source="/home/amartinez/TFG/dataset/images/val/",
        save=True,
        stream=True,
    )
    
def predictImage():
    model = YOLO("/home/amartinez/TFG/runs/detect/train-12/weights/best.pt")
    img_path = r"D:\TFG\cct_images\cct_images\5a2e1320-23d2-11e8-a6a3-ec086b02610b.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))

    results = model.predict(
        source=img,
        save=True,
    )
