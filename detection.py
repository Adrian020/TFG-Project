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