import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt


class YOLOWrapper(torch.nn.Module):
    def __init__(self, model, index):
        super().__init__()
        self.model = model
        self.index = index

    def forward(self, x):
        predictions = self.model(x)[0]
        return predictions[self.index].unsqueeze(0)



def activationMap():
    image_path = "/home/amartinez/TFG/cct_images/5a0b0374-23d2-11e8-a6a3-ec086b02610b.jpg"
    weights_path = YOLO("/home/amartinez/TFG/runs/detect/train-12/weights/best.pt")
    output_path = "/home/amartinez/TFG/activationMap.jpg"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(weights_path)

    img = np.array(Image.open(image_path).convert("RGB"))
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    results = yolo(image_path)
    bboxes = results[0].boxes

    if len(bboxes) == 0:
        print("No detecciones")
        return

    target_index = torch.argmax(bboxes.conf).item()

    model = yolo.model.to(device)
    model.eval()

    wrapped = YOLOWrapper(model, target_index)
    target_layers = [model.model[20]]

    cam = EigenCAM(model=wrapped, target_layers=target_layers)
    grayscale_cam = cam(tensor, aug_smooth=True, eigen_smooth=True)[0]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    Image.fromarray(cam_image).save(output_path)
    
    
def numSpeciesGraphic():
    json_path = "/home/amartinez/TFG/caltech_bboxes_20200316.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    categories = {}
    bbox_count = {}
    
    for cat in data["categories"]:
        categories[cat["id"]] = cat["name"]

    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        specie = categories[cat_id]

        if specie not in bbox_count:
            bbox_count[specie] = 0

        bbox_count[specie] += 1

    bbox_count = sorted(bbox_count.items(), key=lambda x: x[1], reverse=True)[:10]
    species = [x[0] for x in bbox_count]
    num_annotations = [x[1] for x in bbox_count]

    plt.figure(figsize=(12,6))
    bars = plt.bar(species, num_annotations, color="gray")

    plt.xlabel("Especies", fontsize=16)
    plt.ylabel("Numero de anotaciones", fontsize=16)

    plt.xticks(rotation=30, ha="right", fontsize=14)
    plt.yticks(fontsize=14)

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            str(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=14
        )

    plt.tight_layout()

    plt.savefig(
        "speciesGraphic.jpg",
        dpi=300,
        bbox_inches="tight"
    )