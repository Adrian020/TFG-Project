import os
from PIL import Image, ImageEnhance
import json
import random
import copy


def preprocessing(image_list, img_out, label_out, annotations, categories):
    images_path = "/home/amartinez/TFG/cct_images/"

    for image in image_list:
        try:
            img_path = os.path.join(images_path, image)
            img = Image.open(img_path)
            
            original_width, original_height = img.size
            img = img.resize((640, 640))
            img.save(os.path.join(img_out, image))

            image_id = image.split(".")[0]
            
            label_file = os.path.join(label_out, image.replace(".jpg", ".txt"))

            with open(label_file, "w") as f:    
                if image_id in annotations:
                    for ann in annotations[image_id]:
                        cat_id = ann["category_id"]
                        category_name = categories[cat_id]
                        if category_name == "empty" or category_name == "car":
                                continue
                        
                        match category_name:
                            case "dog": cat_num = 0
                            case "cat": cat_num = 1
                            case "deer": cat_num = 2
                            case "squirrel": cat_num = 3
                            case "raccoon": cat_num = 4
                            case "bobcat": cat_num = 5
                            case "coyote": cat_num = 6
                            case "bird": cat_num = 7
                            case "rabbit": cat_num = 8
                            case "opossum": cat_num = 9
                            case _: cat_num = 10
                        
                        x, y, w, h = ann["bbox"]

                        scale_x = (640 / original_width)
                        scale_y = (640 / original_height)

                        x *= scale_x
                        y *= scale_y
                        w *= scale_x
                        h *= scale_y

                        x_center = (x + w / 2) / 640
                        y_center = (y + h / 2) / 640
                        w /= 640
                        h /= 640

                        f.write(f"{cat_num} {x_center} {y_center} {w} {h}\n")
        except Exception as e:
            print(f"Error de {image}: {e}")
            

def dataAugmentation(img, aug_type):
    if aug_type == "flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT), 0

    elif aug_type == "crop":
        width, height = img.size

        crop_percent = random.uniform(0.1, 0.2)

        left = int(width * crop_percent)
        top = int(height * crop_percent)
        right = int(width * (1 - crop_percent))
        bottom = int(height * (1 - crop_percent))

        return img.crop((left, top, right, bottom)), crop_percent
        
    elif aug_type == "bright":
        factor = random.uniform(0.8, 1.2)
        
        return ImageEnhance.Brightness(img).enhance(factor), 0
    
    elif aug_type == "contrast":
        factor = random.uniform(0.8, 1.2)
        
        return ImageEnhance.Contrast(img).enhance(factor), 0

    return img, 0
    

def changeAnnotations(annotations, crop_percent, width, height):
    new_annotations = []
    
    for ann in annotations:
        ann = ann.copy()
        x, y, w, h = ann["bbox"]
    
        if crop_percent == 0:
            new_x = width - (x + w)
            ann["bbox"] = [new_x, y, w, h]
            new_annotations.append(ann)
            
        else:
            new_x = x - (width * crop_percent)
            new_y = y - (height * crop_percent)
            new_width = width * (1 - 2 * crop_percent)
            new_height = height * (1 - 2 * crop_percent)

            new_width = min(new_width, new_x + w) - max(0, new_x)
            new_height = min(new_height, new_y + h) - max(0, new_y)
            
            if new_width > 0 and new_height > 0:
                ann["bbox"] = [max(0, new_x), max(0, new_y), new_width, new_height]
                new_annotations.append(ann)
                
    return new_annotations
                


def createDataset():
    images_path = "/home/amartinez/TFG/cct_images/"
    json_path = "/home/amartinez/TFG/caltech_bboxes_20200316.json"
    output_path = "/home/amartinez/TFG/dataset"
    
    especies_train_size = 3300
    especies_val_size = 700
    extra_size = 10000

    train_img_path = os.path.join(output_path, "images/train")
    val_img_path = os.path.join(output_path, "images/val")
    train_label_path = os.path.join(output_path, "labels/train")
    val_label_path = os.path.join(output_path, "labels/val")

    for path in [train_img_path, val_img_path, train_label_path, val_label_path]:
        os.makedirs(path, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    categories = {}
    for cat in data["categories"]:
        categories[cat["id"]] = cat["name"]

    annotations = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(ann)

    species = [
        "dog",
        "cat",
        "deer",
        "squirrel",
        "raccoon",
        "bobcat",
        "coyote",
        "bird",
        "rabbit",
        "opossum",
    ]

    train_images = set()
    val_images = set()

    for specie in species:
        specie_imgs = set()

        for ann in data["annotations"]:
            category_name = categories[ann["category_id"]]

            if category_name == specie:
                specie_imgs.add(ann["image_id"] + ".jpg")

        specie_imgs = list(specie_imgs)
        random.shuffle(specie_imgs)
        
        val_part = specie_imgs[:especies_val_size]
        train_part = specie_imgs[especies_val_size:especies_val_size+especies_train_size]

        train_images.update(train_part)
        val_images.update(val_part)
        
        missing = especies_train_size - len(train_part)

        if missing > 0:
            print(f"{specie} necesita {missing} augmentations")
        
            original_train = train_part.copy()
        
            aug_count = 0
        
            while aug_count < missing:
                img_name = random.choice(original_train)
        
                aug_name = f"aug{aug_count}_{img_name}"
        
                src_path = os.path.join(images_path, img_name)
                dst_path = os.path.join(images_path, aug_name)
        
                try:
                    img = Image.open(src_path)
        
                    aug_type = random.choice(["flip", "crop", "bright", "contrast"])
        
                    aug_img, crop_percent = dataAugmentation(img, aug_type)
        
                    aug_img.save(dst_path)
        
                    train_images.add(aug_name)
        
                    original_id = img_name.replace(".jpg", "")
                    new_id = aug_name.replace(".jpg", "")
        
                    if original_id in annotations:
                        if aug_type == "bright" or aug_type == "contrast":
                            annotations[new_id] = annotations[original_id]
                            
                        else:  
                            original_width, original_height = img.size
                            annotations[new_id] = changeAnnotations(annotations[original_id], crop_percent, original_width, original_height)  
                                                                                         
                    aug_count += 1
        
                except Exception as e:
                    print(e)

    all_images = set(os.listdir(images_path))
    used_images = train_images.union(val_images)
    remaining = list(all_images - used_images)

    random.shuffle(remaining)
    extra_train = remaining[:extra_size]

    train_images.update(extra_train)

    train_images = list(train_images)
    val_images = list(val_images)

    preprocessing(train_images, train_img_path, train_label_path, annotations, categories)
    preprocessing(val_images, val_img_path, val_label_path, annotations, categories)