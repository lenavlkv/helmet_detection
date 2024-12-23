import torch
import cv2
import os
import numpy as np
from collections import defaultdict
import torchvision.ops as ops

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Папка с изображениями и аннотациями
image_folder = 'images/val'
label_folder = 'labels/val'

# Извлекаем список изображений (файлы с расширением .jpg или .png)
img_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]


# Функция для вычисления IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


# Функция для загрузки аннотаций из текстового файла
def load_annotations(img_name):
    label_path = os.path.join(label_folder, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    annotations = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                # Преобразуем в [x1, y1, x2, y2] (относительные координаты в абсолютные)
                img_width, img_height = 640, 640  # Предположим, что все изображения 640x640
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height
                annotations.append([x1, y1, x2, y2, class_id])

    return annotations


# Функция для обработки изображения
def evaluate_image_with_nms(img_name):
    img_path = os.path.join(image_folder, img_name)
    img_cv = cv2.imread(img_path)

    # Преобразуем в RGB (YOLO принимает изображения в RGB)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    model.eval()

    with torch.no_grad():
        # Предсказания YOLO
        results = model(img_rgb)
        predictions = results.pred[0]

        # Список классов для меток
        class_names = model.names

        # Загрузка аннотаций для этого изображения (ground truth)
        anns = load_annotations(img_name)

        boxes = predictions[:, :4]  # Координаты bounding boxes [x1, y1, x2, y2]
        scores = predictions[:, 4]  # Уверенность предсказания
        labels = predictions[:, 5]  # Классы

        # NMS
        nms_indices = ops.nms(boxes, scores, 0.4)  # 0.4 - порог IoU для NMS
        nms_boxes = boxes[nms_indices]
        nms_scores = scores[nms_indices]
        nms_labels = labels[nms_indices]

        # Словарь для хранения IoU для каждого объекта
        iou_results = defaultdict(list)

        # Сохраняем предсказания и ground truth для дальнейшей оценки
        for ann in anns:
            gt_box = ann[:4]  # Координаты [x1, y1, x2, y2]
            gt_class = ann[4]
            iou_results['gt'].append((gt_box, gt_class))

        for i, pred in enumerate(nms_boxes):
            x1, y1, x2, y2 = map(int, pred)
            confidence = nms_scores[i].item()
            label = int(nms_labels[i].item())
            pred_box = [x1, y1, x2, y2]

            max_iou = 0
            best_gt = None
            for gt_box, gt_class in iou_results['gt']:
                iou = compute_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt_class

            # Записываем результаты
            if max_iou >= 0.5:  # Точный порог IoU для определения правильного предсказания
                iou_results[label].append(1)  # True positive
            else:
                iou_results[label].append(0)  # False positive

    return iou_results

# Процесс по всем изображениям в папке val2017
all_results = defaultdict(list)

for img_name in img_files:
    img_results = evaluate_image_with_nms(img_name)
    for label, results in img_results.items():
        if label != 'gt':  # Только классы, а не "gt"
            all_results[label].extend(results)

# Подсчет mAP
def compute_map(results):
    aps = {}
    for label, result in results.items():
        tp = np.array(result)
        fp = 1 - tp
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recall = cumsum_tp / np.sum(tp)  # Total number of ground truth objects
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)

        # Интерполяция для расчета AP
        recall_interpolated = np.linspace(0, 1, 101)
        precision_interpolated = np.interp(recall_interpolated, recall[::-1], precision[::-1])

        ap = np.mean(precision_interpolated)
        aps[label] = ap

    mAP = np.mean(list(aps.values()))  # Среднее значение по всем классам
    return aps, mAP

# Вычисление mAP для всех классов
aps, mAP = compute_map(all_results)
print(f"Mean Average Precision (mAP): {mAP:.4f}")
for label, ap in aps.items():
    print(f"AP for class {label}: {ap:.4f}")