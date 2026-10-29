"""
Evaluation of the full multi-digit pipeline (TinySSD detector + CNN classifier).

Expected test set (labelme format, images NEVER used for training):
    test_set/images/img_01.jpg
    test_set/annotations/img_01.json   <- one rectangle per digit, label = the digit ("0".."9")

Usage:
    python eval_multidigit.py
    python eval_multidigit.py --images test_set/images --annotations test_set/annotations

Run it from the project folder (needs model.py, CNN_model.py, preprocessing.py,
ssd_digits.pth and cnn_model.pth).
"""
import argparse
import glob
import json
import math
import os
import time

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
from d2l import torch as d2l
from PIL import Image

from model import TinySSD
from CNN_model import model as cnn
from preprocessing import preprocess

# Same settings as main.py, so we measure the system that runs in the demo
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.15
MARGIN = 10
IOU_THRESHOLD = 0.5
AP_MIN_SCORE = 0.01  # low threshold only used to compute AP


# ---------------------------------------------------------------- helpers
def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def wilson(k, n, z=1.96):
    # 95% confidence interval for a proportion (honest with small datasets)
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def ratio(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n * 100:.1f}% ({k}/{n}, 95% CI {lo * 100:.0f}-{hi * 100:.0f}%)"


def average_precision(records, n_gt):
    # records = [(score, is_tp), ...] over the whole test set
    if n_gt == 0 or not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([1 if r[1] else 0 for r in records])
    fp = np.cumsum([0 if r[1] else 1 for r in records])
    recall = tp / n_gt
    precision = tp / (tp + fp)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


def match(gt_boxes, preds):
    """Greedy matching, predictions sorted by score. Returns for each pred the GT index or None."""
    used = set()
    result = []
    for p in preds:
        best_iou, best_gt = 0.0, None
        for g, gt in enumerate(gt_boxes):
            if g in used:
                continue
            iou = box_iou(gt, p[:4])
            if iou > best_iou:
                best_iou, best_gt = iou, g
        if best_gt is not None and best_iou >= IOU_THRESHOLD:
            used.add(best_gt)
            result.append((best_gt, best_iou))
        else:
            result.append((None, best_iou))
    return result


def load_gt(json_path):
    with open(json_path) as f:
        data = json.load(f)
    boxes, labels = [], []
    for shape in data["shapes"]:
        (x1, y1), (x2, y2) = shape["points"][:2]
        boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        labels.append(str(shape["label"]).strip())
    return boxes, labels


# ---------------------------------------------------------------- models (same code as main.py)
def ssd_predict(frame, net):
    img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    w, h = img.size
    scale = 256 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x, pad_y = (256 - new_w) // 2, (256 - new_h) // 2
    canvas = Image.new("RGB", (256, 256), (255, 255, 255))
    canvas.paste(img.resize((new_w, new_h)), (pad_x, pad_y))
    X = T.ToTensor()(canvas).unsqueeze(0)

    with torch.no_grad():
        anchors, cls_preds, bbox_preds = net(X)
        cls_probs = torch.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = d2l.multibox_detection(cls_probs, bbox_preds, anchors, nms_threshold=NMS_THRESHOLD)

    preds = []
    for det in output[0].cpu():
        class_id, score = int(det[0].item()), float(det[1].item())
        if class_id == -1 or score < AP_MIN_SCORE:
            continue
        xmin, ymin, xmax, ymax = [max(0, min(1, v.item())) * 256 for v in det[2:6]]
        xmin = max(0, int((xmin - pad_x) / scale))
        xmax = min(w, int((xmax - pad_x) / scale))
        ymin = max(0, int((ymin - pad_y) / scale))
        ymax = min(h, int((ymax - pad_y) / scale))
        if xmax > xmin and ymax > ymin:
            preds.append([xmin, ymin, xmax, ymax, score])
    return sorted(preds, key=lambda p: -p[4])


def classify(frame, box):
    # margin -> crop -> gray -> Otsu inverted -> preprocess -> CNN (as in main.py)
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0]) - MARGIN)
    y1 = max(0, int(box[1]) - MARGIN)
    x2 = min(w, int(box[2]) + MARGIN)
    y2 = min(h, int(box[3]) + MARGIN)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    with torch.no_grad():
        return torch.argmax(cnn(preprocess(binary)), dim=1).item()


def draw(frame, gt_boxes, gt_labels, preds, digits, ok, path):
    out = frame.copy()
    for b, l in zip(gt_boxes, gt_labels):
        cv.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 180, 0), 2)
        cv.putText(out, f"GT {l}", (int(b[0]), max(12, int(b[1]) - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
    for p, d in zip(preds, digits):
        cv.rectangle(out, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)
        cv.putText(out, f"pred {d} ({p[4]:.2f})", (p[0], min(out.shape[0] - 4, p[3] + 16)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv.imwrite(path, out)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="test_dataset/test_set/images")
    parser.add_argument("--annotations", default="test_dataset/test_set/annotations")
    parser.add_argument("--ssd", default="ssd_digits.pth")
    parser.add_argument("--out", default="eval_out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = TinySSD(num_classes=1)
    net.load_state_dict(torch.load(args.ssd, map_location="cpu"))
    net.eval()

    paths = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(args.images, f"*.{ext}")))
    items = []
    for p in paths:
        j = os.path.join(args.annotations, os.path.splitext(os.path.basename(p))[0] + ".json")
        if os.path.exists(j):
            items.append((p, j))
    if not items:
        raise SystemExit(f"No image + json pair found in {args.images} / {args.annotations}")

    n_img = len(items)
    n_gt = n_pred = tp = 0
    matched_ious = []
    ap_records = []
    gt_cls_ok = gt_cls_total = 0
    pred_cls_ok = pred_cls_total = 0
    e2e_ok = 0
    perfect_images = 0
    gt_conf = np.zeros((10, 10), dtype=int)
    pred_conf = np.zeros((10, 10), dtype=int)
    t_ssd, t_cnn, n_cnn_calls = 0.0, 0.0, 0

    for path, jpath in items:
        frame = cv.imread(path)
        gt_boxes, gt_labels = load_gt(jpath)
        if not all(l.isdigit() and len(l) == 1 for l in gt_labels):
            raise SystemExit(f"{jpath}: every label must be a single digit ('0'..'9'), found {gt_labels}")
        gt_digits = [int(l) for l in gt_labels]

        t0 = time.perf_counter()
        preds_all = ssd_predict(frame, net)
        t_ssd += time.perf_counter() - t0

        # 1) Detection (AP uses every prediction, P/R only those above the demo threshold)
        matches = match(gt_boxes, preds_all)
        ap_records += [(p[4], m[0] is not None) for p, m in zip(preds_all, matches)]
        kept = [(p, m) for p, m in zip(preds_all, matches) if p[4] >= CONF_THRESHOLD]
        preds = [k[0] for k in kept]
        n_gt += len(gt_boxes)
        n_pred += len(preds)

        # 2) CNN on ground-truth boxes (perfect crops)
        for box, digit in zip(gt_boxes, gt_digits):
            pred = classify(frame, box)
            gt_cls_total += 1
            if pred is not None:
                gt_conf[digit, pred] += 1
                gt_cls_ok += int(pred == digit)

        # 3) CNN on predicted boxes + 4) end to end
        digits = []
        image_ok = len(preds) == len(gt_boxes)
        matched_gt = set()
        for p, m in kept:
            t0 = time.perf_counter()
            d = classify(frame, p)
            t_cnn += time.perf_counter() - t0
            n_cnn_calls += 1
            digits.append(d)
            if m[0] is not None:
                tp += 1
                matched_ious.append(m[1])
                matched_gt.add(m[0])
                pred_cls_total += 1
                if d is not None:
                    pred_conf[gt_digits[m[0]], d] += 1
                if d == gt_digits[m[0]]:
                    pred_cls_ok += 1
                    e2e_ok += 1
                else:
                    image_ok = False
            else:
                image_ok = False
        if len(matched_gt) < len(gt_boxes):
            image_ok = False
        perfect_images += int(image_ok)

        tag = "OK" if image_ok else "ERR"
        draw(frame, gt_boxes, gt_labels, preds, digits, image_ok,
             os.path.join(args.out, f"{tag}_{os.path.basename(path)}"))

    # ------------------------------------------------------------ report
    def show(title, matrix):
        print(f"\n{title} (rows = true digit, columns = prediction)")
        print("      " + " ".join(f"{c:>3}" for c in range(10)))
        for d in range(10):
            print(f"  {d} : " + " ".join(f"{v:>3}" for v in matrix[d]))

    ap = average_precision(ap_records, n_gt)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    show("CNN on ground-truth boxes", gt_conf)
    show("CNN on predicted boxes (matched only)", pred_conf)

    print("\n" + "=" * 70)
    print(f"Test set : {n_img} images, {n_gt} digits (conf >= {CONF_THRESHOLD}, NMS {NMS_THRESHOLD}, IoU >= {IOU_THRESHOLD})")
    print("=" * 70)
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Detection recall | {ratio(tp, n_gt)} |")
    print(f"| Detection precision | {ratio(tp, n_pred)} |")
    print(f"| Mean IoU (matched boxes) | {mean_iou:.3f} |")
    print(f"| AP@0.5 | {ap:.3f} |")
    print(f"| CNN on ground-truth boxes | {ratio(gt_cls_ok, gt_cls_total)} |")
    print(f"| CNN on detected boxes | {ratio(pred_cls_ok, pred_cls_total)} |")
    print(f"| End-to-end digit recall | {ratio(e2e_ok, n_gt)} |")
    print(f"| End-to-end digit precision | {ratio(e2e_ok, n_pred)} |")
    print(f"| Images fully correct | {ratio(perfect_images, n_img)} |")
    print(f"| SSD latency / image | {t_ssd / n_img * 1000:.1f} ms |")
    if n_cnn_calls:
        print(f"| CNN latency / digit | {t_cnn / n_cnn_calls * 1000:.1f} ms |")
    print(f"\nAnnotated images saved in '{args.out}/' (green = ground truth, red = prediction, ERR_ = at least one error)")


if __name__ == "__main__":
    main()