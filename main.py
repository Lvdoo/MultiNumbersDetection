import cv2 as cv
import torch
import torchvision.transforms as T

from camera import open_camera, read_frame, release_video
from model import TinySSD
from d2l import torch as d2l
from PIL import Image
from CNN_model import model
from preprocessing import preprocess

MODEL_PATH = "ssd_digits.pth"
CONF_THRESHOLD = 0.15
MARGIN = 10

softmax = torch.nn.Softmax(dim=1)

def resize_frame_with_padding(frame, target_size=256):
    img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    w, h = img.size

    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    img_resized = img.resize((new_w, new_h))
    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    new_img.paste(img_resized, (pad_x, pad_y))

    return new_img, scale, pad_x, pad_y


def predict(frame, net):
    """Prédiction bbox SSD pour une frame"""

    img_padded, scale, pad_x, pad_y = resize_frame_with_padding(frame, 256)

    X = T.ToTensor()(img_padded).unsqueeze(0)

    with torch.no_grad():
        anchors, cls_preds, bbox_preds = net(X)
        cls_probs = torch.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = d2l.multibox_detection(cls_probs, bbox_preds, anchors, nms_threshold = 0.15)

    detections = output[0].cpu()
    boxes = []

    for det in detections:
        class_id = int(det[0].item())
        score = float(det[1].item())
        if class_id == -1 or score < CONF_THRESHOLD:
            continue
        xmin, ymin, xmax, ymax = det[2:6]
        xmin = max(0, min(1, xmin.item()))
        ymin = max(0, min(1, ymin.item()))
        xmax = max(0, min(1, xmax.item()))
        ymax = max(0, min(1, ymax.item()))

        xmin = xmin * 256
        xmax = xmax * 256
        ymin = ymin * 256
        ymax = ymax * 256

        xmin = int((xmin - pad_x) / scale)
        xmax = int((xmax - pad_x) / scale)
        ymin = int((ymin - pad_y) / scale)
        ymax = int((ymax - pad_y) / scale)
        boxes.append((xmin, ymin, xmax, ymax, score))
    return boxes


def main():
    video = open_camera()
    net = TinySSD(num_classes=1)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()

    while True:
        ok, frame = read_frame(video)
        if not ok:
            print("Can't receive frame. Exiting...")
            break

        boxes = predict(frame, net)

        annotated_frame = frame.copy()
        for xmin, ymin, xmax, ymax, score in boxes:

            # Chech da+++
            xmin = max(0, xmin - MARGIN)
            ymin = max(0, ymin - MARGIN)
            xmax = min(frame.shape[1], xmax + MARGIN)
            ymax = min(frame.shape[0], ymax + MARGIN)

            # Vérifier que la box est valide
            if xmax <= xmin or ymax <= ymin:
                continue

            # Crop correct : frame[y, x]
            crop = frame[ymin:ymax, xmin:xmax]

            # Vérifier que le crop n'est pas vide
            if crop is None or crop.size == 0:
                continue
            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            input_tensor = preprocess(binary)

            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_digit = torch.argmax(outputs, dim=1).item()
                cv.putText(
                annotated_frame,
                f"{predicted_digit}",
                (xmin, max(0, ymin - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2)
            cv.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        cv.imshow("SSD digits detection", annotated_frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    release_video(video)


if __name__ == "__main__":
    main()