import cv2
import torch
from torchvision import datasets
import numpy as np
from collections import defaultdict

svhn = datasets.SVHN(root='./data', split='train', download=True)

selected_indices = []
class_counts = defaultdict(int)

target_per_class = 300
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_color = (255, 255, 255)
line_thickness = 1
line_height = 20
resize_dims = (300, 300)

print("Press ENTER to select an image, ESC to skip, Q to quit.")

cv2.namedWindow("SVHN Annotator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SVHN Annotator", 400, 600)
for idx, (img, label) in enumerate(zip(svhn.data, svhn.labels)):
    if label > 9:
        continue

    img_cv = np.transpose(img, (1, 2, 0))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, resize_dims, interpolation=cv2.INTER_NEAREST)

    h, w, _ = img_cv.shape
    panel_height = 250
    canvas = np.zeros((h + panel_height, w, 3), dtype=np.uint8)

    canvas[:h, :, :] = img_cv.copy()

    cv2.putText(canvas, f"Label: {label}", (5, 15), font, 0.4, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    y_pos = h + 30
    x_pos = 15
    for digit in range(10):
        count = class_counts[digit]
        status = "✓" if count >= target_per_class else ""
        text = f"{digit}: {count}{status}"
        cv2.putText(canvas, text, (x_pos, y_pos), font, font_scale, font_color, line_thickness, lineType=cv2.LINE_AA)
        y_pos += 20

    cv2.imshow("SVHN Annotator", canvas)

    key = cv2.waitKey(0)
    if key == 13:
        selected_indices.append(idx)
        class_counts[label] += 1
    elif key == 27:
        continue
    elif key == ord('q') or key == ord('Q'):
        print("Quitting early.")
        break

    if all(class_counts[d] >= target_per_class for d in range(10)):
        print("All classes selected (300 each). Exiting.")
        break

cv2.destroyAllWindows()

with open("selected_svhn_indices.txt", "w") as f:
    for idx in selected_indices:
        f.write(f"{idx}\n")

print(f"Annotation complete. {len(selected_indices)} images selected.")
