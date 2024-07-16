import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pytesseract
import re


pytesseract.pytesseract.tesseract_cmd = os.path.join('data', 'tesseract', "tesseract.exe"
)


def deteksi_plat(sumber_video):
    cap = cv2.VideoCapture(int(sumber_video) if sumber_video.isdigit() else sumber_video)
    if not cap.isOpened():
        messagebox.showerror('','video tidak bisa di buka')
        return

    polylines = np.array([], dtype=np.int32).reshape(-1, 2)
    drawing = False
    modal_active = False

    root.withdraw()


    plate = []
    model = YOLO(os.path.join("data", "model", "best.pt"))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        results = model(frame)

        bounding_box = []
        if not drawing:
            for result in results:
                for box in result.boxes:
                    cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                    cx, cy, w, h = int(cx), int(cy), int(w), int(h)
                    confidence = box.conf.cpu().item()
                    class_id = box.cls.cpu().item()
                    class_name = model.names[class_id]

                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    if class_name == "numberplate":
                        bounding_box = [cx, cy, x1, y1, x2, y2]

        x1, y1 = 100, 100
        x2, y2 = 620, 380

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 255), thickness=2)

        if bounding_box:
            cx, cy, x1, y1, x2, y2 = bounding_box
            target = cv2.pointPolygonTest(pts, (cx, cy), False)
            if target >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                roi = frame[y1:y2, x1:x2]
                text = pytesseract.image_to_string(roi, config="--psm 6").strip()

                text = re.sub(r"[^A-Z0-9]", "", text).upper()

                if text:
                    print("\tTerdeteksi")
                    plate.append(text)
                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )


        cv2.imshow('Deteksi plat', frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()

root = tk.Tk()
root.title("Deteksi plat")


label_sumber_video = tk.Label(root, text="Sumber vidoe:")
label_sumber_video.grid(column=0, row=0, padx=10, pady=10)

entry_sumber_video = tk.Entry(root)
entry_sumber_video.grid(column=1, row=0, padx=10, pady=10)


button_mulai = tk.Button(root, text="Mulai", command=lambda: deteksi_plat(entry_sumber_video.get()))
button_mulai.grid(column=0, row=2, columnspan=2, padx=10, pady=10)


root.mainloop()