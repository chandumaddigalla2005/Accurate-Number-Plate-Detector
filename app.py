import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

# Load model
model = YOLO("model/best.pt")


def draw_and_collect(frame, results, frame_idx=0):
    annotated = frame.copy()
    rows = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy
        confs = results[0].boxes.conf

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(confs[i]) if confs is not None else 0.0

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Plate | {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Only keep minimal info
            rows.append({
                "frame": frame_idx,
                "confidence": round(conf, 4)
            })

    return annotated, rows


def process_image(image):
    frame = np.array(image)
    results = model(frame)

    annotated, rows = draw_and_collect(frame, results, frame_idx=0)
    df = pd.DataFrame(rows)

    return annotated, df


def process_video(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, None

    out_path = os.path.join(tempfile.gettempdir(), "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 20.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    all_rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated, rows = draw_and_collect(frame, results, frame_idx)

        out.write(annotated)
        all_rows.extend(rows)
        frame_idx += 1

    cap.release()
    out.release()

    df = pd.DataFrame(all_rows)

    return out_path, df


with gr.Blocks(title="Number Plate Detection") as demo:

    gr.Markdown("## 🚗 Number Plate Detection System")
    gr.Markdown("Upload an image or video to detect number plates.")

    with gr.Tab("Image"):
        inp_img = gr.Image(type="pil", label="Upload Image / Use Camera")
        btn_img = gr.Button("Run Detection")
        out_img = gr.Image(label="Annotated Output")
        table_img = gr.Dataframe(label="Detections")

        btn_img.click(
            fn=process_image,
            inputs=inp_img,
            outputs=[out_img, table_img]
        )

    with gr.Tab("Video"):
        inp_vid = gr.Video(label="Upload Video")
        btn_vid = gr.Button("Run Detection")
        out_vid = gr.Video(label="Annotated Video")
        table_vid = gr.Dataframe(label="Detections")

        btn_vid.click(
            fn=process_video,
            inputs=inp_vid,
            outputs=[out_vid, table_vid]
        )

demo.launch()