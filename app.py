import os
import tempfile
import cv2
import gdown
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# -------------------- Page --------------------
st.set_page_config(page_title="Number Plate Detection", layout="wide")
st.title("🚗 Number Plate Detection")

# -------------------- Download Model --------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    FILE_ID = "1iicpPO9D7AvBzRxZUjnvKj2f9USQ9uFe"

    with st.spinner("Downloading model..."):
        gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

model = YOLO(MODEL_PATH)

# -------------------- Detection --------------------
def detect(frame):

    results = model.predict(frame, verbose=False)

    rows = []

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0,255,0),2)

            cv2.putText(frame,
                        f"Plate {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,255),
                        2)

            rows.append({
                "confidence": round(float(conf),4)
            })

    return frame, pd.DataFrame(rows)

# -------------------- Sidebar --------------------
option = st.sidebar.selectbox(
    "Choose Input",
    ["Image","Video"]
)

# -------------------- IMAGE --------------------
if option=="Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        file_bytes = np.asarray(
            bytearray(uploaded.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes,1)

        if st.button("Run Detection"):

            out, df = detect(image)

            st.image(
                cv2.cvtColor(out,cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            st.dataframe(df)

# -------------------- VIDEO --------------------
else:

    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4","avi","mov"]
    )

    if uploaded:

        temp_input = tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
        temp_input.write(uploaded.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            fps = 20

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "output.mp4"

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width,height)
        )

        data=[]

        progress = st.progress(0)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count=0

        while True:

            ret,frame = cap.read()

            if not ret:
                break

            frame,df = detect(frame)

            out.write(frame)

            data.extend(df.to_dict("records"))

            count+=1

            progress.progress(min(count/total,1.0))

        cap.release()
        out.release()

        st.success("Completed")

        st.video(output_path)

        st.dataframe(pd.DataFrame(data))
