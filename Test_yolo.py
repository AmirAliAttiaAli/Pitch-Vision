import cv2
from ultralytics import YOLO

MODEL_PATH = "Model/football_detector_best.pt"
VIDEO_PATH = "Test Data/test_video.mp4"
OUTPUT_PATH = "Test Data/yolo_test.mp4"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.10,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    # Print detections
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model.names[cls_id]

            print(
                f"Frame {frame_count}: "
                f"{class_name} | conf={conf:.2f}"
            )

    writer.write(annotated)

    frame_count += 1

cap.release()
writer.release()

print("Done.")
print(f"Output: {OUTPUT_PATH}")