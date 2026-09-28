from pathlib import Path

from ultralytics import YOLO
import cv2

MODEL_DIR = Path(__file__).resolve().parent

# -----------------------------
# Select Camera
# -----------------------------
while True:
    try:
        camera_index = int(input("Enter camera index (0 or 1): "))
        break
    except ValueError:
        print("Please enter a valid integer.\n")

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Could not open camera {camera_index}")
    exit()

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(str(MODEL_DIR / "YOLOv10n_gestures.pt"))

print("\nModel loaded successfully.")
print("Press ESC to exit.\n")

# -----------------------------
# Detection Loop
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False,
        # Keep Ultralytics' output folder in the module, not the launch folder
        project=str(MODEL_DIR.parent / "runs" / "detect"),
        name="predict",
        exist_ok=True
    )

    annotated = results[0].plot()

    if len(results[0].boxes) > 0:
        print("-" * 40)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            gesture = model.names[cls]

            print(f"Gesture : {gesture:20} Confidence : {conf:.3f}")

    cv2.imshow("YOLOv10 Gesture Recognition (.pt)", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:      # ESC
        break

cap.release()
cv2.destroyAllWindows()