import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import filedialog, messagebox
import tkinter as tk


# Load the trained model
model = load_model(r'E:\code\Intelligence system\Finals\food_recog\updated_food_recognition_model.h5')

# Class labels corresponding to your dataset
class_labels = ['Dessert', 'Drink','Meal']

def preprocess_image(image_path):
    """Preprocess image for prediction."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0), image  # Add batch dimension


def detect_objects_in_image(file_path):
    """Detect objects in an image."""
    input_image, display_image = preprocess_image(file_path)
    prediction = model.predict(input_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    cv2.putText(
        display_image,
        f"{predicted_class}: {confidence*100:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.imshow("Detection Result", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_objects_in_video(file_path):
    """Detect objects in a video."""
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        resized_frame = cv2.resize(frame, (224, 224))
        input_frame = np.expand_dims(resized_frame / 255.0, axis=0)

        # Make prediction
        prediction = model.predict(input_frame)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Overlay detection result
        cv2.putText(
            frame,
            f"{predicted_class}: {confidence*100:.2f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Show the video frame
        cv2.imshow("Detection in Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")
        self.root.geometry("800x600")

        # Create navigation panel
        self.nav_frame = tk.Frame(self.root, bg="#2c3e50", width=200)
        self.nav_frame.pack(side="left", fill="y")

        # Create content area
        self.content_frame = tk.Frame(self.root, bg="#ecf0f1")
        self.content_frame.pack(side="right", fill="both", expand=True)

        # Navigation Buttons
        self.home_btn = tk.Button(self.nav_frame, text="Home", bg="#34495e", fg="white", font=("Arial", 12), command=self.home_view)
        self.detect_btn = tk.Button(self.nav_frame, text="Start Detection", bg="#34495e", fg="white", font=("Arial", 12), command=self.detection_view)
        self.settings_btn = tk.Button(self.nav_frame, text="Settings", bg="#34495e", fg="white", font=("Arial", 12), command=self.settings_view)
        self.exit_btn = tk.Button(self.nav_frame, text="Exit", bg="#e74c3c", fg="white", font=("Arial", 12), command=self.root.quit)

        # Pack Navigation Buttons
        self.home_btn.pack(pady=10, padx=10, fill="x")
        self.detect_btn.pack(pady=10, padx=10, fill="x")
        self.settings_btn.pack(pady=10, padx=10, fill="x")
        self.exit_btn.pack(pady=10, padx=10, fill="x")

        # Default View
        self.home_view()

    def clear_content_frame(self):
        """Clear all widgets from the content frame."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def home_view(self):
        """Display the Home View."""
        self.clear_content_frame()
        label = tk.Label(self.content_frame, text="Welcome to the Object Detection App", font=("Arial", 18), bg="#ecf0f1")
        label.pack(pady=20)

        description = """This application is designed to help you detect and recognize objects in real time using 
state-of-the-art computer vision technologies. You can start detecting objects by clicking 'Start Detection', 
customize app settings under 'Settings', or exit the application using the 'Exit' button."""
        description_label = tk.Label(self.content_frame, text=description, font=("Arial", 12), bg="#ecf0f1", wraplength=600, justify="left")
        description_label.pack(pady=10)

    def detection_view(self):
        """Display the Detection View."""
        self.clear_content_frame()
        label = tk.Label(self.content_frame, text="Start Detection", font=("Arial", 16), bg="#ecf0f1")
        label.pack(pady=20)

        # Buttons for Image or Video Selection
        image_btn = tk.Button(self.content_frame, text="Detect in Image", font=("Arial", 12), bg="#27ae60", fg="white", command=self.detect_in_image)
        video_btn = tk.Button(self.content_frame, text="Detect in Video", font=("Arial", 12), bg="#3498db", fg="white", command=self.detect_in_video)
        image_btn.pack(pady=10)
        video_btn.pack(pady=10)

    def settings_view(self):
        """Display the Settings View."""
        self.clear_content_frame()
        label = tk.Label(self.content_frame, text="Settings", font=("Arial", 16), bg="#ecf0f1")
        label.pack(pady=20)

    def detect_in_image(self):
        """Handle image detection."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if file_path:
            detect_objects_in_image(file_path)

    def detect_in_video(self):
        """Handle video detection."""
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            detect_objects_in_video(file_path)


# Main app entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
