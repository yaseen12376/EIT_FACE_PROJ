import cv2
import numpy as np
import csv
import time
import os
from tkinter import Tk, filedialog
from insightface.app import FaceAnalysis

# Initialize InsightFace (includes RetinaFace detection and ArcFace embeddings)
print("Initializing InsightFace models...")
try:
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(960, 960), det_thresh=0.6)  # Optimized for faster processing
    print("InsightFace model initialized successfully")
except Exception as e:
    print(f"Error initializing InsightFace: {e}")
    face_app = None

# Set up time and attendance tracking
current_time = time.strftime("%Y-%m-%d %H-%M-%S")
detected_names = []
attendance_data = {}
# Set up time and attendance tracking (Modified to show time only as HH:MM:SS)
current_time1 = time.strftime("%H-%M-%S")  # Only hour, minute, and second


# Initialize the list to store face encodings
known_face_encodings = []  # Initialize it here to avoid the NameError

# Define how much time before resetting the system
#reset_interval = 3600
#start_time = time.time()

class Face:
    def __init__(self, name, rrn, branch, image):
        self.name = name
        self.rrn = rrn
        self.image = image
        self.branch = branch

    def display_face(self):
        return f"{self.name},{self.rrn},{self.image}"

    def face_upload(self):
        try:
            # Load the image using OpenCV
            img = cv2.imread(self.image)
            if img is None:
                print(f"Image not found: {self.image}")
                return False
            
            # Convert BGR to RGB for InsightFace
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use InsightFace for face detection and embedding
            if face_app is None:
                print("Face analysis app not initialized")
                return False
                
            faces = face_app.get(img_rgb)
            
            if faces and len(faces) > 0:
                # Get the first (most confident) face
                face = faces[0]
                face_embedding = face.embedding  # ArcFace embedding
                
                # Return face encoding and the person's details
                return face_embedding, self.name, self.rrn, self.branch
            else:
                print(f"No faces detected in {self.image}")
                return None
                
        except Exception as e:
            print(f"Error loading image {self.image}: {e}")
            return None

def add_face(name, rrn, branch, image_path):
    face = Face(name, rrn, branch, image_path)
    face_data = face.face_upload()
    if face_data:
        encoding, name, rrn, branch = face_data
        # Append the face encoding and details to the global list
        known_face_encodings.append((encoding, name, rrn, branch))
        print(f"Uploaded face for {name}")
    else:
        print(f"Failed to upload face for {name}")


# Initialize known students (add your student image files to the project directory)
# Note: Make sure the image files exist in the same directory as this script
add_face('yaseen', 1170, 'AI&DS', 'yaseen.jpg')
add_face('sajjad', 1170, 'IT', 'sajjad.jpg')  # This one works
# Add more students by uncommenting and modifying the lines below:
# add_face('naveed', 1152, 'AI&DS', 'naveed.jpg')
# add_face('hameed', 1145, 'AI&DS', 'hameed.jpg')
# add_face('vikinesh', 1146, 'AI&DS', 'viki.jpg')
# add_face('chatu', 2381, 'AI&DS', 'chatu.jpg')
# add_face('faaz', 4927, 'AI&DS', 'faaz.jpg')
# add_face('hasim', 3852, 'AI&DS', 'hasim.jpg')
# add_face('leo', 1743, 'AI&DS', 'leo.jpg')
# add_face('maida', 5612, 'AI&DS', 'maida.jpg')
# add_face('marofa', 8234, 'AI&DS', 'marofa.jpg')
# add_face('nizam', 6723, 'AI&DS', 'nizam.jpg')
# add_face('sabila', 3156, 'AI&DS', 'sabila.jpg')
# add_face('sandy', 7812, 'AI&DS', 'sandy.jpg')
# add_face('shabaz', 4590, 'AI&DS', 'shabaz.jpg')
# add_face('shameer69', 6901, 'AI&DS', 'shameer69.jpg')
# add_face('sheik_vili', 8420, 'AI&DS', 'sheik_vili.jpg')
# add_face('stefina', 1204, 'AI&DS', 'stefina.jpg')
# add_face('suthika', 2345, 'AI&DS', 'suthika.jpg')
# add_face('swathy', 5678, 'AI&DS', 'swathy.jpg')
# add_face('tawheed', 8765, 'AI&DS', 'tawheed.jpg')
# add_face('vanathi', 4321, 'AI&DS', 'vanathi.jpg')
# add_face('viswa', 3456, 'AI&DS', 'viswa.jpg')
# add_face('zarah', 9876, 'AI&DS', 'zarah.jpg')


# Face recognition configuration
class ImageConfig:
    ARCFACE_SIMILARITY_THRESHOLD = 0.4 # ArcFace cosine similarity threshold (lowered for better recognition)
    STANDARD_FACE_SIZE = (112, 112)  # ArcFace standard input size

def recognize_face_from_frame(frame):
    """Recognize faces in a single frame using InsightFace"""
    try:
        # Convert BGR to RGB for InsightFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use InsightFace for face detection and embedding
        if face_app is None:
            print("Face analysis app not initialized")
            return []
            
        faces = face_app.get(rgb_frame)
        
        results = []
        if faces:
            for face in faces:
                # Get face embedding
                face_embedding = face.embedding
                
                # Calculate cosine similarities with known faces
                similarities = []
                for encoding, _, _, _ in known_face_encodings:
                    # Cosine similarity for ArcFace embeddings
                    similarity = np.dot(face_embedding, encoding) / (
                        np.linalg.norm(face_embedding) * np.linalg.norm(encoding)
                    )
                    similarities.append(similarity)
                
                if similarities:
                    max_similarity_index = np.argmax(similarities)
                    max_similarity = similarities[max_similarity_index]
                    
                    # Check if similarity exceeds threshold
                    if max_similarity > ImageConfig.ARCFACE_SIMILARITY_THRESHOLD:
                        _, name, rrn, branch = known_face_encodings[max_similarity_index]
                        confidence = max_similarity
                        
                        results.append({
                            "name": name,
                            "rrn": rrn,
                            "branch": branch,
                            "confidence": float(confidence),
                            "bbox": face.bbox.astype(int).tolist(),
                            "similarity": float(max_similarity)
                        })
                    else:
                        results.append({
                            "name": "unknown",
                            "rrn": None,
                            "branch": None,
                            "confidence": 0,
                            "bbox": face.bbox.astype(int).tolist(),
                            "similarity": float(max_similarity)
                        })
        
        return results
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return []

# --- New function to upload and process a video ---
def upload_and_recognize_video():
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a video file", 
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("All files", "*.*")
        ]
    )
    if not file_path:
        print("No file selected.")
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error opening video file: {file_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n=== Video Information ===")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"=========================\n")
    
    # Check if we have any known faces loaded
    if not known_face_encodings:
        print("Warning: No known faces loaded! Please add student images first.")
        cap.release()
        return
    
    print(f"Loaded {len(known_face_encodings)} known face(s) for recognition")
    print("Starting video processing...")
    
    # Create output video writer
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, f"recognized_{base_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Dynamic frame skip based on video duration for optimal speed
    if duration <= 30:  # Short videos (≤30 seconds)
        frame_skip = fps // 4  # Process 4 frames per second
    elif duration <= 120:  # Medium videos (≤2 minutes)
        frame_skip = fps // 2  # Process 2 frames per second
    else:  # Long videos (>2 minutes)
        frame_skip = fps  # Process 1 frame every second

    frame_count = 0
    processed_frames = 0
    recognized_students = set()  # Keep track of recognized students
    recognition_count = {}  # Count how many times each student is recognized
    start_time = time.time()
    
    print(f"Optimized processing: analyzing 1 frame every {frame_skip/fps:.1f} second(s)")
    print(f"Estimated processing time: {(total_frames / frame_skip * 0.1):.1f} seconds")
    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame for recognition every frame_skip frames
        if frame_count % frame_skip == 0:
            processed_frames += 1
            
            # Recognize faces in current frame
            results = recognize_face_from_frame(frame)
            
            # Draw bounding boxes and labels
            for result in results:
                bbox = result["bbox"]
                name = result["name"]
                confidence = result["confidence"]
                similarity = result["similarity"]
                
                x1, y1, x2, y2 = bbox
                
                # Choose color and label based on recognition
                if name != "unknown":
                    color = (0, 255, 0)  # Green for recognized
                    recognized_students.add(name)
                    recognition_count[name] = recognition_count.get(name, 0) + 1
                    label = f"{name} ({similarity:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = f"Unknown ({similarity:.2f})"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Show progress every 10% or every 2 seconds of processing
        if frame_count % (fps * 2) == 0 or frame_count % (total_frames // 10) == 0:
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            if progress > 0:
                estimated_total = elapsed_time / (progress / 100)
                remaining_time = estimated_total - elapsed_time
                print(f"Progress: {progress:.1f}% | Elapsed: {elapsed_time:.1f}s | ETA: {remaining_time:.1f}s | Analyzed: {processed_frames} frames")
            else:
                print(f"Progress: {progress:.1f}% | Processed frames: {processed_frames}")
    
    # Release everything
    cap.release()
    out.release()
    
    total_processing_time = time.time() - start_time
    print(f"\n=== Processing Completed ===")
    print(f"Output video saved: {output_path}")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Video frames: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
    print(f"Analyzed frames: {processed_frames} (every {frame_skip/fps:.1f} seconds)")
    print(f"Processing speed: {frame_count/total_processing_time:.1f} fps")
    
    if recognized_students:
        print(f"\n=== Recognition Results ===")
        for student in recognized_students:
            count = recognition_count.get(student, 0)
            print(f"- {student}: recognized in {count} frame(s)")
        
        # Mark attendance for recognized students
        attendance_file = f"attendance_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(attendance_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'Recognition_Count', 'Time', 'Date', 'Video_File'])
            for student in recognized_students:
                count = recognition_count.get(student, 0)
                writer.writerow([student, count, current_time1, time.strftime("%Y-%m-%d"), os.path.basename(file_path)])
        print(f"\nAttendance saved: {attendance_file}")
    else:
        print("\nNo students were recognized in the video.")
    
    print("===========================")

# --- Uncomment below to use webcam as before ---
# video_capture = cv2.VideoCapture(0)
# while True:
#     ...existing code...
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()

# --- Call the new function to upload and process a video ---
if __name__ == "__main__":
    print("Face Recognition System - Video Processing")
    print("This system will process a video file and recognize faces.")
    print("Supported formats: MP4, AVI, MOV, MKV, WMV, FLV")
    print("\nSelect a video file to process...")
    upload_and_recognize_video()

