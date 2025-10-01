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
    face_app.prepare(ctx_id=0, det_size=(960, 960), det_thresh=0.4)  # Lower threshold for better detection
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

def add_face_from_folder(name, rrn, branch, folder_path=None):
    """Load all images from a person's folder automatically"""
    if folder_path is None:
        folder_path = f"images/{name}"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        print(f"💡 Please create folder and add 3-4 images of {name}")
        return False
    
    # Get all image files from the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, file))
    
    if not image_paths:
        print(f"⚠️  No images found in {folder_path}")
        print(f"💡 Please add 3-4 photos of {name} to this folder")
        return False
    
    print(f"📂 Loading {len(image_paths)} images for {name}...")
    
    encodings_for_person = []
    successful_uploads = 0
    
    for i, image_path in enumerate(image_paths):
        face = Face(name, rrn, branch, image_path)
        face_data = face.face_upload()
        if face_data:
            encoding, _, _, _ = face_data
            encodings_for_person.append(encoding)
            successful_uploads += 1
            print(f"  ✅ {os.path.basename(image_path)}")
        else:
            print(f"  ❌ {os.path.basename(image_path)} - failed to detect face")
    
    if encodings_for_person:
        # Store all encodings for this person
        known_face_encodings.append((encodings_for_person, name, rrn, branch))
        print(f"✅ Successfully loaded {successful_uploads}/{len(image_paths)} images for {name}")
        return True
    else:
        print(f"❌ Failed to load any faces for {name}")
        return False

def add_face(name, rrn, branch, image_paths):
    """Add a person with multiple reference images for better recognition (legacy method)"""
    if isinstance(image_paths, str):
        image_paths = [image_paths]  # Convert single path to list
    
    encodings_for_person = []
    successful_uploads = 0
    
    for i, image_path in enumerate(image_paths):
        face = Face(name, rrn, branch, image_path)
        face_data = face.face_upload()
        if face_data:
            encoding, _, _, _ = face_data
            encodings_for_person.append(encoding)
            successful_uploads += 1
        else:
            print(f"Failed to upload image {i+1} for {name}: {image_path}")
    
    if encodings_for_person:
        # Store all encodings for this person
        known_face_encodings.append((encodings_for_person, name, rrn, branch))
        print(f"Uploaded {successful_uploads} face encoding(s) for {name}")
    else:
        print(f"Failed to upload any faces for {name}")


# Initialize known students using organized folder structure
# 📂 Folder Structure: images/student_name/ (contains 3-4 photos)
print("\n📂 Loading student faces from organized folders...")
print("=" * 50)

# Load faces from folders (recommended method)
students_loaded = 0
total_students = 0

# Current students (only include those with organized image folders)
student_data = [
    ('yaseen', 1170, 'AI&DS'),
    ('sajjad', 1170, 'IT'),
    # Add more students here: ('name', rrn, 'branch')
    # ('naveed', 1152, 'AI&DS'),  # Uncomment when images folder is ready
    # ('hameed', 1145, 'AI&DS'),  # Uncomment when images folder is ready
]

for name, rrn, branch in student_data:
    total_students += 1
    if add_face_from_folder(name, rrn, branch):
        students_loaded += 1
    print()  # Empty line for readability

print(f"📊 Summary: {students_loaded}/{total_students} students loaded successfully")
print("=" * 50)

# 💡 To add a new student:
# 1. Create folder: images/student_name/
# 2. Add 3-4 photos to the folder
# 3. Add entry to student_data list above

# 🔄 Fallback: Load from individual files (if folders don't exist)
if students_loaded == 0:
    print("⚠️  No organized folders found. Using fallback method...")
    # Uncomment and modify these lines if you have individual image files:
    # add_face('yaseen', 1170, 'AI&DS', ['yaseen.jpg', 'yaseen_front.jpg'])
    # add_face('sajjad', 1170, 'IT', ['sajjad.jpg', 'sajjad_front.jpg'])
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
    ARCFACE_SIMILARITY_THRESHOLD = 0.3  # Aggressive threshold for better recognition
    STANDARD_FACE_SIZE = (112, 112)  # ArcFace standard input size

def recognize_face_from_frame(frame, debug=False):
    """Recognize faces in a single frame using InsightFace"""
    try:
        # Convert BGR to RGB for InsightFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use InsightFace for face detection and embedding
        if face_app is None:
            print("Face analysis app not initialized")
            return []
            
        faces = face_app.get(rgb_frame)
        
        if debug:
            print(f"\n[DEBUG] Detected {len(faces)} face(s) in frame")
        
        results = []
        if faces:
            for i, face in enumerate(faces):
                # Get face embedding
                face_embedding = face.embedding
                
                # Calculate cosine similarities with known faces
                similarities = []
                similarity_details = []
                for encodings_list, name, _, _ in known_face_encodings:
                    # Calculate similarity with all encodings for this person
                    person_similarities = []
                    for encoding in encodings_list:
                        similarity = np.dot(face_embedding, encoding) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(encoding)
                        )
                        person_similarities.append(similarity)
                    
                    # Use the maximum similarity from all their reference images
                    max_person_similarity = max(person_similarities)
                    similarities.append(max_person_similarity)
                    similarity_details.append((name, max_person_similarity))
                
                if similarities:
                    max_similarity_index = np.argmax(similarities)
                    max_similarity = similarities[max_similarity_index]
                    
                    if debug:
                        print(f"[DEBUG] Face {i+1} similarities:")
                        for name, sim in similarity_details:
                            print(f"  - {name}: {sim:.3f}")
                        print(f"[DEBUG] Best match: {similarity_details[max_similarity_index][0]} ({max_similarity:.3f})")
                        print(f"[DEBUG] Threshold: {ImageConfig.ARCFACE_SIMILARITY_THRESHOLD}")
                    
                    # Check if similarity exceeds threshold
                    if max_similarity > ImageConfig.ARCFACE_SIMILARITY_THRESHOLD:
                        _, name, rrn, branch = known_face_encodings[max_similarity_index]
                        confidence = max_similarity
                        
                        if debug:
                            print(f"[DEBUG] ✅ RECOGNIZED: {name}")
                        
                        results.append({
                            "name": name,
                            "rrn": rrn,
                            "branch": branch,
                            "confidence": float(confidence),
                            "bbox": face.bbox.astype(int).tolist(),
                            "similarity": float(max_similarity)
                        })
                    else:
                        if debug:
                            print(f"[DEBUG] ❌ UNKNOWN: Best similarity {max_similarity:.3f} below threshold {ImageConfig.ARCFACE_SIMILARITY_THRESHOLD}")
                        
                        results.append({
                            "name": "unknown",
                            "rrn": None,
                            "branch": None,
                            "confidence": 0,
                            "bbox": face.bbox.astype(int).tolist(),
                            "similarity": float(max_similarity)
                        })
        elif debug:
            print("[DEBUG] No faces detected in this frame")
        
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
        print("❌ No known faces loaded!")
        print("💡 Please organize your images:")
        print("   1. Create folders: images/student_name/")
        print("   2. Add 3-4 photos per student")
        print("   3. Run the system again")
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
            
            # Only show detection summary, not detailed debug info
            debug_mode = False  # Disabled verbose debugging
            
            # Recognize faces in current frame
            results = recognize_face_from_frame(frame, debug=debug_mode)
            
            # Simple detection summary (only when faces are found)
            if results:
                detected_names = [r['name'] for r in results if r['name'] != 'unknown']
                if detected_names:
                    print(f"✅ Frame {processed_frames}: Detected {', '.join(detected_names)}")
            
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
        
        # Show simple progress every 25%
        if frame_count % (total_frames // 4) == 0 and frame_count > 0:
            progress = (frame_count / total_frames) * 100
            elapsed_time = time.time() - start_time
            print(f"📊 {progress:.0f}% complete - {processed_frames} frames analyzed")
    
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

