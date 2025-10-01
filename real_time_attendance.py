import cv2
import numpy as np
import csv
import time
import os
from insightface.app import FaceAnalysis

# Real-time optimized configuration for office attendance
class RealTimeConfig:
    # Detection optimizations for speed
    DETECTION_SIZE = (640, 640)        # Smaller for real-time performance
    DETECTION_THRESHOLD = 0.6          # Higher threshold for faster detection
    
    # Recognition optimizations
    SIMILARITY_THRESHOLD = 0.25        # Balanced threshold for office environment
    MAX_FACES_PER_FRAME = 3           # Limit for typical office scenarios
    
    # Performance optimizations
    FRAME_RESIZE_FACTOR = 0.6         # Resize input for speed
    PROCESS_EVERY_N_FRAMES = 2        # Skip frames for performance
    
    # Real-time specific settings
    RECOGNITION_COOLDOWN = 5.0        # Seconds before re-recognizing same person
    MIN_CONSECUTIVE_DETECTIONS = 3   # Confirm recognition over multiple frames
    
    # Display settings
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

print("🏢 Initializing Real-Time Office Attendance System...")

# Initialize with optimized settings for real-time performance
try:
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, 
                    det_size=RealTimeConfig.DETECTION_SIZE,
                    det_thresh=RealTimeConfig.DETECTION_THRESHOLD)
    print("✓ Face recognition initialized for real-time processing")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Global variables for real-time tracking
known_face_encodings = []
last_recognition_time = {}      # Track when each person was last recognized
consecutive_detections = {}     # Track consecutive detections for confirmation
today_attendance = set()        # Track who attended today

class Employee:
    def __init__(self, name, employee_id, department, image_path):
        self.name = name
        self.employee_id = employee_id
        self.department = department
        self.image_path = image_path

    def enroll_face(self):
        """Enroll face with optimized processing for speed"""
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                return None
            
            # Resize for faster enrollment processing
            img = cv2.resize(img, (640, 640))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            faces = face_app.get(img_rgb)
            if faces and len(faces) > 0:
                face = faces[0]
                return face.embedding, self.name, self.employee_id, self.department
            return None
        except Exception as e:
            print(f"Error enrolling {self.name}: {e}")
            return None

def add_employee(name, employee_id, department, image_path):
    """Add employee to the system"""
    employee = Employee(name, employee_id, department, image_path)
    face_data = employee.enroll_face()
    if face_data:
        encoding, name, emp_id, dept = face_data
        known_face_encodings.append((encoding, name, emp_id, dept))
        print(f"✓ Enrolled: {name} (ID: {emp_id})")
        return True
    else:
        print(f"✗ Failed to enroll: {name}")
        return False

def recognize_faces_realtime(frame):
    """Optimized face recognition for real-time processing"""
    current_time = time.time()
    
    # Resize frame for faster processing
    height, width = frame.shape[:2]
    new_width = int(width * RealTimeConfig.FRAME_RESIZE_FACTOR)
    new_height = int(height * RealTimeConfig.FRAME_RESIZE_FACTOR)
    small_frame = cv2.resize(frame, (new_width, new_height))
    
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    try:
        faces = face_app.get(rgb_frame)
        
        # Limit faces for real-time performance
        if len(faces) > RealTimeConfig.MAX_FACES_PER_FRAME:
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:RealTimeConfig.MAX_FACES_PER_FRAME]
        
        results = []
        if faces and known_face_encodings:
            # Pre-compute known encodings for speed
            known_encodings = np.array([enc for enc, _, _, _ in known_face_encodings], dtype=np.float32)
            known_norms = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)
            
            for face in faces:
                face_embedding = face.embedding.astype(np.float32)
                face_norm = face_embedding / np.linalg.norm(face_embedding)
                
                # Fast vectorized similarity computation
                similarities = np.dot(known_norms, face_norm)
                max_idx = np.argmax(similarities)
                max_similarity = similarities[max_idx]
                
                # Scale bbox back to original frame size
                bbox = face.bbox.astype(int)
                scale_factor = 1.0 / RealTimeConfig.FRAME_RESIZE_FACTOR
                bbox = (bbox * scale_factor).astype(int)
                
                if max_similarity > RealTimeConfig.SIMILARITY_THRESHOLD:
                    _, name, emp_id, dept = known_face_encodings[max_idx]
                    
                    # Check cooldown period
                    if name not in last_recognition_time or \
                       (current_time - last_recognition_time[name]) > RealTimeConfig.RECOGNITION_COOLDOWN:
                        
                        # Track consecutive detections for confirmation
                        consecutive_detections[name] = consecutive_detections.get(name, 0) + 1
                        
                        # Confirm recognition after consecutive detections
                        if consecutive_detections[name] >= RealTimeConfig.MIN_CONSECUTIVE_DETECTIONS:
                            if name not in today_attendance:
                                mark_attendance(name, emp_id, dept)
                                today_attendance.add(name)
                                last_recognition_time[name] = current_time
                                consecutive_detections[name] = 0
                                status = "marked"
                            else:
                                status = "already_marked"
                        else:
                            status = "confirming"
                    else:
                        status = "cooldown"
                    
                    results.append({
                        "name": name,
                        "employee_id": emp_id,
                        "department": dept,
                        "confidence": float(max_similarity),
                        "bbox": bbox.tolist(),
                        "status": status
                    })
                else:
                    results.append({
                        "name": "unknown",
                        "confidence": 0,
                        "bbox": bbox.tolist(),
                        "status": "unknown"
                    })
        
        return results
    except Exception as e:
        print(f"Recognition error: {e}")
        return []

def mark_attendance(name, emp_id, dept):
    """Mark attendance with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    date = time.strftime("%Y-%m-%d")
    time_only = time.strftime("%H:%M:%S")
    
    attendance_file = f"daily_attendance_{date}.csv"
    
    # Check if file exists, create with headers if not
    file_exists = os.path.exists(attendance_file)
    
    with open(attendance_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Name', 'Employee_ID', 'Department', 'Date', 'Time', 'Full_Timestamp'])
        
        writer.writerow([name, emp_id, dept, date, time_only, timestamp])
    
    print(f"✅ ATTENDANCE MARKED: {name} (ID: {emp_id}) at {time_only}")

def display_statistics(frame):
    """Display real-time statistics on frame"""
    # Display attendance count
    cv2.putText(frame, f"Today's Attendance: {len(today_attendance)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display enrolled employees
    cv2.putText(frame, f"Enrolled Employees: {len(known_face_encodings)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display current time
    current_time = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def real_time_attendance_system():
    """Main real-time attendance system"""
    
    print("📚 Loading employee database...")
    # Add your employees here
    add_employee('yaseen', '1170', 'AI&DS', 'yaseen.jpg')
    add_employee('swathy', '5678', 'AI&DS', 'swathy.jpg')
    add_employee('sajj', '9876', 'AI&DS', 'sajj.jpg')
    # Add more employees as needed
    
    if not known_face_encodings:
        print("❌ No employees enrolled! Add employee images first.")
        return
    
    print(f"✅ System ready with {len(known_face_encodings)} enrolled employee(s)")
    print("\n🎥 Starting real-time attendance system...")
    print("📋 Instructions:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to show today's attendance")
    print("   - Press 'r' to reset today's attendance")
    print("   - Press 'h' for help")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Optimize camera settings for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RealTimeConfig.DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RealTimeConfig.DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    print("🟢 System is now LIVE and monitoring...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start_time
            fps = 30 / elapsed if elapsed > 0 else 0
            fps_start_time = time.time()
        
        # Process every N frames for performance
        if frame_count % RealTimeConfig.PROCESS_EVERY_N_FRAMES == 0:
            results = recognize_faces_realtime(frame)
            
            # Draw results on frame
            for result in results:
                bbox = result["bbox"]
                name = result["name"]
                confidence = result["confidence"]
                status = result.get("status", "unknown")
                
                x1, y1, x2, y2 = bbox
                
                # Color coding based on status
                if status == "marked":
                    color = (0, 255, 0)  # Green - attendance marked
                    label = f"{name} ✅ MARKED"
                elif status == "already_marked":
                    color = (0, 255, 255)  # Yellow - already attended
                    label = f"{name} ✓ Present"
                elif status == "confirming":
                    color = (255, 255, 0)  # Cyan - confirming identity
                    label = f"{name} ? Confirming..."
                elif status == "cooldown":
                    color = (128, 128, 255)  # Light purple - in cooldown
                    label = f"{name} (Recent)"
                else:
                    color = (0, 0, 255)  # Red - unknown person
                    label = "Unknown Person"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                             (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display statistics and FPS
        display_statistics(frame)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display system title
        cv2.putText(frame, "REAL-TIME OFFICE ATTENDANCE SYSTEM", 
                   (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Real-Time Office Attendance', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n📊 TODAY'S ATTENDANCE SUMMARY:")
            print(f"Date: {time.strftime('%Y-%m-%d')}")
            print(f"Total Present: {len(today_attendance)}")
            if today_attendance:
                for name in sorted(today_attendance):
                    # Find employee details
                    for _, emp_name, emp_id, dept in known_face_encodings:
                        if emp_name == name:
                            print(f"  ✓ {name} (ID: {emp_id}, Dept: {dept})")
                            break
            else:
                print("  No one has attended yet today.")
            print()
        elif key == ord('r'):
            today_attendance.clear()
            last_recognition_time.clear()
            consecutive_detections.clear()
            print("🔄 Today's attendance has been reset!")
        elif key == ord('h'):
            print("\n📋 HELP - Keyboard Controls:")
            print("  'q' - Quit the system")
            print("  's' - Show today's attendance summary")
            print("  'r' - Reset today's attendance")
            print("  'h' - Show this help message")
            print()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print(f"\n📊 FINAL SESSION SUMMARY:")
    print(f"Total employees enrolled: {len(known_face_encodings)}")
    print(f"Employees present today: {len(today_attendance)}")
    print("✅ Real-time attendance system stopped")

if __name__ == "__main__":
    real_time_attendance_system()