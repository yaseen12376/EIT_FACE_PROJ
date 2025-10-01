import cv2
import numpy as np
import csv
import time
import os
from insightface.app import FaceAnalysis
from tkinter import Tk, messagebox

# Real-time optimized configuration for CCTV attendance
class CCTVAttendanceConfig:
    # CCTV Connection Settings
    RTSP_URL = "rtsp://admin:AK@MrA!4501$uf@192.168.0.109:554/cam/realmonitor?channel=8&subtype=0"
    ALTERNATIVE_URLS = []  # Add backup URLs if needed
    
    # Detection optimizations for CCTV processing
    DETECTION_SIZE = (960, 960)        # Balanced for CCTV quality
    DETECTION_THRESHOLD = 0.6          # Higher threshold for CCTV noise reduction
    
    # Recognition optimizations
    SIMILARITY_THRESHOLD = 0.28        # Slightly higher for CCTV conditions
    MAX_FACES_PER_FRAME = 4           # More faces for office CCTV
    
    # Performance optimizations for continuous CCTV processing
    FRAME_RESIZE_FACTOR = 0.8         # Resize for CCTV processing
    PROCESS_EVERY_N_FRAMES = 2        # Process every 2nd frame for CCTV
    
    # CCTV-specific settings
    RECOGNITION_COOLDOWN = 10.0       # Longer cooldown for CCTV (10 seconds)
    MIN_CONSECUTIVE_DETECTIONS = 2    # Fewer detections needed for CCTV
    CONNECTION_RETRY_INTERVAL = 5.0   # Retry connection every 5 seconds
    
    # Display settings for CCTV monitoring
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    SAVE_SCREENSHOTS = True           # Save attendance screenshots
    AUTO_RECONNECT = True             # Auto-reconnect on connection loss

print("🏢📹 Initializing CCTV-Based Office Attendance System...")

# Check GUI availability
GUI_AVAILABLE = True
try:
    import sys
    if sys.platform.startswith('win'):
        GUI_AVAILABLE = True
        print("✓ GUI enabled for Windows platform")
    else:
        try:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test") 
            GUI_AVAILABLE = True
        except:
            GUI_AVAILABLE = False
except Exception:
    GUI_AVAILABLE = True

print(f"GUI Status: {'Available' if GUI_AVAILABLE else 'Not Available'}")

# Initialize InsightFace for CCTV processing
try:
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, 
                    det_size=CCTVAttendanceConfig.DETECTION_SIZE,
                    det_thresh=CCTVAttendanceConfig.DETECTION_THRESHOLD)
    print("✓ Face recognition initialized for CCTV processing")
except Exception as e:
    print(f"❌ Error initializing face recognition: {e}")
    exit()

# Global variables for CCTV attendance tracking
known_face_encodings = []
last_recognition_time = {}      # Track when each person was last recognized
consecutive_detections = {}     # Track consecutive detections for confirmation
today_attendance = set()        # Track who attended today
screenshot_counter = 0          # Counter for screenshot naming

class Employee:
    def __init__(self, name, employee_id, department, image_path):
        self.name = name
        self.employee_id = employee_id
        self.department = department
        self.image_path = image_path

    def enroll_face(self):
        """Enroll face with optimized processing for CCTV system"""
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                print(f"❌ Image not found: {self.image_path}")
                return None
            
            # Resize for consistent enrollment processing
            img = cv2.resize(img, (640, 640))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            faces = face_app.get(img_rgb)
            if faces and len(faces) > 0:
                face = faces[0]
                return face.embedding, self.name, self.employee_id, self.department
            else:
                print(f"❌ No face detected in: {self.image_path}")
                return None
        except Exception as e:
            print(f"❌ Error enrolling {self.name}: {e}")
            return None

def add_employee(name, employee_id, department, image_path):
    """Add employee to the CCTV attendance system"""
    employee = Employee(name, employee_id, department, image_path)
    face_data = employee.enroll_face()
    if face_data:
        encoding, name, emp_id, dept = face_data
        known_face_encodings.append((encoding, name, emp_id, dept))
        print(f"✓ Enrolled: {name} (ID: {emp_id}, Dept: {dept})")
        return True
    else:
        print(f"✗ Failed to enroll: {name}")
        return False

def test_cctv_connection():
    """Test CCTV RTSP connection with fallback options"""
    print("🔌 Testing CCTV connection...")
    
    # Try main RTSP URL
    print(f"Trying main CCTV URL...")
    try:
        cap = cv2.VideoCapture(CCTVAttendanceConfig.RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for live stream
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("✓ CCTV connection successful!")
                # Get video properties
                try:
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if width <= 0 or height <= 0:
                        width, height = 1280, 720
                        
                    print(f"✓ CCTV Properties: {width}x{height} @ {fps} FPS")
                    return cap, (width, height, fps)
                except:
                    return cap, (1280, 720, 25)
            else:
                print("✗ CCTV opened but can't read frames")
                cap.release()
        else:
            print("✗ CCTV failed to open")
            cap.release()
    except Exception as e:
        print(f"✗ CCTV connection error: {e}")
    
    # Try alternative URLs if available
    for i, alt_url in enumerate(CCTVAttendanceConfig.ALTERNATIVE_URLS):
        print(f"Trying alternative CCTV {i+1}...")
        try:
            cap = cv2.VideoCapture(alt_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✓ Alternative CCTV {i+1} connected!")
                    return cap, (1280, 720, 25)
                else:
                    cap.release()
        except Exception as e:
            print(f"✗ Alternative CCTV {i+1} error: {e}")
    
    print("❌ All CCTV connections failed. Please check:")
    print("  1. Network connection")
    print("  2. CCTV IP address (192.168.0.109)")
    print("  3. Username/password (admin:AK@MrA!4501$uf)")
    print("  4. RTSP port (554)")
    print("  5. CCTV streaming settings")
    
    # Offer webcam fallback
    if GUI_AVAILABLE:
        try:
            root = Tk()
            root.withdraw()
            use_webcam = messagebox.askyesno("CCTV Failed", 
                                           "CCTV connection failed. Use webcam instead?")
            root.destroy()
            
            if use_webcam:
                print("📷 Falling back to webcam...")
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    print("✓ Webcam connected as fallback")
                    return cap, (640, 480, 30)
                else:
                    cap.release()
        except:
            pass
    
    return None, None

def recognize_faces_cctv(frame):
    """Optimized face recognition for CCTV processing"""
    current_time = time.time()
    
    # Resize frame for faster CCTV processing
    height, width = frame.shape[:2]
    new_width = int(width * CCTVAttendanceConfig.FRAME_RESIZE_FACTOR)
    new_height = int(height * CCTVAttendanceConfig.FRAME_RESIZE_FACTOR)
    small_frame = cv2.resize(frame, (new_width, new_height))
    
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    try:
        faces = face_app.get(rgb_frame)
        
        # Limit faces for CCTV performance
        if len(faces) > CCTVAttendanceConfig.MAX_FACES_PER_FRAME:
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:CCTVAttendanceConfig.MAX_FACES_PER_FRAME]
        
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
                scale_factor = 1.0 / CCTVAttendanceConfig.FRAME_RESIZE_FACTOR
                bbox = (bbox * scale_factor).astype(int)
                
                if max_similarity > CCTVAttendanceConfig.SIMILARITY_THRESHOLD:
                    _, name, emp_id, dept = known_face_encodings[max_idx]
                    
                    # Check cooldown period (longer for CCTV)
                    if name not in last_recognition_time or \
                       (current_time - last_recognition_time[name]) > CCTVAttendanceConfig.RECOGNITION_COOLDOWN:
                        
                        # Track consecutive detections for confirmation
                        consecutive_detections[name] = consecutive_detections.get(name, 0) + 1
                        
                        # Confirm recognition after consecutive detections
                        if consecutive_detections[name] >= CCTVAttendanceConfig.MIN_CONSECUTIVE_DETECTIONS:
                            if name not in today_attendance:
                                mark_attendance_cctv(name, emp_id, dept, frame, bbox)
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
        print(f"❌ Recognition error: {e}")
        return []

def mark_attendance_cctv(name, emp_id, dept, frame, bbox):
    """Mark attendance with CCTV screenshot and timestamp"""
    global screenshot_counter
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    date = time.strftime("%Y-%m-%d")
    time_only = time.strftime("%H:%M:%S")
    
    # Create attendance directory if it doesn't exist
    attendance_dir = "attendance_records"
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
    
    # Save screenshot if enabled
    screenshot_path = None
    if CCTVAttendanceConfig.SAVE_SCREENSHOTS:
        try:
            # Create screenshot with highlighted face
            screenshot_frame = frame.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(screenshot_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(screenshot_frame, f"{name} - ATTENDANCE MARKED", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(screenshot_frame, timestamp, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            screenshot_counter += 1
            screenshot_filename = f"attendance_{name}_{date}_{time_only.replace(':', '-')}_{screenshot_counter}.jpg"
            screenshot_path = os.path.join(attendance_dir, screenshot_filename)
            cv2.imwrite(screenshot_path, screenshot_frame)
            print(f"📸 Screenshot saved: {screenshot_filename}")
        except Exception as e:
            print(f"❌ Failed to save screenshot: {e}")
    
    # Save to CSV
    attendance_file = os.path.join(attendance_dir, f"daily_attendance_{date}.csv")
    file_exists = os.path.exists(attendance_file)
    
    with open(attendance_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Name', 'Employee_ID', 'Department', 'Date', 'Time', 
                           'Full_Timestamp', 'Screenshot_Path', 'Source'])
        
        writer.writerow([name, emp_id, dept, date, time_only, timestamp, 
                        screenshot_path or 'N/A', 'CCTV'])
    
    print(f"✅ CCTV ATTENDANCE MARKED: {name} (ID: {emp_id}) at {time_only}")

def display_cctv_statistics(frame):
    """Display real-time CCTV statistics on frame"""
    # Background overlay for better readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Display attendance count
    cv2.putText(frame, f"Today's Attendance: {len(today_attendance)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display enrolled employees
    cv2.putText(frame, f"Enrolled Employees: {len(known_face_encodings)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display current time
    current_time = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display CCTV status
    cv2.putText(frame, "CCTV LIVE", 
                (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def cctv_attendance_system():
    """Main CCTV-based attendance system"""
    
    print("📚 Loading employee database...")
    
    # Load employees - CUSTOMIZABLE SECTION
    employees_loaded = 0
    employees_loaded += add_employee('yaseen', '1170', 'AI&DS', 'yaseen.jpg')
    employees_loaded += add_employee('sajj', '9876', 'AI&DS', 'sajj.jpg')
    
    # Add more employees here as needed:
    # employees_loaded += add_employee('employee_name', 'employee_id', 'department', 'image.jpg')
    
    if not known_face_encodings:
        print("❌ No employees enrolled! Add employee images first.")
        return
    
    print(f"✅ System ready with {len(known_face_encodings)} enrolled employee(s)")
    
    # Test CCTV connection
    cap, video_props = test_cctv_connection()
    if cap is None:
        print("❌ Cannot establish video connection. Exiting...")
        return
    
    width, height, fps = video_props
    print(f"✅ Video source connected: {width}x{height} @ {fps} FPS")
    
    print("\n🎥 Starting CCTV attendance monitoring...")
    print("📋 Instructions:")
    if GUI_AVAILABLE:
        print("   - Press 'q' to quit")
        print("   - Press 's' to show today's attendance")
        print("   - Press 'r' to reset today's attendance")
        print("   - Press 'h' for help")
        print("   - Press 'c' to capture manual screenshot")
    else:
        print("   - Press Ctrl+C to stop")
    
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0
    last_connection_attempt = 0
    
    print("🟢 CCTV attendance system is now LIVE and monitoring...")
    
    try:
        while True:
            ret, frame = cap.read()
            
            # Handle connection loss
            if not ret or frame is None:
                current_time = time.time()
                if CCTVAttendanceConfig.AUTO_RECONNECT and \
                   (current_time - last_connection_attempt) > CCTVAttendanceConfig.CONNECTION_RETRY_INTERVAL:
                    
                    print("📡 Connection lost. Attempting to reconnect...")
                    cap.release()
                    cap, video_props = test_cctv_connection()
                    last_connection_attempt = current_time
                    
                    if cap is None:
                        print("❌ Reconnection failed. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        print("✅ Reconnected successfully!")
                        continue
                else:
                    continue
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
            
            # Resize frame to standard size for consistent processing
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Process every N frames for performance
            if frame_count % CCTVAttendanceConfig.PROCESS_EVERY_N_FRAMES == 0:
                results = recognize_faces_cctv(frame)
                
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
            display_cctv_statistics(frame)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display system title
            cv2.putText(frame, "CCTV OFFICE ATTENDANCE SYSTEM", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display frame if GUI available
            if GUI_AVAILABLE:
                try:
                    cv2.imshow('CCTV Office Attendance System', frame)
                    
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
                    elif key == ord('c'):
                        # Manual screenshot capture
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        screenshot_name = f"manual_capture_{timestamp}.jpg"
                        cv2.imwrite(screenshot_name, frame)
                        print(f"📸 Manual screenshot saved: {screenshot_name}")
                    elif key == ord('h'):
                        print("\n📋 HELP - Keyboard Controls:")
                        print("  'q' - Quit the system")
                        print("  's' - Show today's attendance summary")
                        print("  'r' - Reset today's attendance")
                        print("  'c' - Capture manual screenshot")
                        print("  'h' - Show this help message")
                        print()
                except cv2.error as e:
                    print(f"❌ Display error: {e}")
            else:
                # Non-GUI mode
                time.sleep(0.01)
            
            # Show streaming info every 30 seconds
            if frame_count % (fps * 30) == 0:
                elapsed = time.time() - fps_start_time
                print(f"📡 CCTV Monitoring: {frame_count} frames processed - FPS: {current_fps:.1f}")
                print(f"📊 Current Status: {len(today_attendance)} attendance(s) marked today")
                
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        if cap:
            cap.release()
        if GUI_AVAILABLE:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        # Final summary
        print(f"\n📊 FINAL CCTV SESSION SUMMARY:")
        print(f"Total employees enrolled: {len(known_face_encodings)}")
        print(f"Employees present today: {len(today_attendance)}")
        print(f"Total frames processed: {frame_count}")
        if today_attendance:
            print("📋 Today's Attendees:")
            for name in sorted(today_attendance):
                for _, emp_name, emp_id, dept in known_face_encodings:
                    if emp_name == name:
                        print(f"  ✓ {name} (ID: {emp_id}, Dept: {dept})")
                        break
        print("✅ CCTV attendance system stopped")

if __name__ == "__main__":
    cctv_attendance_system()