import cv2
import numpy as np
import csv
import time
import os
import sys
import urllib.request
from insightface.app import FaceAnalysis
from tkinter import Tk, messagebox

# Import optional libraries for different models
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available. Balanced model (SCRFD) and RetinaFace will not be available.")

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("⚠️ RetinaFace not available. Install with: pip install retina-face")

# Real-time optimized configuration for CCTV attendance
class CCTVAttendanceConfig:
    # MODEL SELECTION - Will be set by user choice during runtime
    # Available options: "insightface", "balanced", "retinaface"
    MODEL_TYPE = None  # Will be set by user selection
    
    # CCTV Connection Settings
    RTSP_URL = "rtsp://admin:AK@MrA!4501$uf@192.168.0.109:554/cam/realmonitor?channel=8&subtype=0"
    ALTERNATIVE_URLS = []  # Add backup URLs if needed
    
    # Detection optimizations for CCTV processing (auto-adjusted per model)
    DETECTION_SIZE = (960, 960)        # Balanced for CCTV quality
    DETECTION_THRESHOLD = 0.3          # Lower threshold for better face detection
    
    # Recognition optimizations (auto-adjusted per model)
    SIMILARITY_THRESHOLD = 0.1        # Very low threshold for testing
    MAX_FACES_PER_FRAME = 4           # More faces for office CCTV
    
    # Model-specific optimizations
    MODEL_CONFIGS = {
        "balanced": {
            "detection_confidence": 0.6,  # Balanced confidence for SCRFD
            "similarity_threshold": 0.3,  # Balanced threshold for accuracy
            "process_every_n_frames": 2,  # Good balance of speed and accuracy
            "frame_resize_factor": 0.75,  # Balanced resolution
            "max_faces_per_frame": 3,     # Moderate face limit
            "detection_size": (640, 640)  # Optimized size for SCRFD
        },
        "retinaface": {
            "detection_confidence": 0.7,  # High confidence for RetinaFace
            "similarity_threshold": 0.2,  # Good threshold for accuracy
            "process_every_n_frames": 2,  # Good balance of speed and accuracy
            "frame_resize_factor": 0.8,   # Higher resolution for better accuracy
            "max_faces_per_frame": 4,     # Allow more faces for office CCTV
            "detection_size": (640, 640)  # Standard size for RetinaFace
        },
        "insightface": {
            "detection_size": (960, 960),
            "detection_threshold": 0.3,
            "similarity_threshold": 0.1,  # Much lower for testing
            "process_every_n_frames": 2,
            "frame_resize_factor": 0.8
        }
    }
    
    # Performance optimizations for continuous CCTV processing
    FRAME_RESIZE_FACTOR = 0.8         # Resize for CCTV processing
    PROCESS_EVERY_N_FRAMES = 2        # Process every 2nd frame for CCTV
    
    # CCTV-specific settings
    RECOGNITION_COOLDOWN = 5.0       # Longer cooldown for CCTV (10 seconds)
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

class SCRFDDetector:
    """SCRFD (Sample and Computation Redistributed Face Detection) for balanced performance"""
    
    def __init__(self):
        self.session = None
        self.input_size = (640, 640)
        self.confidence_threshold = 0.6
        self.nms_threshold = 0.4
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize SCRFD ONNX model"""
        try:
            # Try to download and load SCRFD model
            model_path = self._download_scrfd_model()
            if model_path:
                providers = ['CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                print("✓ SCRFD model loaded successfully")
            else:
                raise Exception("SCRFD model download failed")
        except Exception as e:
            print(f"❌ SCRFD initialization error: {e}")
            raise e
    
    def _download_scrfd_model(self):
        """Download SCRFD model if not available"""
        import urllib.request
        import os
        
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, "scrfd_2.5g.onnx")
        
        if not os.path.exists(model_path):
            print("📥 Downloading SCRFD model (this may take a few minutes)...")
            try:
                # SCRFD 2.5G model URL (balanced version)
                model_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g.onnx"
                urllib.request.urlretrieve(model_url, model_path)
                print(f"✓ SCRFD model downloaded to {model_path}")
            except Exception as e:
                print(f"❌ Failed to download SCRFD model: {e}")
                return None
        
        return model_path
    
    def detect_faces(self, image):
        """Detect faces using SCRFD"""
        if self.session is None:
            return []
        
        try:
            # Preprocess image
            original_height, original_width = image.shape[:2]
            
            # Resize to model input size
            resized = cv2.resize(image, self.input_size)
            
            # Normalize and prepare input
            input_blob = cv2.dnn.blobFromImage(resized, 1.0/128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_blob})
            
            # Parse outputs and get detections
            detections = self._parse_outputs(outputs, original_width, original_height)
            
            return detections
            
        except Exception as e:
            print(f"❌ SCRFD detection error: {e}")
            return []
    
    def _parse_outputs(self, outputs, original_width, original_height):
        """Parse SCRFD model outputs"""
        detections = []
        
        try:
            # SCRFD typically outputs [scores, bboxes, landmarks]
            if len(outputs) >= 2:
                scores = outputs[0]
                bboxes = outputs[1]
                
                # Scale factors
                scale_x = original_width / self.input_size[0]
                scale_y = original_height / self.input_size[1]
                
                for i in range(scores.shape[1]):
                    confidence = float(scores[0, i, 1])  # Class 1 is face
                    
                    if confidence > self.confidence_threshold:
                        # Extract and scale bbox
                        x1 = int(bboxes[0, i, 0] * scale_x)
                        y1 = int(bboxes[0, i, 1] * scale_y)
                        x2 = int(bboxes[0, i, 2] * scale_x)
                        y2 = int(bboxes[0, i, 3] * scale_y)
                        
                        # Validate bbox
                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence
                            })
            
            # Apply Non-Maximum Suppression
            if detections:
                detections = self._apply_nms(detections)
            
            return detections
            
        except Exception as e:
            print(f"❌ SCRFD output parsing error: {e}")
            return []
    
    def _apply_nms(self, detections):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        try:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            
            # Convert to (x, y, w, h) format for NMS
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), 
                                     self.confidence_threshold, self.nms_threshold)
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [detections[i] for i in indices]
            
        except Exception as e:
            print(f"❌ NMS error: {e}")
        
        return detections

# Initialize face recognition model based on configuration
face_app = None
balanced_detector = None  # For SCRFD balanced model
retinaface_detector = None  # For RetinaFace model
active_model = None

def initialize_face_model():
    """Initialize the selected face recognition model"""
    global face_app, balanced_detector, retinaface_detector, active_model
    
    model_type = CCTVAttendanceConfig.MODEL_TYPE
    if model_type is None:
        print("❌ No model type selected!")
        return None
        
    model_type = model_type.lower()
    print(f"\n🤖 Initializing {model_type.upper()} model...")
    print("⏳ Please wait while the model loads...")
    
    try:
        if model_type == "retinaface" and RETINAFACE_AVAILABLE:
            # Initialize RetinaFace model
            try:
                print("📥 Loading RetinaFace model...")
                # RetinaFace initialization - it will download models automatically
                active_model = "retinaface"
                print("✅ RetinaFace model initialized successfully!")
                print("🚀 Ready for high accuracy face detection and recognition")
            except Exception as retina_error:
                print(f"❌ RetinaFace initialization failed: {retina_error}")
                print("🔄 Falling back to InsightFace...")
                # Fall back to InsightFace
                face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                face_app.prepare(ctx_id=0, 
                                det_size=CCTVAttendanceConfig.DETECTION_SIZE,
                                det_thresh=CCTVAttendanceConfig.DETECTION_THRESHOLD)
                active_model = "insightface"
                print("✅ InsightFace fallback initialized successfully")
            
        elif model_type == "balanced" and ONNX_AVAILABLE:
            # Initialize SCRFD balanced model
            try:
                print("📥 Loading SCRFD model (may download on first use)...")
                balanced_detector = SCRFDDetector()
                active_model = "balanced"
                print("✅ SCRFD Balanced model initialized successfully!")
                print("🚀 Ready for optimal balance of speed and accuracy")
            except Exception as scrfd_error:
                print(f"❌ SCRFD initialization failed: {scrfd_error}")
                print("🔄 Falling back to InsightFace...")
                # Fall back to InsightFace
                face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                face_app.prepare(ctx_id=0, 
                                det_size=CCTVAttendanceConfig.DETECTION_SIZE,
                                det_thresh=CCTVAttendanceConfig.DETECTION_THRESHOLD)
                active_model = "insightface"
                print("✅ InsightFace fallback initialized successfully")
            
        elif model_type == "insightface" or not (ONNX_AVAILABLE or RETINAFACE_AVAILABLE):
            # Fall back to InsightFace (current implementation)
            face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, 
                            det_size=CCTVAttendanceConfig.DETECTION_SIZE,
                            det_thresh=CCTVAttendanceConfig.DETECTION_THRESHOLD)
            active_model = "insightface"
            print("✅ InsightFace model initialized successfully!")
            print("🚀 Ready for highest accuracy face recognition")
            
        else:
            raise Exception(f"Model {model_type} not available or not installed")
            
    except Exception as e:
        print(f"❌ Error initializing {model_type} model: {e}")
        if model_type != "insightface":
            print("🔄 Falling back to InsightFace...")
            try:
                face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                face_app.prepare(ctx_id=0, 
                                det_size=CCTVAttendanceConfig.DETECTION_SIZE,
                                det_thresh=CCTVAttendanceConfig.DETECTION_THRESHOLD)
                active_model = "insightface"
                print("✓ InsightFace fallback initialized successfully")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                exit()
        else:
            exit()
    
    # Update configuration based on active model
    if active_model in CCTVAttendanceConfig.MODEL_CONFIGS:
        config = CCTVAttendanceConfig.MODEL_CONFIGS[active_model]
        if "process_every_n_frames" in config:
            CCTVAttendanceConfig.PROCESS_EVERY_N_FRAMES = config["process_every_n_frames"]
        if "frame_resize_factor" in config:
            CCTVAttendanceConfig.FRAME_RESIZE_FACTOR = config["frame_resize_factor"]
        if "similarity_threshold" in config:
            CCTVAttendanceConfig.SIMILARITY_THRESHOLD = config["similarity_threshold"]
        if "max_faces_per_frame" in config:
            CCTVAttendanceConfig.MAX_FACES_PER_FRAME = config["max_faces_per_frame"]
    
    print(f"🔧 Model configuration updated for {active_model.upper()}")
    return active_model

def select_face_recognition_model():
    """Interactive model selection for face recognition"""
    print("\n🎯 FACE RECOGNITION MODEL SELECTION")
    print("=" * 50)
    
    # Display available models with their status
    models = {
        "1": {
            "name": "RetinaFace",
            "code": "retinaface",
            "available": RETINAFACE_AVAILABLE,
            "speed": "⚡ FAST",
            "accuracy": "🎯 Very Good",
            "description": "High accuracy face detection with good speed",
            "install": "pip install retina-face"
        },
        "2": {
            "name": "Balanced (SCRFD)",
            "code": "balanced",
            "available": ONNX_AVAILABLE,
            "speed": "⚖️ BALANCED",
            "accuracy": "🎯 Very Good",
            "description": "Best balance of speed and accuracy using SCRFD",
            "install": "pip install onnxruntime"
        },
        "3": {
            "name": "InsightFace",
            "code": "insightface",
            "available": True,  # Always available
            "speed": "🐌 SLOWER",
            "accuracy": "🎯 HIGHEST",
            "description": "Highest accuracy, best for precise recognition",
            "install": "Already installed"
        }
    }
    
    # Display model options
    available_choices = []
    for key, model in models.items():
        status = "✅ Available" if model["available"] else "❌ Not Installed"
        print(f"\n{key}. {model['name']} - {status}")
        print(f"   Speed: {model['speed']} | Accuracy: {model['accuracy']}")
        print(f"   📝 {model['description']}")
        if not model["available"]:
            print(f"   💾 Install: {model['install']}")
        else:
            available_choices.append(key)
    
    # Get user choice
    while True:
        print(f"\n🔹 Available options: {', '.join(available_choices)}")
        choice = input("👆 Select your preferred model (1-3): ").strip()
        
        if choice in models:
            selected_model = models[choice]
            if selected_model["available"]:
                print(f"\n✅ Selected: {selected_model['name']} ({selected_model['code']})")
                print(f"📊 Performance: {selected_model['speed']} speed, {selected_model['accuracy']} accuracy")
                print(f"📝 Description: {selected_model['description']}")
                
                # Confirm choice
                confirm = input(f"\n🤔 Confirm selection of {selected_model['name']}? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '1', 'ok', '']:  # Empty input also confirms
                    CCTVAttendanceConfig.MODEL_TYPE = selected_model["code"]
                    return selected_model["code"]
                else:
                    print("🔄 Let's choose again...")
                    continue
            else:
                print(f"\n❌ {selected_model['name']} is not installed!")
                install_choice = input(f"💾 Would you like installation instructions? (y/n): ").strip().lower()
                if install_choice in ['y', 'yes']:
                    print(f"\n📦 To install {selected_model['name']}:")
                    print(f"   {selected_model['install']}")
                    print("   Then restart the program.")
                
                # Fallback suggestion
                print(f"\n💡 Suggestion: Try InsightFace (option 4) - it's already installed!")
                continue
        else:
            print("❌ Invalid choice! Please select 1, 2, or 3.")
            continue

# Check available models and get user selection
print("📋 Checking available models...")
print(f"  RetinaFace: {'✓ Available' if RETINAFACE_AVAILABLE else '✗ Not installed (pip install retina-face)'}")
print(f"  Balanced (SCRFD): {'✓ Available' if ONNX_AVAILABLE else '✗ Not installed (pip install onnxruntime)'}")
print(f"  InsightFace: ✓ Available (already installed)")

# Get user's model choice
selected_model_type = select_face_recognition_model()
active_model = initialize_face_model()

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
        """Enroll face with optimized processing for CCTV system using active model"""
        global face_app  # Declare global at the beginning of function
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                print(f"❌ Image not found: {self.image_path}")
                return None
            
            # Resize for consistent enrollment processing
            img = cv2.resize(img, (640, 640))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if active_model == "retinaface":
                # Use RetinaFace for detection and face feature extraction
                try:
                    # Try multiple detection approaches for better enrollment
                    detections = None
                    
                    # First try with original image
                    try:
                        detections = RetinaFace.detect_faces(img_rgb)
                    except:
                        pass
                    
                    # If no detection, try with different image sizes
                    if not detections:
                        try:
                            # Try with larger image
                            img_large = cv2.resize(img_rgb, (800, 800))
                            detections = RetinaFace.detect_faces(img_large)
                            if detections:
                                # Scale coordinates back to original size
                                scale_factor = 640 / 800
                                for key in detections:
                                    facial_area = detections[key]['facial_area']
                                    detections[key]['facial_area'] = [
                                        int(coord * scale_factor) for coord in facial_area
                                    ]
                        except:
                            pass
                    
                    # If still no detection, try with different preprocessing
                    if not detections:
                        try:
                            # Try with enhanced contrast
                            img_enhanced = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=10)
                            detections = RetinaFace.detect_faces(img_enhanced)
                        except:
                            pass
                    
                    if detections:
                        # Get the best detection (highest confidence)
                        best_key = max(detections.keys(), key=lambda k: detections[k]['score'])
                        best_detection = detections[best_key]
                        
                        print(f"✓ RetinaFace detected face with confidence: {best_detection['score']:.3f}")
                        
                        # Extract face region
                        facial_area = best_detection['facial_area']
                        x1, y1, x2, y2 = facial_area
                        
                        # Extract face region with padding
                        padding = 20
                        face_region = img_rgb[max(0, y1-padding):min(img_rgb.shape[0], y2+padding), 
                                            max(0, x1-padding):min(img_rgb.shape[1], x2+padding)]
                        
                        if face_region.size > 0:
                            # Use InsightFace for embedding generation
                            if face_app is None:
                                from insightface.app import FaceAnalysis
                                face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                                face_app.prepare(ctx_id=0, det_size=(640, 640))
                            
                            faces = face_app.get(face_region)
                            if faces and len(faces) > 0:
                                print(f"✓ Generated embedding for {self.name}")
                                return faces[0].embedding, self.name, self.employee_id, self.department
                    else:
                        print(f"⚠️ RetinaFace could not detect face in {self.image_path}")
                        
                except Exception as retina_error:
                    print(f"RetinaFace processing error: {retina_error}")
                    
            elif active_model == "balanced":
                # Use SCRFD for detection and InsightFace for recognition
                try:
                    detections = balanced_detector.detect_faces(img_rgb)
                    if detections:
                        # Get the best detection
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        x1, y1, x2, y2 = best_detection['bbox']
                        
                        # Extract face region
                        face_region = img_rgb[y1:y2, x1:x2]
                        if face_region.size > 0:
                            # Use InsightFace for embedding generation
                            if face_app is None:
                                # Initialize InsightFace for embedding if not already done
                                from insightface.app import FaceAnalysis
                                face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                                face_app.prepare(ctx_id=0, det_size=(640, 640))
                            
                            faces = face_app.get(face_region)
                            if faces and len(faces) > 0:
                                return faces[0].embedding, self.name, self.employee_id, self.department
                except Exception as balanced_error:
                    print(f"Balanced model processing error: {balanced_error}")
                    
            elif active_model == "insightface":
                # Use InsightFace (original implementation)
                faces = face_app.get(img_rgb)
                if faces and len(faces) > 0:
                    face = faces[0]
                    print(f"✓ InsightFace detected face with score: {face.det_score:.3f}")
                    return face.embedding, self.name, self.employee_id, self.department
            
            # Fallback to InsightFace if the selected model fails
            if active_model != "insightface":
                print(f"⚠️ {active_model} failed, trying InsightFace fallback...")
                try:
                    if face_app is None:
                        from insightface.app import FaceAnalysis
                        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                        face_app.prepare(ctx_id=0, det_size=(640, 640))
                    
                    faces = face_app.get(img_rgb)
                    if faces and len(faces) > 0:
                        face = faces[0]
                        print(f"✓ InsightFace fallback succeeded with score: {face.det_score:.3f}")
                        return face.embedding, self.name, self.employee_id, self.department
                except Exception as fallback_error:
                    print(f"❌ InsightFace fallback also failed: {fallback_error}")
            
            print(f"❌ No face detected in: {self.image_path} using {active_model}")
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
    """Optimized face recognition for CCTV processing using active model"""
    global active_model, face_app
    current_time = time.time()
    
    # Debug: Print model being used (only occasionally to avoid spam)
    if hasattr(recognize_faces_cctv, 'debug_counter'):
        recognize_faces_cctv.debug_counter += 1
    else:
        recognize_faces_cctv.debug_counter = 1
    
    if recognize_faces_cctv.debug_counter % 30 == 1:  # More frequent debug for troubleshooting
        print(f"🔍 DEBUG: Using {active_model.upper()} model - Frame {recognize_faces_cctv.debug_counter}")
    
    # Resize frame for faster CCTV processing
    height, width = frame.shape[:2]
    new_width = int(width * CCTVAttendanceConfig.FRAME_RESIZE_FACTOR)
    new_height = int(height * CCTVAttendanceConfig.FRAME_RESIZE_FACTOR)
    small_frame = cv2.resize(frame, (new_width, new_height))
    
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    if recognize_faces_cctv.debug_counter % 300 == 1:
        print(f"🔍 DEBUG: Frame size: {width}x{height} -> {new_width}x{new_height}")
    
    try:
        faces = []
        face_embeddings = []
        face_boxes = []
        
        if active_model == "retinaface":
            # RetinaFace face detection and recognition
            try:
                detections = RetinaFace.detect_faces(rgb_frame)
                
                if recognize_faces_cctv.debug_counter % 100 == 1:
                    print(f"🔍 DEBUG: RetinaFace detected {len(detections) if detections else 0} faces")
                
                if detections:
                    # Sort by confidence and take top faces
                    detection_items = [(k, v) for k, v in detections.items()]
                    detection_items = sorted(detection_items, key=lambda x: x[1]['score'], reverse=True)
                    detection_items = detection_items[:CCTVAttendanceConfig.MAX_FACES_PER_FRAME]
                    
                    for detection_key, detection in detection_items:
                        facial_area = detection['facial_area']
                        x1, y1, x2, y2 = facial_area
                        confidence = detection['score']
                        
                        # Validate face size and position
                        face_w, face_h = x2 - x1, y2 - y1
                        if face_w < 40 or face_h < 40 or face_w > 300 or face_h > 300:
                            continue
                        
                        aspect_ratio = face_w / face_h
                        if aspect_ratio < 0.6 or aspect_ratio > 1.5:
                            continue
                        
                        face_boxes.append([x1, y1, x2, y2])
                        
                        # Extract face region for recognition
                        face_roi = rgb_frame[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            try:
                                # Use InsightFace for embedding generation
                                if face_app is None:
                                    from insightface.app import FaceAnalysis
                                    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                                    face_app.prepare(ctx_id=0, det_size=(640, 640))
                                
                                faces_in_roi = face_app.get(face_roi)
                                if faces_in_roi and len(faces_in_roi) > 0:
                                    face_embeddings.append(faces_in_roi[0].embedding)
                                else:
                                    # Remove invalid bbox if no face found
                                    if [x1, y1, x2, y2] in face_boxes:
                                        face_boxes.remove([x1, y1, x2, y2])
                            except Exception as retina_error:
                                print(f"❌ RetinaFace embedding error: {retina_error}")
                                if [x1, y1, x2, y2] in face_boxes:
                                    face_boxes.remove([x1, y1, x2, y2])
            except Exception as e:
                print(f"❌ RetinaFace detection error: {e}")
                
        elif active_model == "balanced":
            # Use SCRFD for balanced detection and InsightFace for recognition
            detections = balanced_detector.detect_faces(rgb_frame)
            
            if recognize_faces_cctv.debug_counter % 100 == 1:
                print(f"🔍 DEBUG: SCRFD Balanced detected {len(detections)} faces")
            
            if detections:
                # Sort by confidence and take top faces
                detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                detections = detections[:CCTVAttendanceConfig.MAX_FACES_PER_FRAME]
                
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Validate face size and position
                    face_w, face_h = x2 - x1, y2 - y1
                    if face_w < 40 or face_h < 40 or face_w > 300 or face_h > 300:
                        continue
                    
                    aspect_ratio = face_w / face_h
                    if aspect_ratio < 0.6 or aspect_ratio > 1.5:
                        continue
                    
                    face_boxes.append([x1, y1, x2, y2])
                    
                    # Extract face region for recognition
                    face_roi = rgb_frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        try:
                            # Use InsightFace for embedding generation
                            if face_app is None:
                                # Initialize InsightFace for embedding if not already done
                                from insightface.app import FaceAnalysis
                                face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                                face_app.prepare(ctx_id=0, det_size=(640, 640))
                            
                            faces = face_app.get(face_roi)
                            if faces and len(faces) > 0:
                                face_embeddings.append(faces[0].embedding)
                            else:
                                # Remove invalid bbox if no face found
                                if [x1, y1, x2, y2] in face_boxes:
                                    face_boxes.remove([x1, y1, x2, y2])
                                    face_boxes.remove([x1, y1, x2, y2])
                        except Exception as balanced_error:
                            print(f"❌ Balanced model embedding error: {balanced_error}")
                            if [x1, y1, x2, y2] in face_boxes:
                                face_boxes.remove([x1, y1, x2, y2])
                
        elif active_model == "insightface":
            # Original InsightFace implementation
            faces = face_app.get(rgb_frame)
            
            # Debug output for InsightFace detection
            if recognize_faces_cctv.debug_counter % 30 == 1:
                print(f"🔍 DEBUG: InsightFace detected {len(faces)} faces")
                if len(faces) > 0:
                    for i, face in enumerate(faces):
                        print(f"🔍 DEBUG: Face {i+1} - Detection score: {face.det_score:.3f}")
            
            # Limit faces for CCTV performance
            if len(faces) > CCTVAttendanceConfig.MAX_FACES_PER_FRAME:
                faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:CCTVAttendanceConfig.MAX_FACES_PER_FRAME]
        
        results = []
        
        # Debug output for detection results (more frequent for debugging)
        if recognize_faces_cctv.debug_counter % 30 == 1:
            print(f"🔍 DEBUG: Found {len(face_boxes)} face boxes, {len(face_embeddings)} embeddings, {len(known_face_encodings)} known faces")
            if active_model == "insightface" and 'faces' in locals():
                print(f"🔍 DEBUG: InsightFace faces found: {len(faces)}")
            if len(face_embeddings) > 0 and len(known_face_encodings) > 0:
                print(f"🔍 DEBUG: Embedding size: {len(face_embeddings[0]) if face_embeddings else 'N/A'}")
                print(f"🔍 DEBUG: Known encoding sizes: {[len(enc) for enc, _, _, _ in known_face_encodings]}")
        
        # Process faces based on active model
        if active_model == "insightface" and faces and known_face_encodings:
            # Original InsightFace processing
            known_encodings = np.array([enc for enc, _, _, _ in known_face_encodings], dtype=np.float32)
            known_norms = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)
            
            for face in faces:
                face_embedding = face.embedding.astype(np.float32)
                face_norm = face_embedding / np.linalg.norm(face_embedding)
                
                # Fast vectorized similarity computation
                similarities = np.dot(known_norms, face_norm)
                max_idx = np.argmax(similarities)
                max_similarity = similarities[max_idx]
                
                # Debug output for similarity calculations
                if recognize_faces_cctv.debug_counter % 30 == 1:
                    known_names = [name for _, name, _, _ in known_face_encodings]
                    print(f"🔍 DEBUG: InsightFace similarities: {[f'{s:.3f}' for s in similarities]}")
                    print(f"🔍 DEBUG: Known faces: {known_names}")
                    print(f"🔍 DEBUG: Max similarity: {max_similarity:.3f}, Threshold: {CCTVAttendanceConfig.SIMILARITY_THRESHOLD:.3f}")
                
                # Get bbox in (x1,y1,x2,y2) format and scale to original frame size
                bbox = face.bbox.astype(int)  # InsightFace gives [x1,y1,x2,y2]
                scale_factor = 1.0 / CCTVAttendanceConfig.FRAME_RESIZE_FACTOR
                
                # Scale bbox coordinates back to original frame size
                x1, y1, x2, y2 = bbox
                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)
                bbox = [x1, y1, x2, y2]
                
                if max_similarity > CCTVAttendanceConfig.SIMILARITY_THRESHOLD:
                    _, name, emp_id, dept = known_face_encodings[max_idx]
                    
                    # Check cooldown and process recognition
                    status = process_recognition(name, emp_id, dept, current_time, frame, bbox)
                    
                    results.append({
                        "name": name,
                        "employee_id": emp_id,
                        "department": dept,
                        "confidence": float(max_similarity),
                        "bbox": bbox,
                        "status": status
                    })
                else:
                    results.append({
                        "name": "unknown",
                        "confidence": 0,
                        "bbox": bbox,
                        "status": "unknown"
                    })
                    
        elif (active_model in ["retinaface", "balanced"]) and face_embeddings and known_face_encodings:
            # Processing for RetinaFace and other models
            known_encodings = np.array([enc for enc, _, _, _ in known_face_encodings], dtype=np.float32)
            
            # Handle different embedding sizes
            if len(face_embeddings) > 0:
                embedding_size = len(face_embeddings[0])
                # Filter known encodings to match embedding size
                valid_known_encodings = []
                for enc, name, emp_id, dept in known_face_encodings:
                    if len(enc) == embedding_size:
                        valid_known_encodings.append((enc, name, emp_id, dept))
                
                if valid_known_encodings:
                    known_encodings = np.array([enc for enc, _, _, _ in valid_known_encodings], dtype=np.float32)
                    
                    for i, face_embedding in enumerate(face_embeddings):
                        if i < len(face_boxes):
                            face_embedding = np.array(face_embedding, dtype=np.float32)
                            
                            # Use model-specific threshold
                            threshold = CCTVAttendanceConfig.MODEL_CONFIGS.get(active_model, {}).get("similarity_threshold", CCTVAttendanceConfig.SIMILARITY_THRESHOLD)
                            
                            # Compute cosine similarities consistently
                            similarities = []
                            for known_enc in known_encodings:
                                # Normalize vectors for cosine similarity
                                face_norm = face_embedding / (np.linalg.norm(face_embedding) + 1e-8)
                                known_norm = known_enc / (np.linalg.norm(known_enc) + 1e-8)
                                # Cosine similarity
                                similarity = np.dot(face_norm, known_norm)
                                similarities.append(similarity)
                            
                            if similarities:
                                max_idx = np.argmax(similarities)
                                max_similarity = similarities[max_idx]
                                
                                # Debug output for similarity matching (more frequent for debugging)
                                if recognize_faces_cctv.debug_counter % 100 == 1:
                                    print(f"🔍 DEBUG: Face {i+1} similarities: {[f'{s:.3f}' for s in similarities]}")
                                    print(f"🔍 DEBUG: Max similarity: {max_similarity:.3f}, Threshold: {threshold:.3f}")
                                    print(f"🔍 DEBUG: Known faces: {[name for _, name, _, _ in valid_known_encodings]}")
                                
                                # Scale bbox back to original frame size
                                bbox = face_boxes[i]  # Already in [x1,y1,x2,y2] format
                                scale_factor = 1.0 / CCTVAttendanceConfig.FRAME_RESIZE_FACTOR
                                
                                # Scale coordinates properly
                                x1, y1, x2, y2 = bbox
                                x1 = int(x1 * scale_factor)
                                y1 = int(y1 * scale_factor)
                                x2 = int(x2 * scale_factor)
                                y2 = int(y2 * scale_factor)
                                scaled_bbox = [x1, y1, x2, y2]
                                
                                if max_similarity > threshold:
                                    _, name, emp_id, dept = valid_known_encodings[max_idx]
                                    
                                    # Check cooldown and process recognition
                                    status = process_recognition(name, emp_id, dept, current_time, frame, scaled_bbox)
                                    
                                    results.append({
                                        "name": name,
                                        "employee_id": emp_id,
                                        "department": dept,
                                        "confidence": float(max_similarity),
                                        "bbox": scaled_bbox,
                                        "status": status
                                    })
                                else:
                                    results.append({
                                        "name": "unknown",
                                        "confidence": 0,
                                        "bbox": scaled_bbox,
                                        "status": "unknown"
                                    })
                                
                                if max_similarity > threshold:
                                    _, name, emp_id, dept = valid_known_encodings[max_idx]
                                    
                                    # Check cooldown and process recognition
                                    status = process_recognition(name, emp_id, dept, current_time, frame, scaled_bbox)
                                    
                                    results.append({
                                        "name": name,
                                        "employee_id": emp_id,
                                        "department": dept,
                                        "confidence": float(max_similarity),
                                        "bbox": scaled_bbox,
                                        "status": status
                                    })
                                else:
                                    results.append({
                                        "name": "unknown",
                                        "confidence": 0,
                                        "bbox": scaled_bbox,
                                        "status": "unknown"
                                    })
        
        return results
    except Exception as e:
        print(f"❌ Recognition error: {e}")
        return []
                
                
def process_recognition(name, emp_id, dept, current_time, frame, bbox):
    """Helper function to process recognition and attendance marking"""
    try:
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
                    return "marked"
                else:
                    return "already_marked"
            else:
                return "confirming"
        else:
            return "cooldown"
    except Exception as e:
        print(f"❌ Process recognition error: {e}")
        return "error"

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
    
    print("\n" + "="*60)
    print("🏢📹 CCTV ATTENDANCE SYSTEM - EMPLOYEE DATABASE")
    print("="*60)
    
    # Load employees - CUSTOMIZABLE SECTION
    employees_loaded = 0
    employees_loaded += add_employee('yaseen', '1170', 'AI&DS', 'yaseen.jpg')
    employees_loaded += add_employee('sajj', '9876', 'AI&DS', 'sajj.jpg')
    
    # Add more employees here as needed:
    # employees_loaded += add_employee('employee_name', 'employee_id', 'department', 'image.jpg')
    
    if not known_face_encodings:
        print("❌ No employees enrolled! Add employee images first.")
        return
    
    print(f"\n✅ System ready with {len(known_face_encodings)} enrolled employee(s)")
    
    # Display selected model information
    model_info = {
        "retinaface": "⚡ FAST - High accuracy face detection",
        "balanced": "🎯 OPTIMAL - Best balance with SCRFD",
        "insightface": "🎯 HIGHEST ACCURACY - Best precision"
    }
    
    print(f"\n🤖 ACTIVE MODEL: {active_model.upper()}")
    print(f"📊 Performance Profile: {model_info.get(active_model, 'Unknown')}")
    
    # Debug: Show embedding information
    print("\n🔍 Enrolled Face Embeddings:")
    for i, (encoding, name, emp_id, dept) in enumerate(known_face_encodings):
        print(f"  {i+1}. {name}: Embedding size = {len(encoding)}, Type = {type(encoding).__name__}")
    
    print(f"\n{'='*60}")
    print("� STARTING CCTV MONITORING...")
    print(f"{'='*60}")
    
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
                    
                    # Ensure bbox is in correct format [x1, y1, x2, y2]
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    else:
                        continue  # Skip invalid bounding boxes
                    
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
            
            # Display system title with model info
            model_info = f" ({active_model.upper()})"
            cv2.putText(frame, f"CCTV ATTENDANCE SYSTEM{model_info}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
            
            # Show streaming info every 30 seconds with model-specific info
            if frame_count % (fps * 30) == 0:
                elapsed = time.time() - fps_start_time
                print(f"📡 CCTV Monitoring ({active_model.upper()}): {frame_count} frames processed - FPS: {current_fps:.1f}")
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