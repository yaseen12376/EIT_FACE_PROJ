import cv2
import numpy as np
import csv
import time
import os
import sys
import urllib.request
from insightface.app import FaceAnalysis
from tkinter import Tk, messagebox
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

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
    
    # DUAL CAMERA CONNECTION SETTINGS
    # Primary Camera (CHECK_IN) - Existing CCTV
    CHECKIN_CAMERA_URL = "rtsp://admin:AK@MrA!4501$uf@192.168.0.109:554/cam/realmonitor?channel=8&subtype=0"
    CHECKIN_CAMERA_TYPE = "RTSP"  # RTSP, HTTP, USB
    
    # Secondary Camera (CHECK_OUT) - IP WebCam Mobile Phone
    CHECKOUT_CAMERA_URL = "http://192.168.0.180:8080/video"  # Your IP WebCam URL
    CHECKOUT_CAMERA_TYPE = "HTTP"  # IP WebCam uses HTTP/HTTPS
    
    # Legacy RTSP URL (for backward compatibility)
    RTSP_URL = "rtsp://admin:AK@MrA!4501$uf@192.168.0.109:554/cam/realmonitor?channel=8&subtype=0"
    ALTERNATIVE_URLS = []  # Add backup URLs if needed
    
    # Dual Camera Configuration
    DUAL_CAMERA_MODE = True       # Enable dual camera system
    SINGLE_CAMERA_FALLBACK = True # Fallback to single camera if one fails
    
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
            "similarity_threshold": 0.25,  # Balanced threshold for accuracy
            "process_every_n_frames": 2,  # Good balance of speed and accuracy
            "frame_resize_factor": 0.9,  # Balanced resolution
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
    RECOGNITION_COOLDOWN = 0.0        # NO COOLDOWN for dual camera mode - immediate checkout allowed
    MIN_CONSECUTIVE_DETECTIONS = 2    # Fewer detections needed for CCTV
    CONNECTION_RETRY_INTERVAL = 5.0   # Retry connection every 5 seconds
    
    # Dual-Camera Session Management
    ENABLE_SESSION_MANAGEMENT = True  # Track check-in/check-out status
    AUTO_CHECKOUT_TIME = "18:00"       # Automatic checkout time (6 PM)
    MANUAL_CHECKOUT_ENABLED = True     # Allow manual checkout via keyboard
    SESSION_TIMEOUT_HOURS = 8          # Max session length before auto-checkout
    
    # Disable Auto-Checkout (Use Mobile Camera Instead)
    ENABLE_AUTO_CHECKOUT = False       # DISABLED - Use mobile camera for checkout
    AUTO_CHECKOUT_MINUTES = 0          # DISABLED - No more 2-minute auto-checkout
    
    # IP WebCam Configuration (for mobile phone checkout camera)
    IP_WEBCAM_CONFIG = {
        'default_port': 8080,
        'video_path': '/video',      # IP WebCam default path
        'photo_path': '/photo.jpg',  # For single frame capture
        'username': '',              # Optional authentication
        'password': '',              # Optional authentication
        'quality': 'medium',         # low, medium, high
        'timeout': 10,               # Connection timeout seconds
        'retry_attempts': 3          # Connection retry attempts
    }
    
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
            print("📥 SCRFD model not found. Checking for manual installation...")
            print("\n" + "="*60)
            print("📦 SCRFD MODEL SETUP REQUIRED")
            print("="*60)
            print("The SCRFD model needs to be downloaded manually.")
            print("Please follow these steps:")
            print("\n1. Download the SCRFD model from:")
            print("   https://github.com/deepinsight/insightface/tree/master/detection/scrfd")
            print("   OR search for 'scrfd_2.5g.onnx' online")
            print("\n2. Place the downloaded 'scrfd_2.5g.onnx' file in:")
            print(f"   {os.path.abspath(model_dir)}/")
            print("\n3. Restart the program")
            print("\n💡 Alternative: Use RetinaFace or InsightFace which work without manual setup")
            print("="*60)
            
            # Try a simplified approach with a different model
            print("\n🔄 Attempting to use a simplified balanced approach...")
            print("⚠️ SCRFD model unavailable - falling back to InsightFace with optimizations")
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
            
            # Convert BGR to RGB if needed
            if len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] and convert to CHW format
            input_tensor = resized.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
            
            print(f"🔍 DEBUG: SCRFD input shape: {input_tensor.shape}")
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # Parse outputs and get detections
            detections = self._parse_outputs(outputs, original_width, original_height)
            
            return detections
            
        except Exception as e:
            print(f"❌ SCRFD detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_outputs(self, outputs, original_width, original_height):
        """Parse SCRFD model outputs"""
        detections = []
        
        try:
            # SCRFD outputs multiple scales - check actual output structure
            print(f"🔍 DEBUG: SCRFD outputs count: {len(outputs)}")
            
            # SCRFD typically has multiple output layers for different scales
            # Each output contains [batch, num_anchors, 5] where 5 = [x1,y1,x2,y2,score]
            for i, output in enumerate(outputs):
                print(f"🔍 DEBUG: Output {i} shape: {output.shape}")
                
                # Skip if output doesn't have expected dimensions
                if len(output.shape) != 3 or output.shape[2] < 5:
                    continue
                
                # Scale factors
                scale_x = original_width / self.input_size[0]
                scale_y = original_height / self.input_size[1]
                
                # Process each detection in this output layer
                for j in range(output.shape[1]):
                    # Extract confidence (last element)
                    confidence = float(output[0, j, 4])  # Score is at index 4
                    
                    if confidence > self.confidence_threshold:
                        # Extract bbox coordinates (first 4 elements)
                        x1 = int(output[0, j, 0] * scale_x)
                        y1 = int(output[0, j, 1] * scale_y)
                        x2 = int(output[0, j, 2] * scale_x)
                        y2 = int(output[0, j, 3] * scale_y)
                        
                        # Validate bbox
                        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence
                            })
            
            print(f"🔍 DEBUG: SCRFD raw detections: {len(detections)}")
            
            # Apply Non-Maximum Suppression
            if detections:
                detections = self._apply_nms(detections)
                print(f"🔍 DEBUG: SCRFD after NMS: {len(detections)}")
            
            return detections
            
        except Exception as e:
            print(f"❌ SCRFD output parsing error: {e}")
            print(f"🔍 DEBUG: Available outputs: {[out.shape for out in outputs]}")
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
            
            # Ensure positive dimensions
            valid_indices = (boxes_xywh[:, 2] > 0) & (boxes_xywh[:, 3] > 0)
            if not np.any(valid_indices):
                return []
            
            boxes_xywh = boxes_xywh[valid_indices]
            scores = scores[valid_indices]
            valid_detections = [detections[i] for i in range(len(detections)) if valid_indices[i]]
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), 
                                     self.confidence_threshold, self.nms_threshold)
            
            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                elif isinstance(indices, list):
                    indices = [idx[0] if isinstance(idx, list) else idx for idx in indices]
                
                return [valid_detections[i] for i in indices]
            
        except Exception as e:
            print(f"❌ NMS error: {e}")
            # Return original detections if NMS fails
            return detections[:5]  # Limit to top 5 to avoid too many false positives
        
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
            # Initialize optimized InsightFace for balanced performance (SCRFD alternative)
            try:
                print("📥 Loading Balanced model (optimized InsightFace)...")
                # Instead of SCRFD, use InsightFace with balanced settings
                face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
                # Use smaller detection size for better speed
                balanced_det_size = CCTVAttendanceConfig.MODEL_CONFIGS["balanced"]["detection_size"]
                face_app.prepare(ctx_id=0, 
                                det_size=balanced_det_size,
                                det_thresh=0.5)  # Higher threshold for speed
                active_model = "balanced"
                print("✅ Balanced model (optimized InsightFace) initialized successfully!")
                print("🚀 Ready for optimal balance of speed and accuracy")
            except Exception as balanced_error:
                print(f"❌ Balanced model initialization failed: {balanced_error}")
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

# Enhanced Time Tracking Variables
employee_time_entries = defaultdict(list)  # Store all time entries for each employee
employee_work_sessions = defaultdict(list)  # Store calculated work sessions
time_tracking_enabled = True

# Single-Camera Session Management
employee_checked_in_status = {}    # Track who is currently checked in
employee_last_checkout = {}        # Track last checkout time for each employee
manual_checkout_queue = set()      # Queue for manual checkouts

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
                # Use optimized InsightFace for balanced enrollment
                try:
                    faces = face_app.get(img_rgb)
                    if faces and len(faces) > 0:
                        # Get the best face (highest detection score)
                        best_face = max(faces, key=lambda x: x.det_score)
                        print(f"✓ Balanced model detected face with score: {best_face.det_score:.3f}")
                        return best_face.embedding, self.name, self.employee_id, self.department
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

def test_camera_connection(camera_url, camera_type="RTSP", camera_name="Camera"):
    """Test individual camera connection and return capture object with properties"""
    print(f"🔗 Testing {camera_name} connection...")
    print(f"   🌐 URL: {camera_url}")
    print(f"   📡 Type: {camera_type}")
    
    cap = None
    
    try:
        if camera_type.upper() == "HTTP" or camera_type.upper() == "HTTPS":
            # For IP WebCam (HTTP streams)
            cap = cv2.VideoCapture(camera_url)
            # Reduce buffer for HTTP streams to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        elif camera_type.upper() == "RTSP":
            # For RTSP streams (traditional CCTV)
            cap = cv2.VideoCapture(camera_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        else:  # USB or other
            cap = cv2.VideoCapture(int(camera_url) if camera_url.isdigit() else camera_url)
        
        if cap and cap.isOpened():
            # Test if we can read a frame
            ret, test_frame = cap.read()
            
            if ret and test_frame is not None:
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Use sensible defaults if properties are invalid
                if width <= 0 or height <= 0:
                    width, height = 1280, 720  # Default resolution
                if fps <= 0:
                    fps = 25 if camera_type.upper() == "RTSP" else 30
                
                print(f"✅ {camera_name} connected successfully")
                print(f"   📐 Resolution: {width} x {height}")
                print(f"   🎬 FPS: {fps}")
                print(f"   🎯 Test frame: {test_frame.shape}")
                
                return cap, (width, height, fps)
            else:
                print(f"❌ Cannot read frames from {camera_name}")
        else:
            print(f"❌ Cannot open {camera_name} stream")
    
    except Exception as e:
        print(f"❌ {camera_name} connection error: {e}")
    
    if cap:
        cap.release()
    
    return None, (0, 0, 0)

def test_dual_camera_connection():
    """Test both CHECK_IN and CHECK_OUT cameras"""
    print("\n" + "="*60)
    print("🎥🎥 DUAL CAMERA CONNECTION TEST")
    print("="*60)
    
    cameras_connected = {}
    
    # Test CHECK_IN Camera (Primary CCTV)
    if CCTVAttendanceConfig.CHECKIN_CAMERA_URL:
        checkin_cap, checkin_props = test_camera_connection(
            CCTVAttendanceConfig.CHECKIN_CAMERA_URL,
            CCTVAttendanceConfig.CHECKIN_CAMERA_TYPE,
            "CHECK_IN Camera (CCTV)"
        )
        cameras_connected['checkin'] = (checkin_cap, checkin_props)
    else:
        print("⚠️ CHECK_IN camera URL not configured")
        cameras_connected['checkin'] = (None, (0, 0, 0))
    
    print()  # Spacing
    
    # Test CHECK_OUT Camera (IP WebCam)
    if CCTVAttendanceConfig.CHECKOUT_CAMERA_URL:
        checkout_cap, checkout_props = test_camera_connection(
            CCTVAttendanceConfig.CHECKOUT_CAMERA_URL,
            CCTVAttendanceConfig.CHECKOUT_CAMERA_TYPE,
            "CHECK_OUT Camera (IP WebCam)"
        )
        cameras_connected['checkout'] = (checkout_cap, checkout_props)
    else:
        print("⚠️ CHECK_OUT camera URL not configured")
        print("💡 Please set up IP WebCam on your mobile phone")
        cameras_connected['checkout'] = (None, (0, 0, 0))
    
    # Summary
    checkin_ok = cameras_connected['checkin'][0] is not None
    checkout_ok = cameras_connected['checkout'][0] is not None
    
    print(f"\n📊 CONNECTION SUMMARY:")
    print(f"   CHECK_IN Camera: {'✅ Connected' if checkin_ok else '❌ Failed'}")
    print(f"   CHECK_OUT Camera: {'✅ Connected' if checkout_ok else '❌ Failed'}")
    
    if checkin_ok and checkout_ok:
        print("🎉 DUAL CAMERA MODE: Both cameras ready!")
        return cameras_connected, "DUAL"
    elif checkin_ok:
        print("⚠️ SINGLE CAMERA MODE: Only CHECK_IN camera available")
        if CCTVAttendanceConfig.SINGLE_CAMERA_FALLBACK:
            print("🔄 Continuing with single camera + manual checkout")
            return cameras_connected, "SINGLE_CHECKIN"
        else:
            return cameras_connected, "FAILED"
    elif checkout_ok:
        print("⚠️ SINGLE CAMERA MODE: Only CHECK_OUT camera available")
        return cameras_connected, "SINGLE_CHECKOUT"
    else:
        print("❌ NO CAMERAS AVAILABLE")
        return cameras_connected, "FAILED"

def recognize_faces_cctv(frame, entry_type="CHECK_IN"):
    """Optimized face recognition for CCTV processing using active model with dual camera support"""
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
            # Use optimized InsightFace for balanced performance
            faces = face_app.get(rgb_frame)
            
            # Debug output for balanced detection
            if recognize_faces_cctv.debug_counter % 100 == 1:
                print(f"🔍 DEBUG: Balanced (Optimized InsightFace) detected {len(faces)} faces")
            
            # Apply balanced model optimizations
            if len(faces) > CCTVAttendanceConfig.MAX_FACES_PER_FRAME:
                # Sort by detection score and keep top faces
                faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:CCTVAttendanceConfig.MAX_FACES_PER_FRAME]
            
            # Filter faces by confidence for balanced performance
            balanced_threshold = CCTVAttendanceConfig.MODEL_CONFIGS["balanced"]["detection_confidence"]
            faces = [face for face in faces if face.det_score > balanced_threshold]
            
            if recognize_faces_cctv.debug_counter % 100 == 1:
                print(f"🔍 DEBUG: Balanced model after filtering: {len(faces)} faces (threshold: {balanced_threshold})")
                
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
        if active_model in ["insightface", "balanced"] and faces and known_face_encodings:
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
                    status = process_recognition(name, emp_id, dept, current_time, frame, bbox, entry_type)
                    
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
                                    status = process_recognition(name, emp_id, dept, current_time, frame, scaled_bbox, entry_type)
                                    
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
                                    status = process_recognition(name, emp_id, dept, current_time, frame, scaled_bbox, entry_type)
                                    
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
                
                
def process_recognition(name, emp_id, dept, current_time, frame, bbox, entry_type="CHECK_IN"):
    """Enhanced recognition processing with dual-camera session management"""
    global employee_checked_in_status, employee_last_checkout
    
    try:
        # Check if employee is currently checked in
        is_checked_in = employee_checked_in_status.get(name, False)
        
        # Check cooldown period - DISABLED for dual camera mode
        cooldown_expired = (CCTVAttendanceConfig.RECOGNITION_COOLDOWN == 0.0 or 
                           name not in last_recognition_time or 
                           (current_time - last_recognition_time[name]) > CCTVAttendanceConfig.RECOGNITION_COOLDOWN)
        
        if cooldown_expired:
            
            # Track consecutive detections for confirmation
            consecutive_detections[name] = consecutive_detections.get(name, 0) + 1
            
            # Confirm recognition after consecutive detections
            if consecutive_detections[name] >= CCTVAttendanceConfig.MIN_CONSECUTIVE_DETECTIONS:
                
                # Handle dual camera logic based on entry_type
                if entry_type == "CHECK_IN":
                    if not is_checked_in:  # Person is not currently checked in
                        # Mark as CHECK-IN
                        if name not in today_attendance:
                            today_attendance.add(name)
                        
                        employee_checked_in_status[name] = True
                        mark_attendance_cctv(name, emp_id, dept, frame, bbox, entry_type="CHECK_IN")
                        last_recognition_time[name] = current_time
                        consecutive_detections[name] = 0
                        
                        print(f"✅ CHECK-IN: {name} is now checked in")
                        return "checked_in"
                    else:
                        # Person is already checked in
                        print(f"ℹ️ {name} already checked in")
                        return "already_checked_in"
                        
                elif entry_type == "CHECK_OUT":
                    if is_checked_in:  # Person is currently checked in
                        # Mark as CHECK-OUT
                        employee_checked_in_status[name] = False
                        mark_attendance_cctv(name, emp_id, dept, frame, bbox, entry_type="CHECK_OUT")
                        last_recognition_time[name] = current_time
                        consecutive_detections[name] = 0
                        
                        print(f"✅ CHECK-OUT: {name} has checked out")
                        return "checked_out"
                    else:
                        # Person is not checked in, cannot check out
                        print(f"⚠️ {name} cannot check out - not currently checked in")
                        return "not_checked_in"
            else:
                return "confirming"
        else:
            # Cooldown active (should not happen in dual camera mode with RECOGNITION_COOLDOWN = 0.0)
            if CCTVAttendanceConfig.RECOGNITION_COOLDOWN > 0:
                remaining_time = CCTVAttendanceConfig.RECOGNITION_COOLDOWN - (current_time - last_recognition_time[name])
                print(f"⏱️ {name} cooldown: {remaining_time:.0f} seconds remaining")
            return "cooldown"
            
    except Exception as e:
        print(f"❌ Process recognition error: {e}")
        return "error"

def mark_attendance_cctv(name, emp_id, dept, frame, bbox, entry_type="CHECK_IN"):
    """Enhanced attendance marking with check-in/check-out support for single-camera system"""
    global screenshot_counter, employee_time_entries, employee_work_sessions
    
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    date = current_datetime.strftime("%Y-%m-%d")
    time_only = current_datetime.strftime("%H:%M:%S")
    
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
            
            # Color coding for entry type
            color = (0, 255, 0) if entry_type == "CHECK_IN" else (0, 0, 255)  # Green for IN, Red for OUT
            
            cv2.rectangle(screenshot_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(screenshot_frame, f"{name} - {entry_type}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(screenshot_frame, timestamp, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            screenshot_counter += 1
            screenshot_filename = f"{entry_type.lower()}_{name}_{date}_{time_only.replace(':', '-')}_{screenshot_counter}.jpg"
            screenshot_path = os.path.join(attendance_dir, screenshot_filename)
            cv2.imwrite(screenshot_path, screenshot_frame)
            print(f"📸 {entry_type} Screenshot saved: {screenshot_filename}")
        except Exception as e:
            print(f"❌ Failed to save screenshot: {e}")
    
    # Enhanced Time Tracking Logic
    if entry_type == "CHECK_IN":
        process_check_in(name, emp_id, dept, current_datetime, screenshot_path)
    else:  # CHECK_OUT
        process_check_out(name, emp_id, dept, current_datetime, screenshot_path)
    
    # Save to traditional CSV (for backward compatibility)
    save_traditional_attendance_csv(name, emp_id, dept, date, time_only, timestamp, screenshot_path, entry_type)
    
    print(f"✅ {entry_type}: {name} (ID: {emp_id}) at {time_only}")

def process_check_in(name, emp_id, dept, entry_datetime, screenshot_path):
    """Process check-in entry for single-camera system"""
    global employee_time_entries
    
    # Add new time entry
    employee_time_entries[name].append({
        'datetime': entry_datetime,
        'emp_id': emp_id,
        'dept': dept,
        'screenshot_path': screenshot_path,
        'entry_type': 'CHECK_IN'
    })
    
    print(f"🟢 {name} checked in at {entry_datetime.strftime('%H:%M:%S')} (Auto-checkout in 1 min)")
    
    # Force immediate Excel update for check-ins
    print("🔄 Updating Excel immediately...")
    update_time_tracking_excel()

def process_check_out(name, emp_id, dept, exit_datetime, screenshot_path):
    """Process check-out entry and calculate work session"""
    global employee_time_entries, employee_work_sessions, employee_checked_in_status
    
    # Find the last check-in entry for this employee
    if name in employee_time_entries and employee_time_entries[name]:
        # Get the most recent check-in
        recent_entries = [entry for entry in employee_time_entries[name] if entry['entry_type'] == 'CHECK_IN']
        
        if recent_entries:
            last_checkin = recent_entries[-1]
            
            # Calculate work duration
            work_duration = exit_datetime - last_checkin['datetime']
            duration_minutes = work_duration.total_seconds() / 60
            
            # Create work session
            work_session = {
                'name': name,
                'emp_id': emp_id,
                'dept': dept,
                'in_time': last_checkin['datetime'],
                'out_time': exit_datetime,
                'duration_minutes': duration_minutes,
                'in_screenshot': last_checkin['screenshot_path'],
                'out_screenshot': screenshot_path,
                'auto_generated_out': False  # Manual checkout
            }
            
            employee_work_sessions[name].append(work_session)
            
            # Add checkout entry
            employee_time_entries[name].append({
                'datetime': exit_datetime,
                'emp_id': emp_id,
                'dept': dept,
                'screenshot_path': screenshot_path,
                'entry_type': 'CHECK_OUT'
            })
            
            # Update check-in status
            employee_checked_in_status[name] = False
            employee_last_checkout[name] = exit_datetime
            
            print(f"🔴 {name} checked out - Work session: {duration_minutes:.1f} minutes ({duration_minutes/60:.2f} hours)")
        else:
            print(f"⚠️ No check-in found for {name} - cannot process checkout")
    
    # Update Excel file
    update_time_tracking_excel()

def manual_checkout_employee(name):
    """Manually checkout an employee via keyboard command"""
    global employee_checked_in_status
    
    if name in employee_checked_in_status and employee_checked_in_status[name]:
        # Find employee details
        emp_id, dept = None, None
        for _, emp_name, emp_id_val, dept_val in known_face_encodings:
            if emp_name == name:
                emp_id, dept = emp_id_val, dept_val
                break
        
        if emp_id:
            current_datetime = datetime.now()
            # Create a dummy frame and bbox for manual checkout
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_bbox = [100, 100, 200, 200]
            
            mark_attendance_cctv(name, emp_id, dept, dummy_frame, dummy_bbox, entry_type="CHECK_OUT")
            print(f"✅ Manual checkout completed for {name}")
            return True
        else:
            print(f"❌ Employee {name} not found in database")
            return False
    else:
        print(f"⚠️ {name} is not currently checked in")
        return False

def auto_checkout_after_1_minute():
    """DISABLED: Auto-checkout replaced with mobile camera detection for dual camera mode"""
    # Auto-checkout is disabled in dual camera mode - checkout only happens via mobile camera
    if CCTVAttendanceConfig.ENABLE_AUTO_CHECKOUT:
        current_datetime = datetime.now()
        
        for name, is_checked_in in employee_checked_in_status.items():
            if is_checked_in and name in employee_time_entries:
                # Find the most recent check-in
                checkin_entries = [e for e in employee_time_entries[name] if e['entry_type'] == 'CHECK_IN']
                if checkin_entries:
                    last_checkin = checkin_entries[-1]
                    time_since_checkin = current_datetime - last_checkin['datetime']
                    
                    # Auto-checkout after exactly 1 minute (60 seconds)
                    if time_since_checkin.total_seconds() >= 60:
                        print(f"⏰ AUTO-CHECKOUT: {name} worked for 1 minute - enabling next check-in")
                        manual_checkout_employee(name)
    # If auto-checkout is disabled, do nothing - wait for mobile camera checkout

def auto_checkout_all_employees():
    """Auto checkout all employees at end of day"""
    current_time = datetime.now().strftime("%H:%M")
    auto_checkout_time = CCTVAttendanceConfig.AUTO_CHECKOUT_TIME
    
    if current_time >= auto_checkout_time:
        checked_in_employees = [name for name, status in employee_checked_in_status.items() if status]
        
        for name in checked_in_employees:
            print(f"🕕 Auto-checkout: {name} (End of day: {auto_checkout_time})")
            manual_checkout_employee(name)
        
        if checked_in_employees:
            print(f"✅ Auto-checkout completed for {len(checked_in_employees)} employees")

# Duplicate function removed - using single auto_checkout_after_1_minute function above

def display_checked_in_employees():
    """Display currently checked-in employees"""
    checked_in = [name for name, status in employee_checked_in_status.items() if status]
    
    if checked_in:
        print(f"\n👥 CURRENTLY CHECKED IN ({len(checked_in)} employees):")
        for name in checked_in:
            # Find last check-in time
            if name in employee_time_entries:
                checkin_entries = [e for e in employee_time_entries[name] if e['entry_type'] == 'CHECK_IN']
                if checkin_entries:
                    last_checkin = checkin_entries[-1]['datetime']
                    duration = datetime.now() - last_checkin
                    hours = duration.total_seconds() / 3600
                    print(f"  🟢 {name} - Checked in for {hours:.1f} hours")
    else:
        print("\n👥 No employees currently checked in")

def save_traditional_attendance_csv(name, emp_id, dept, date, time_only, timestamp, screenshot_path, entry_type="CHECK_IN"):
    """Save to traditional CSV format for backward compatibility"""
    attendance_dir = "attendance_records"
    attendance_file = os.path.join(attendance_dir, f"daily_attendance_{date}.csv")
    file_exists = os.path.exists(attendance_file)
    
    try:
        with open(attendance_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['Name', 'Employee_ID', 'Department', 'Date', 'Time', 
                               'Full_Timestamp', 'Screenshot_Path', 'Entry_Type', 'Source'])
            
            writer.writerow([name, emp_id, dept, date, time_only, timestamp, 
                            screenshot_path or 'N/A', entry_type, 'CCTV'])
    except PermissionError:
        print(f"⚠️ Permission denied writing to {attendance_file} - file may be open in Excel")

def update_time_tracking_excel():
    """Create Excel with TWO SHEETS: Working Hours Summary + Detailed In/Out Times"""
    if not time_tracking_enabled:
        return
        
    try:
        attendance_dir = "attendance_records"
        if not os.path.exists(attendance_dir):
            os.makedirs(attendance_dir)
            
        date = datetime.now().strftime("%Y-%m-%d")
        excel_file = os.path.join(attendance_dir, f"time_tracking_{date}.xlsx")
        
        print(f"🔄 Creating TWO-SHEET Excel: {excel_file}")
        
        # Force refresh by deleting old file
        if os.path.exists(excel_file):
            try:
                os.remove(excel_file)
                print("🗑️ Deleted old Excel file")
            except:
                print("⚠️ Could not delete old Excel file - it may be open")
        
        # SHEET 1 DATA: Working Hours Summary
        summary_data = []
        
        # SHEET 2 DATA: Detailed In/Out Times (consecutive entries)
        detailed_data = []
        
        # Process each employee
        for name, entries in employee_time_entries.items():
            if not entries:
                continue
                
            # Sort entries by datetime
            sorted_entries = sorted(entries, key=lambda x: x['datetime'])
            
            # For SHEET 2: Add all consecutive entries (in/out pattern)
            total_work_minutes = 0
            sessions_count = 0
            
            for i, entry in enumerate(sorted_entries):
                # Add every entry to detailed log
                detailed_data.append({
                    'Name': name,
                    'Employee_ID': entry['emp_id'],
                    'Department': entry['dept'],
                    'Entry_Type': entry['entry_type'],
                    'Time': entry['datetime'].strftime('%H:%M:%S'),
                    'Full_DateTime': entry['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Date': entry['datetime'].strftime('%Y-%m-%d'),
                    'Screenshot_Path': entry['screenshot_path'] or 'N/A'
                })
                
                # Calculate work sessions for summary
                if entry['entry_type'] == 'CHECK_IN':
                    # Look for corresponding checkout
                    for j in range(i + 1, len(sorted_entries)):
                        if sorted_entries[j]['entry_type'] == 'CHECK_OUT':
                            checkout_entry = sorted_entries[j]
                            duration = checkout_entry['datetime'] - entry['datetime']
                            work_minutes = duration.total_seconds() / 60
                            total_work_minutes += work_minutes
                            sessions_count += 1
                            break
            
            # For SHEET 1: Summary data
            total_work_hours = total_work_minutes / 60
            current_status = 'Checked In' if employee_checked_in_status.get(name, False) else 'Checked Out'
            
            if sorted_entries:  # Only add if employee has entries
                summary_data.append({
                    'Name': name,
                    'Employee_ID': sorted_entries[0]['emp_id'],
                    'Department': sorted_entries[0]['dept'],
                    'Total_Work_Hours': round(total_work_hours, 2),
                    'Total_Work_Minutes': round(total_work_minutes, 1),
                    'Number_of_Sessions': sessions_count,
                    'Current_Status': current_status,
                    'First_Check_In': sorted_entries[0]['datetime'].strftime('%H:%M:%S'),
                    'Last_Activity': sorted_entries[-1]['datetime'].strftime('%H:%M:%S'),
                    'Date': date
                })
        
        # Create DataFrames for both sheets
        df_summary = pd.DataFrame(summary_data)
        df_detailed = pd.DataFrame(detailed_data)
        
        # Create Excel with TWO SHEETS
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # SHEET 1: Working Hours Summary
            if not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='Working_Hours_Summary', index=False)
            else:
                # Empty summary sheet with headers
                empty_summary = pd.DataFrame(columns=[
                    'Name', 'Employee_ID', 'Department', 'Total_Work_Hours', 'Total_Work_Minutes',
                    'Number_of_Sessions', 'Current_Status', 'First_Check_In', 'Last_Activity', 'Date'
                ])
                empty_summary.to_excel(writer, sheet_name='Working_Hours_Summary', index=False)
            
            # SHEET 2: Detailed In/Out Times (Consecutive)
            if not df_detailed.empty:
                df_detailed.to_excel(writer, sheet_name='Detailed_InOut_Log', index=False)
            else:
                # Empty detailed sheet with headers
                empty_detailed = pd.DataFrame(columns=[
                    'Name', 'Employee_ID', 'Department', 'Entry_Type', 'Time', 
                    'Full_DateTime', 'Date', 'Screenshot_Path'
                ])
                empty_detailed.to_excel(writer, sheet_name='Detailed_InOut_Log', index=False)
            
            # Auto-adjust column widths for both sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 25)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ TWO-SHEET Excel created successfully!")
        print(f"📊 Sheet 1 (Summary): {len(summary_data)} employees")
        print(f"📋 Sheet 2 (Detailed): {len(detailed_data)} entries")
        
        # Show summary of what was created
        if summary_data:
            print("\\n📈 WORKING HOURS SUMMARY:")
            for emp in summary_data:
                print(f"   {emp['Name']}: {emp['Total_Work_Hours']}h ({emp['Current_Status']})")
        
        if detailed_data:
            print("\\n📝 DETAILED LOG (Recent entries):")
            for entry in detailed_data[-5:]:  # Show last 5 entries
                print(f"   {entry['Name']}: {entry['Entry_Type']} at {entry['Time']}")
        
    except Exception as e:
        print(f"❌ Excel creation failed: {e}")
        import traceback
        traceback.print_exc()

def save_time_tracking_csv():
    """Enhanced CSV saving with better error handling and original format"""
    try:
        attendance_dir = "attendance_records"
        date = datetime.now().strftime("%Y-%m-%d")
        
        # Save in original yaseen format
        main_file = os.path.join(attendance_dir, f"daily_attendance_{date}.csv")
        sessions_file = os.path.join(attendance_dir, f"work_sessions_{date}.csv")
        
        # Main attendance file (original format)
        try:
            with open(main_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name', 'Employee_ID', 'Department', 'In_Time', 'Out_Time', 
                               'Duration_Minutes', 'Duration_Hours', 'Auto_Generated_Out', 'Date'])
                
                for name, entries in employee_time_entries.items():
                    sorted_entries = sorted(entries, key=lambda x: x['datetime'])
                    
                    for i, entry in enumerate(sorted_entries):
                        if entry['entry_type'] == 'CHECK_IN':
                            out_time = ""
                            duration_minutes = 0
                            duration_hours = 0
                            auto_generated = False
                            
                            # Look for corresponding checkout
                            for j in range(i + 1, len(sorted_entries)):
                                if sorted_entries[j]['entry_type'] == 'CHECK_OUT':
                                    checkout_entry = sorted_entries[j]
                                    out_time = checkout_entry['datetime'].strftime('%H:%M:%S')
                                    duration = checkout_entry['datetime'] - entry['datetime']
                                    duration_minutes = duration.total_seconds() / 60
                                    duration_hours = duration_minutes / 60
                                    break
                            
                            # If still checked in, show as current status
                            if not out_time and employee_checked_in_status.get(name, False):
                                out_time = "Currently Checked In"
                                auto_generated = True
                            
                            writer.writerow([
                                name, entry['emp_id'], entry['dept'],
                                entry['datetime'].strftime('%H:%M:%S'),
                                out_time,
                                round(duration_minutes, 1),
                                round(duration_hours, 2),
                                auto_generated,
                                entry['datetime'].strftime('%Y-%m-%d')
                            ])
            
            print(f"📊 Main CSV saved: {main_file}")
        except Exception as e:
            print(f"❌ Failed to save main CSV: {e}")
        
        # Work sessions file (detailed sessions)
        try:
            with open(sessions_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name', 'Employee_ID', 'Department', 'In_Time', 'Out_Time', 
                               'Duration_Minutes', 'Duration_Hours', 'Auto_Generated_Out', 'Date'])
                
                for name, sessions in employee_work_sessions.items():
                    for session in sessions:
                        writer.writerow([
                            session['name'], session['emp_id'], session['dept'],
                            session['in_time'].strftime('%H:%M:%S'),
                            session['out_time'].strftime('%H:%M:%S'),
                            round(session['duration_minutes'], 1),
                            round(session['duration_minutes'] / 60, 2),
                            session.get('auto_generated_out', False),
                            session['in_time'].strftime('%Y-%m-%d')
                        ])
            
            print(f"📊 Sessions CSV saved: {sessions_file}")
        except Exception as e:
            print(f"❌ Failed to save sessions CSV: {e}")
            
    except Exception as e:
        print(f"❌ Failed to save CSV backup: {e}")

def get_employee_total_work_time(name):
    """Get total work time for an employee today"""
    if name not in employee_work_sessions:
        return 0, 0  # minutes, hours
    
    total_minutes = sum(session['duration_minutes'] for session in employee_work_sessions[name])
    total_hours = total_minutes / 60
    
    return total_minutes, total_hours

def display_time_tracking_summary():
    """Display comprehensive time tracking summary"""
    print(f"\n📊 TIME TRACKING SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    if not employee_work_sessions:
        print("No work sessions recorded yet today.")
        return
    
    for name in employee_work_sessions:
        total_minutes, total_hours = get_employee_total_work_time(name)
        sessions_count = len(employee_work_sessions[name])
        entries_count = len(employee_time_entries[name])
        
        print(f"\n👤 {name}:")
        print(f"   📝 Total entries today: {entries_count}")
        print(f"   ⏱️  Work sessions: {sessions_count}")
        print(f"   🕐 Total work time: {total_hours:.2f} hours ({total_minutes:.1f} minutes)")
        
        if name in employee_work_sessions:
            print(f"   📋 Session details:")
            for i, session in enumerate(employee_work_sessions[name], 1):
                in_time = session['in_time'].strftime('%H:%M:%S')
                out_time = session['out_time'].strftime('%H:%M:%S')
                duration = session['duration_minutes']
                print(f"      {i}. {in_time} → {out_time} ({duration:.1f} min)")

def display_cctv_statistics(frame):
    """Display real-time CCTV statistics with time tracking info on frame"""
    # Background overlay for better readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Display attendance count
    cv2.putText(frame, f"Today's Attendance: {len(today_attendance)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display enrolled employees
    cv2.putText(frame, f"Enrolled Employees: {len(known_face_encodings)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display time tracking info
    total_work_sessions = sum(len(sessions) for sessions in employee_work_sessions.values())
    cv2.putText(frame, f"Work Sessions: {total_work_sessions}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display current time
    current_time = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display total active employees with work time today
    active_employees = len([name for name in employee_work_sessions if employee_work_sessions[name]])
    if active_employees > 0:
        cv2.putText(frame, f"Active Workers: {active_employees}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display CCTV status
    cv2.putText(frame, "CCTV LIVE + TIME TRACKING", 
                (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
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
    
    # ADD YOUR 4 FRIENDS HERE - Update with their actual names, IDs, and photo filenames:
    employees_loaded += add_employee('zayd', '1001', 'AI&DS', 'zayd.jpg')
    employees_loaded += add_employee('darun', '1002', 'AI&DS', 'darun.jpg')
    employees_loaded += add_employee('iyaaa', '1003', 'AI&DS', 'iyaaa.jpg')
    employees_loaded += add_employee('lokesh', '1004', 'AI&DS', 'lokesh.jpg')
    
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
    
    # Setup Dual Camera System
    print("\n" + "="*60)
    print("🎥🎥 DUAL CAMERA SYSTEM SETUP")
    print("="*60)
    
    # Configure cameras
    CCTVAttendanceConfig.CHECKIN_CAMERA_URL = CCTVAttendanceConfig.RTSP_URL
    
    # Test both cameras
    cameras, camera_mode = test_dual_camera_connection()
    
    if camera_mode == "FAILED":
        print("❌ Cannot establish any camera connection. Exiting...")
        return
    elif camera_mode == "DUAL":
        print("🎉 DUAL CAMERA SYSTEM READY!")
        print("   📥 CHECK_IN: CCTV Camera")
        print("   📤 CHECK_OUT: Mobile IP WebCam")
    else:
        print(f"⚠️ Running in {camera_mode} mode")
    
    # Set primary camera for display
    if cameras['checkin'][0] is not None:
        cap, video_props = cameras['checkin']
        camera_name = "CHECK_IN Camera (CCTV)"
    else:
        cap, video_props = cameras['checkout']
        camera_name = "CHECK_OUT Camera (IP WebCam)"
        
    width, height, fps = video_props
    print(f"✅ Primary display: {camera_name} - {width}x{height} @ {fps} FPS")
    
    print("\n🎥 Starting Dual Camera Attendance System...")
    print("📋 IMPORTANT INSTRUCTIONS:")
    print(f"   � CHECK_IN: Stand in front of CCTV camera")
    print(f"   📤 CHECK_OUT: Stand in front of mobile phone camera")
    print(f"   🔄 No auto-checkout - Use mobile camera to checkout!")
    print(f"   � Daily auto-checkout: 6 PM")
    print("\n🎮 ESSENTIAL CONTROLS:")
    if GUI_AVAILABLE:
        print("   'O' - Manual checkout menu")
        print("   'C' - Show checked-in employees")
        print("   'A' - Checkout all employees")
        print("   'Q' - Quit system")
        print("   'H' - Full help menu")
        print("\n📊 REPORTS & DATA:")
        print("   'S' - Show attendance summary")
        print("   'T' - Show time tracking summary")
        print("   'E' - Export Excel report")
    else:
        print("   - Press Ctrl+C to stop")
    
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0
    last_connection_attempt = 0
    
    print("🟢 CCTV attendance system is now LIVE and monitoring...")
    
    try:
        while True:
            # DUAL CAMERA PROCESSING
            frames = {}
            all_results = []
            
            # Ensure camera_mode is defined (fallback)
            if 'camera_mode' not in locals():
                camera_mode = "SINGLE_LEGACY"
                cameras = {'checkin': (cap, video_props), 'checkout': (None, (0, 0, 0))}
            
            if camera_mode in ["DUAL", "SINGLE_CHECKIN", "SINGLE_LEGACY"]:
                # Process CHECK_IN camera (primary display)
                checkin_cap = cameras['checkin'][0]
                if checkin_cap is not None:
                    ret_in, frame_in = checkin_cap.read()
                    if ret_in and frame_in is not None:
                        frames['checkin'] = frame_in
                        frame = frame_in  # Primary display frame
                        
                        # Process CHECK_IN faces
                        if frame_count % CCTVAttendanceConfig.PROCESS_EVERY_N_FRAMES == 0:
                            results_in = recognize_faces_cctv(frame_in, entry_type="CHECK_IN")
                            all_results.extend(results_in)
                    else:
                        # Handle CHECK_IN camera connection loss
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(frame, "CHECK_IN CAMERA DISCONNECTED", (50, height//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if camera_mode == "DUAL":
                # Process CHECK_OUT camera (background processing)
                checkout_cap = cameras['checkout'][0]
                if checkout_cap is not None:
                    ret_out, frame_out = checkout_cap.read()
                    if ret_out and frame_out is not None:
                        frames['checkout'] = frame_out
                        
                        # Process CHECK_OUT faces
                        if frame_count % CCTVAttendanceConfig.PROCESS_EVERY_N_FRAMES == 0:
                            results_out = recognize_faces_cctv(frame_out, entry_type="CHECK_OUT")
                            all_results.extend(results_out)
                            
                            # Print checkout detections to console for debugging
                            if results_out:
                                print(f"🔍 CHECK_OUT Camera: {len(results_out)} detections")
                                for result in results_out:
                                    if result.get("name", "unknown") != "unknown":
                                        print(f"  📤 {result['name']} detected at checkout camera")
            
            elif camera_mode == "SINGLE_CHECKOUT":
                # Only CHECK_OUT camera available
                checkout_cap = cameras['checkout'][0]
                if checkout_cap is not None:
                    ret_out, frame_out = checkout_cap.read()
                    if ret_out and frame_out is not None:
                        frames['checkout'] = frame_out
                        frame = frame_out  # Use as primary display
                        
                        # Process CHECK_OUT faces
                        if frame_count % CCTVAttendanceConfig.PROCESS_EVERY_N_FRAMES == 0:
                            results_out = recognize_faces_cctv(frame_out, entry_type="CHECK_OUT")
                            all_results.extend(results_out)
            
            elif camera_mode == "SINGLE_LEGACY":
                # Legacy single camera mode (fallback)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Process single camera with CHECK_IN logic
                    if frame_count % CCTVAttendanceConfig.PROCESS_EVERY_N_FRAMES == 0:
                        results_legacy = recognize_faces_cctv(frame, entry_type="CHECK_IN")
                        all_results.extend(results_legacy)
                else:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA DISCONNECTED", (50, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Handle connection loss for primary display
            if 'frame' not in locals() or frame is None:
                current_time = time.time()
                if CCTVAttendanceConfig.AUTO_RECONNECT and \
                   (current_time - last_connection_attempt) > CCTVAttendanceConfig.CONNECTION_RETRY_INTERVAL:
                    
                    print("📡 Camera connection lost. Attempting to reconnect...")
                    last_connection_attempt = current_time
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(frame, "RECONNECTING...", (width//2-100, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    continue
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                
                # Auto-checkout DISABLED for dual camera mode
                # auto_checkout_after_1_minute()  # DISABLED - Use mobile camera for checkout
            
            # Resize frame to standard size for consistent processing
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Use results from dual camera processing
            results = all_results
            
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
                                # Find employee details and show work time
                                for _, emp_name, emp_id, dept in known_face_encodings:
                                    if emp_name == name:
                                        total_minutes, total_hours = get_employee_total_work_time(name)
                                        print(f"  ✓ {name} (ID: {emp_id}, Dept: {dept}) - Work time: {total_hours:.2f}h")
                                        break
                        else:
                            print("  No one has attended yet today.")
                        print()
                    elif key == ord('t'):
                        # Show time tracking summary
                        display_time_tracking_summary()
                    elif key == ord('w'):
                        # Show work sessions details
                        print(f"\n🕒 WORK SESSIONS DETAILS - {time.strftime('%Y-%m-%d')}")
                        print("=" * 50)
                        if employee_work_sessions:
                            for name, sessions in employee_work_sessions.items():
                                print(f"\n👤 {name}:")
                                for i, session in enumerate(sessions, 1):
                                    in_time = session['in_time'].strftime('%H:%M:%S')
                                    out_time = session['out_time'].strftime('%H:%M:%S')
                                    duration = session['duration_minutes']
                                    print(f"  {i}. {in_time} → {out_time} ({duration:.1f} min)")
                        else:
                            print("No work sessions recorded yet.")
                        print()
                    elif key == ord('e'):
                        # Export Excel report manually
                        update_time_tracking_excel()
                        print("📊 Excel report exported manually!")
                    elif key == ord('o'):
                        # Manual checkout - show currently checked in employees
                        checked_in = [name for name, status in employee_checked_in_status.items() if status]
                        if checked_in:
                            print(f"\n🔴 MANUAL CHECKOUT AVAILABLE:")
                            for i, name in enumerate(checked_in, 1):
                                print(f"  {i}. {name}")
                            print(f"\n👆 Press number key (1-{len(checked_in)}) to checkout employee")
                            print("   Or press 'A' to checkout ALL employees")
                        else:
                            print("\n⚠️ No employees currently checked in")
                    elif key >= ord('1') and key <= ord('9'):
                        # Checkout specific employee by number
                        checkout_index = int(chr(key)) - 1
                        checked_in = [name for name, status in employee_checked_in_status.items() if status]
                        if 0 <= checkout_index < len(checked_in):
                            employee_name = checked_in[checkout_index]
                            manual_checkout_employee(employee_name)
                        else:
                            print(f"❌ Invalid selection. Available: 1-{len(checked_in)}")
                    elif key == ord('a'):
                        # Auto checkout all employees
                        checked_in = [name for name, status in employee_checked_in_status.items() if status]
                        if checked_in:
                            print(f"\n🔴 CHECKOUT ALL EMPLOYEES ({len(checked_in)} employees):")
                            for name in checked_in:
                                manual_checkout_employee(name)
                            print("✅ All employees checked out!")
                        else:
                            print("\n⚠️ No employees currently checked in")
                    elif key == ord('c'):
                        # Show currently checked-in employees
                        display_checked_in_employees()
                    elif key == ord('r'):
                        # Reset all tracking data
                        today_attendance.clear()
                        last_recognition_time.clear()
                        consecutive_detections.clear()
                        employee_time_entries.clear()
                        employee_work_sessions.clear()
                        print("🔄 All attendance and time tracking data has been reset!")
                    elif key == ord('c'):
                        # Manual screenshot capture
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        screenshot_name = f"manual_capture_{timestamp}.jpg"
                        cv2.imwrite(screenshot_name, frame)
                        print(f"📸 Manual screenshot saved: {screenshot_name}")
                    elif key == ord('h'):
                        print("\n📋 ENHANCED HELP - Keyboard Controls:")
                        print("  'q' - Quit the system")
                        print("  's' - Show today's attendance summary with work hours")
                        print("  't' - Show detailed time tracking summary")
                        print("  'w' - Show work sessions details")
                        print("  'e' - Export Excel report manually")
                        print("  'r' - Reset all attendance and time tracking data")
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
        
        # Final enhanced summary with time tracking
        print(f"\n📊 FINAL ENHANCED CCTV SESSION SUMMARY:")
        print(f"Total employees enrolled: {len(known_face_encodings)}")
        print(f"Employees present today: {len(today_attendance)}")
        print(f"Total frames processed: {frame_count}")
        
        # Time tracking summary
        total_work_sessions = sum(len(sessions) for sessions in employee_work_sessions.values())
        print(f"Total work sessions: {total_work_sessions}")
        
        if today_attendance:
            print("📋 Today's Attendees with Work Time:")
            for name in sorted(today_attendance):
                for _, emp_name, emp_id, dept in known_face_encodings:
                    if emp_name == name:
                        total_minutes, total_hours = get_employee_total_work_time(name)
                        sessions_count = len(employee_work_sessions.get(name, []))
                        print(f"  ✓ {name} (ID: {emp_id}, Dept: {dept}) - {total_hours:.2f}h in {sessions_count} sessions")
                        break
        
        # Generate final Excel report
        if employee_work_sessions or employee_time_entries:
            update_time_tracking_excel()
            print("📊 Final Excel time tracking report generated")
        
        print("✅ Enhanced CCTV attendance & time tracking system stopped")

if __name__ == "__main__":
    cctv_attendance_system()
