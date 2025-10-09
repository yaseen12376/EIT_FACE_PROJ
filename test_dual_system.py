#!/usr/bin/env python3
"""
Quick system test to check if the dual camera modifications work
"""

print("🧪 TESTING DUAL CAMERA SYSTEM...")

try:
    # Test imports
    import cv2
    import numpy as np
    print("✅ OpenCV and numpy imported")
    
    # Test configuration
    import sys
    sys.path.append('.')
    
    # Import the main system
    from cctv_attendance_system import CCTVAttendanceConfig
    print(f"✅ Configuration loaded")
    print(f"   DUAL_CAMERA_MODE: {CCTVAttendanceConfig.DUAL_CAMERA_MODE}")
    print(f"   ENABLE_AUTO_CHECKOUT: {CCTVAttendanceConfig.ENABLE_AUTO_CHECKOUT}")
    print(f"   CHECKOUT_CAMERA_URL: {CCTVAttendanceConfig.CHECKOUT_CAMERA_URL}")
    
    # Test function definition
    from cctv_attendance_system import recognize_faces_cctv
    print("✅ recognize_faces_cctv function imported")
    
    # Test with dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test both entry types
    result_checkin = recognize_faces_cctv(dummy_frame, entry_type="CHECK_IN")
    result_checkout = recognize_faces_cctv(dummy_frame, entry_type="CHECK_OUT")
    
    print("✅ Function calls with entry_type work!")
    print(f"   CHECK_IN results: {len(result_checkin) if result_checkin else 0}")
    print(f"   CHECK_OUT results: {len(result_checkout) if result_checkout else 0}")
    
    print("🎉 Dual camera system test PASSED!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()