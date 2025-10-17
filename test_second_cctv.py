#!/usr/bin/env python3
"""
Test connection to the second CCTV camera (CHECK_OUT camera)
"""

import cv2
import time

def test_second_cctv_camera():
    """Test connection to the second CCTV camera"""
    
    print("🎥 Testing Second CCTV Camera Connection")
    print("=" * 60)
    
    # Second CCTV camera configuration
    CHECKOUT_CAMERA_URL = "rtsp://admin:admin@777@192.168.0.135:554/cam/realmonitor?channel=1&subtype=0"
    
    print(f"📹 Connecting to: 192.168.0.135 (Channel 1)")
    print(f"🔐 Credentials: admin / admin@777")
    print(f"🔗 Full URL: {CHECKOUT_CAMERA_URL}")
    print("\n⏳ Attempting connection (this may take 10-15 seconds)...\n")
    
    try:
        # Try to connect to the camera
        cap = cv2.VideoCapture(CHECKOUT_CAMERA_URL)
        
        # Set timeout
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Wait a bit for connection
        time.sleep(3)
        
        if not cap.isOpened():
            print("❌ Failed to connect to second CCTV camera")
            print("\n🔍 Troubleshooting:")
            print("  1. Check if camera IP is correct: 192.168.0.135")
            print("  2. Verify credentials: admin / admin@777")
            print("  3. Ensure channel 1 is active")
            print("  4. Check network connection")
            print("  5. Try accessing via browser: rtsp://192.168.0.135:554")
            return False
        
        print("✅ Successfully connected to second CCTV camera!")
        
        # Try to read a frame
        print("📸 Reading test frame...")
        ret, frame = cap.read()
        
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"✅ Frame captured successfully!")
            print(f"   Resolution: {width}x{height}")
            print(f"   Frame shape: {frame.shape}")
            
            # Display the frame continuously until user presses 'q'
            print("\n🖼️ Displaying camera feed continuously...")
            print("   📺 Press 'q' to quit")
            print("   📺 Press 'f' to toggle fullscreen")
            print("   📺 Window will resize to show full camera view")
            
            # Create a named window that can be resized
            window_name = 'Second CCTV - CHECK_OUT Camera Test (Press Q to quit)'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Make the window larger (adjust these values as needed)
            cv2.resizeWindow(window_name, 1280, 720)
            
            start_time = time.time()
            frame_count = 0
            fullscreen = False
            
            print("\n⏺️ Camera feed is now live - showing full view...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Lost connection to camera")
                    break
                
                frame_count += 1
                
                # Add overlay information
                cv2.putText(frame, "CHECK-OUT Camera (Channel 1)", 
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                          1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"IP: 192.168.0.135", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Press 'Q' to quit | 'F' for fullscreen", 
                          (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (255, 255, 0), 2)
                
                # Show FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                          (frame.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, (0, 255, 255), 2)
                
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n🛑 Stopping camera feed...")
                    break
                elif key == ord('f') or key == ord('F'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("📺 Fullscreen enabled")
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print("📺 Normal view")
            
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 Test Statistics:")
            print(f"   Frames captured: {frame_count}")
            print(f"   Duration: {elapsed:.1f} seconds")
            print(f"   Average FPS: {fps:.1f}")
            
            print("\n✅ Second CCTV camera is working perfectly!")
            print("🎯 Ready to use for CHECK-OUT detection!")
            
        else:
            print("❌ Could not read frame from camera")
            print("   Camera connected but not streaming properly")
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error testing second CCTV camera: {e}")
        return False

if __name__ == "__main__":
    success = test_second_cctv_camera()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SECOND CCTV CAMERA TEST PASSED!")
        print("=" * 60)
        print("\n📋 Next Steps:")
        print("  1. Run the main system: python cctv_attendance_system.py")
        print("  2. Test CHECK-IN at first camera (192.168.0.109)")
        print("  3. Test CHECK-OUT at second camera (192.168.0.135)")
        print("  4. Verify dual camera attendance tracking works!")
    else:
        print("\n" + "=" * 60)
        print("❌ SECOND CCTV CAMERA TEST FAILED!")
        print("=" * 60)
        print("\n🔧 Please check:")
        print("  - Camera is powered on and connected to network")
        print("  - IP address 192.168.0.135 is correct")
        print("  - Credentials admin/admin@777 are correct")
        print("  - Channel 1 is the correct channel")
        print("  - Firewall allows RTSP connections")
