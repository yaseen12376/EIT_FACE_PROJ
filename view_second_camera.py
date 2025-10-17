#!/usr/bin/env python3
"""
Continuous viewer for second CCTV camera - Full screen view
Shows the camera feed until you press 'q' to quit
"""

import cv2
import time
from datetime import datetime

def view_second_camera_continuous():
    """Display second CCTV camera feed continuously"""
    
    print("=" * 70)
    print("🎥 SECOND CCTV CAMERA - CONTINUOUS VIEWER")
    print("=" * 70)
    
    # Second CCTV camera configuration
    CHECKOUT_CAMERA_URL = "rtsp://admin:admin@777@192.168.0.135:554/cam/realmonitor?channel=1&subtype=0"
    
    print(f"\n📹 Camera: 192.168.0.135 (Channel 1)")
    print(f"🔗 Connecting to CHECK-OUT camera...")
    print(f"⏳ Please wait 5-10 seconds for connection...\n")
    
    try:
        # Connect to camera
        cap = cv2.VideoCapture(CHECKOUT_CAMERA_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Wait for connection
        time.sleep(3)
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera!")
            print("\n🔍 Troubleshooting:")
            print("  1. Check IP: 192.168.0.135")
            print("  2. Check credentials: admin / admin@777")
            print("  3. Check channel: 1")
            print("  4. Verify camera is online: ping 192.168.0.135")
            return
        
        print("✅ Connected successfully!\n")
        
        # Get initial frame to check resolution
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"📺 Camera Resolution: {width}x{height}")
            print(f"🎬 Starting live feed...\n")
            print("=" * 70)
            print("CONTROLS:")
            print("  Q or ESC  - Quit")
            print("  F         - Toggle fullscreen")
            print("  S         - Save screenshot")
            print("=" * 70)
            print()
        
        # Create resizable window
        window_name = '🎥 CHECK-OUT Camera (Channel 1) - Press Q to quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set window to large size (adjust based on your screen)
        cv2.resizeWindow(window_name, 1600, 900)
        
        start_time = time.time()
        frame_count = 0
        fullscreen = False
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\n⚠️ Lost connection to camera!")
                print("Attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(CHECKOUT_CAMERA_URL)
                continue
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add overlays
            # Title
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
            cv2.putText(frame, "CHECK-OUT CAMERA - CHANNEL 1", 
                       (20, 50), cv2.FONT_HERSHEY_DUPLEX, 
                       1.5, (0, 255, 0), 3)
            
            # Camera info
            cv2.putText(frame, f"IP: 192.168.0.135:554", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # FPS counter (top right)
            fps_text = f"FPS: {current_fps:.1f}"
            (text_width, text_height), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, fps_text, 
                       (frame.shape[1] - text_width - 20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 255), 2)
            
            # Time stamp
            cv2.putText(frame, current_time, 
                       (frame.shape[1] - 300, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Bottom bar with controls
            cv2.rectangle(frame, (0, frame.shape[0] - 50), 
                         (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, "Q=Quit | F=Fullscreen | S=Screenshot", 
                       (20, frame.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)
            
            # Resolution
            cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]}", 
                       (frame.shape[1] - 150, frame.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC
                print("\n🛑 Stopping camera viewer...")
                break
            
            elif key == ord('f') or key == ord('F'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                    print("📺 Fullscreen mode enabled")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
                    print("📺 Normal view mode")
            
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f"checkout_camera_test_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 70)
        print("📊 VIEWING SESSION STATISTICS")
        print("=" * 70)
        print(f"⏱️  Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"🎬 Frames processed: {frame_count}")
        print(f"📊 Average FPS: {avg_fps:.1f}")
        print(f"📸 Screenshots taken: {screenshot_count}")
        print("=" * 70)
        print("\n✅ Camera viewer closed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n🎥 Starting Second CCTV Camera Viewer...")
    print("   This will show the camera feed continuously")
    print("   until you press 'Q' to quit.\n")
    
    view_second_camera_continuous()
    
    print("\n👋 Viewer closed. Have a great day!")
