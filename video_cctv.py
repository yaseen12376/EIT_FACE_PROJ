import cv2
import time
import os

# RTSP URL with authentication - Channel 8 (working URL only)
url = "rtsp://admin:AK@MrA!4501$uf@192.168.0.109:554/cam/realmonitor?channel=8&subtype=0"

# No alternative URLs needed - using only the working URL
alternative_urls = []

# Check if GUI is available
GUI_AVAILABLE = True
try:
    # Force GUI availability for Windows
    import sys
    if sys.platform.startswith('win'):
        GUI_AVAILABLE = True  # Assume GUI is available on Windows
        print("GUI enabled for Windows platform")
    else:
        # For other platforms, test GUI
        try:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test") 
            GUI_AVAILABLE = True
        except:
            GUI_AVAILABLE = False
except Exception:
    GUI_AVAILABLE = True  # Default to GUI available

print(f"GUI Status: {'Available' if GUI_AVAILABLE else 'Not Available'}")

def test_rtsp_connection():
    """Test RTSP connection with different URL formats"""
    print("Testing RTSP connection...")
    
    # Try main URL first
    print(f"Trying main URL...")
    try:
        cap = cv2.VideoCapture(url)
        
        # Set buffer size to prevent lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("✓ Main URL connected successfully!")
                # Don't reset frame position for live streams
                return cap
            else:
                print("✗ Main URL opened but can't read frames")
                cap.release()
        else:
            print("✗ Main URL failed to open")
            cap.release()
    except Exception as e:
        print(f"✗ Main URL error: {e}")
        
    # Try alternative URLs
    for i, alt_url in enumerate(alternative_urls):
        print(f"Trying alternative {i+1}...")
        try:
            cap = cv2.VideoCapture(alt_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✓ Alternative URL {i+1} connected successfully!")
                    return cap
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"✗ Alternative {i+1} error: {e}")
    
    print("✗ All URLs failed. Please check:")
    print("  1. Network connection")
    print("  2. Camera IP address (192.168.0.109)")
    print("  3. Username/password (admin:AK@MrA!4501$uf)")
    print("  4. RTSP port (554)")
    print("  5. Camera streaming settings")
    print("  6. Firewall blocking RTSP traffic")
    return None

def main():
    global GUI_AVAILABLE  # Make sure we can access the global variable
    
    # Test connection first
    cap = test_rtsp_connection()
    
    if cap is None:
        print("Failed to connect to RTSP stream. Exiting...")
        return
    
    # Get video properties for display
    try:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # Default to 25 if can't get FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate dimensions
        if width <= 0 or height <= 0:
            print("Warning: Invalid video dimensions, using defaults")
            width, height = 640, 480
            
        print(f"\nVideo properties: {width}x{height} @ {fps} FPS")
        
    except Exception as e:
        print(f"Error getting video properties: {e}")
        width, height, fps = 640, 480, 25
    
    print("\nStarting video stream...")
    if GUI_AVAILABLE:
        print("Press 'q' to quit")
        print("Displaying live stream in window...")
    else:
        print("GUI not available. Stream running in background.")
        print("Press Ctrl+C to stop.")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame. Stream may have ended or connection lost.")
                # Try to reconnect
                print("Attempting to reconnect...")
                cap.release()
                cap = test_rtsp_connection()
                if cap is None:
                    break
                continue
            
            frame_count += 1
            
            # Ensure frame is the right size if needed
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Add frame counter and timestamp overlay
            timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, timestamp_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "LIVE STREAM", (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Handle display
            if GUI_AVAILABLE:
                try:
                    cv2.imshow("CCTV Channel 8 Live Stream", frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        # Save single frame
                        screenshot_name = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_name, frame)
                        print(f"Screenshot saved: {screenshot_name}")
                        
                except cv2.error as e:
                    print(f"Display error: {e}")
                    print("Trying alternative display method...")
                    # Try with different window flags
                    try:
                        cv2.namedWindow("CCTV Stream", cv2.WINDOW_AUTOSIZE)
                        cv2.imshow("CCTV Stream", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            screenshot_name = f"screenshot_{int(time.time())}.jpg"
                            cv2.imwrite(screenshot_name, frame)
                            print(f"Screenshot saved: {screenshot_name}")
                    except Exception as e2:
                        print(f"Alternative display also failed: {e2}")
                        print("Continuing stream without display...")
                        GUI_AVAILABLE = False
            else:
                # Non-GUI mode - just add a small delay and show we're streaming
                time.sleep(0.01)
            
            # Show streaming info every 10 seconds
            if frame_count % (fps * 10) == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Streaming: {elapsed:.1f}s - {frame_count} frames - Current FPS: {current_fps:.1f}")
                        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        # Clean up
        print("Cleaning up...")
        if cap:
            cap.release()
        if GUI_AVAILABLE:
            try:
                cv2.destroyAllWindows()
            except:
                pass  # Ignore cleanup errors
        
        elapsed = time.time() - start_time
        print(f"Streaming completed!")
        print(f"- Duration: {elapsed:.1f} seconds")
        print(f"- Frames displayed: {frame_count}")
        if elapsed > 0:
            print(f"- Average FPS: {frame_count/elapsed:.1f}")

if __name__ == "__main__":
    main()