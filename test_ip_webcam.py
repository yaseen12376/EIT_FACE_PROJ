#!/usr/bin/env python3
"""
IP WebCam Test Script
Test connection to mobile phone IP WebCam before running main system
"""

import cv2
import time

def test_ip_webcam():
    """Test IP WebCam connection"""
    
    # Your IP WebCam URL
    ip_webcam_url = "http://192.168.0.180:8080/video"
    
    print("🔗 Testing IP WebCam Connection...")
    print(f"📱 URL: {ip_webcam_url}")
    
    # Test connection
    cap = cv2.VideoCapture(ip_webcam_url)
    
    if cap.isOpened():
        print("✅ IP WebCam opened successfully!")
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"✅ Frame received: {width}x{height}")
            print("🎉 IP WebCam is working properly!")
            
            # Show frame for 3 seconds
            cv2.imshow('IP WebCam Test', frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            
        else:
            print("❌ Cannot read frames from IP WebCam")
            print("💡 Check if IP WebCam server is running on your phone")
        
        cap.release()
        
    else:
        print("❌ Cannot connect to IP WebCam")
        print("💡 Troubleshooting steps:")
        print("   1. Make sure IP WebCam app is running on your phone")
        print("   2. Check if phone and computer are on same WiFi network")
        print("   3. Verify the IP address: http://192.168.0.180:8080")
        print("   4. Try opening the URL in a web browser first")

if __name__ == "__main__":
    test_ip_webcam()