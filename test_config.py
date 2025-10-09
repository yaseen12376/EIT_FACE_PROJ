#!/usr/bin/env python3
"""
Quick test to verify dual camera system settings
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cctv_attendance_system import CCTVAttendanceConfig

def test_dual_camera_config():
    """Test dual camera configuration"""
    print("🔧 DUAL CAMERA CONFIGURATION TEST")
    print("=" * 50)
    
    # Test key settings
    print(f"✅ Dual Camera Mode: {CCTVAttendanceConfig.DUAL_CAMERA_MODE}")
    print(f"✅ Auto-Checkout Enabled: {CCTVAttendanceConfig.ENABLE_AUTO_CHECKOUT}")
    print(f"✅ Recognition Cooldown: {CCTVAttendanceConfig.RECOGNITION_COOLDOWN} seconds")
    print(f"✅ Auto-Checkout Minutes: {CCTVAttendanceConfig.AUTO_CHECKOUT_MINUTES}")
    
    print(f"\n📷 CHECK-IN Camera: {CCTVAttendanceConfig.CHECKIN_CAMERA_URL}")
    print(f"📱 CHECK-OUT Camera: {CCTVAttendanceConfig.CHECKOUT_CAMERA_URL}")
    
    # Verify correct settings for dual camera mode
    print("\n🎯 DUAL CAMERA VALIDATION:")
    
    if CCTVAttendanceConfig.DUAL_CAMERA_MODE:
        print("✅ Dual camera mode is ENABLED")
    else:
        print("❌ Dual camera mode is DISABLED")
    
    if not CCTVAttendanceConfig.ENABLE_AUTO_CHECKOUT:
        print("✅ Auto-checkout is DISABLED (correct for dual camera)")
    else:
        print("❌ Auto-checkout is ENABLED (should be disabled)")
    
    if CCTVAttendanceConfig.RECOGNITION_COOLDOWN == 0.0:
        print("✅ Recognition cooldown is DISABLED (correct for immediate checkout)")
    else:
        print(f"❌ Recognition cooldown is {CCTVAttendanceConfig.RECOGNITION_COOLDOWN}s (should be 0.0)")
    
    if CCTVAttendanceConfig.AUTO_CHECKOUT_MINUTES == 0:
        print("✅ Auto-checkout timer is DISABLED (correct)")
    else:
        print(f"❌ Auto-checkout timer is {CCTVAttendanceConfig.AUTO_CHECKOUT_MINUTES} minutes")
    
    print("\n🚀 EXPECTED BEHAVIOR:")
    print("  1. Employee stands at CCTV → CHECK-IN logged immediately")
    print("  2. Employee can work for any duration (no auto-checkout)")  
    print("  3. Employee goes to mobile camera → CHECK-OUT logged immediately")
    print("  4. Employee can CHECK-IN again immediately (no cooldown)")
    
    print("\n✅ Configuration test complete!")

if __name__ == "__main__":
    test_dual_camera_config()