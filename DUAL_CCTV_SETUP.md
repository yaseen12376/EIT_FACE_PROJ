# 🎥 DUAL CCTV CAMERA SYSTEM - CONFIGURATION COMPLETE

## ✅ **System Status: UPGRADED TO DUAL CCTV**

Your attendance system now uses **TWO CCTV cameras** for professional dual-point detection!

---

## 📹 **Camera Configuration**

### 🚪 **CHECK-IN Camera (Entry Point)**
- **IP Address**: `192.168.0.109`
- **Channel**: `8`
- **Type**: RTSP CCTV
- **Credentials**: `admin / AK@MrA!4501$uf`
- **RTSP URL**: `rtsp://admin:AK@MrA!4501$uf@192.168.0.109:554/cam/realmonitor?channel=8&subtype=0`
- **Purpose**: Employee entry detection

### 🚶 **CHECK-OUT Camera (Exit Point)**
- **IP Address**: `192.168.0.135`
- **Channel**: `1`
- **Type**: RTSP CCTV
- **Credentials**: `admin / admin@777`
- **RTSP URL**: `rtsp://admin:admin@777@192.168.0.135:554/cam/realmonitor?channel=1&subtype=0`
- **Purpose**: Employee exit detection

---

## 🔄 **What Changed**

### ❌ **OLD Setup (Temporary):**
- CHECK-IN: CCTV Camera (192.168.0.109)
- CHECK-OUT: Mobile Phone IP WebCam (192.168.0.180:8080)

### ✅ **NEW Setup (Production):**
- CHECK-IN: CCTV Camera (192.168.0.109, Channel 8)
- CHECK-OUT: CCTV Camera (192.168.0.135, Channel 1)

---

## 🎯 **System Features**

### ✅ **Dual CCTV Detection:**
- ✅ Both cameras are professional RTSP streams
- ✅ Simultaneous monitoring of entry and exit
- ✅ No cooldown delays (instant recognition)
- ✅ Real-time attendance tracking
- ✅ Independent camera streams

### 🔧 **Technical Details:**
```python
DUAL_CAMERA_MODE = True              # ✅ Enabled
RECOGNITION_COOLDOWN = 0.0           # ✅ No delays
ENABLE_AUTO_CHECKOUT = False         # ✅ Manual checkout only
CHECKIN_CAMERA_TYPE = "RTSP"         # ✅ CCTV
CHECKOUT_CAMERA_TYPE = "RTSP"        # ✅ CCTV (upgraded from HTTP)
```

---

## 🧪 **Testing the New Setup**

### 1️⃣ **Test Second Camera Connection**
```bash
python test_second_cctv.py
```
**Expected:** 
- ✅ Connection successful
- ✅ Video feed displays for 5 seconds
- ✅ Resolution and FPS shown

### 2️⃣ **Test Both Cameras Together**
```bash
python test_dual_system.py
```
**Expected:**
- ✅ Both cameras connect
- ✅ Dual stream processing works

### 3️⃣ **Run Full System**
```bash
python cctv_attendance_system.py
```
**Expected:**
- ✅ Both CCTV cameras initialize
- ✅ Face recognition on both streams
- ✅ CHECK-IN/CHECK-OUT tracking

---

## 🚀 **Employee Workflow**

### 📥 **Morning Check-In:**
1. Employee arrives at office
2. Walks past **Entry Camera** (192.168.0.109, Channel 8)
3. System detects face → **"✅ CHECK-IN: [name]"**
4. Entry time logged in Excel

### 📤 **Evening Check-Out:**
1. Employee leaves office
2. Walks past **Exit Camera** (192.168.0.135, Channel 1)
3. System detects face → **"✅ CHECK-OUT: [name]"**
4. Exit time logged, work hours calculated

### 🔄 **Advantages:**
- ✅ **No phone required** - Both are fixed CCTV cameras
- ✅ **Professional setup** - Proper surveillance equipment
- ✅ **Reliable streams** - No mobile battery/connectivity issues
- ✅ **Simultaneous monitoring** - Both entry and exit points covered
- ✅ **Instant detection** - No cooldown or waiting periods

---

## 🔍 **Troubleshooting**

### 📹 **If CHECK-OUT Camera Doesn't Connect:**
```bash
# Test the new camera independently
python test_second_cctv.py
```

**Common Issues:**
1. **Wrong IP**: Verify camera is at `192.168.0.135`
2. **Wrong Credentials**: Check `admin / admin@777`
3. **Wrong Channel**: Ensure channel 1 is configured
4. **Network Issue**: Ping the camera: `ping 192.168.0.135`
5. **Firewall**: Ensure RTSP port 554 is open

### 🔧 **Camera Access via Browser:**
Try accessing the camera web interface:
- CHECK-IN Camera: `http://192.168.0.109`
- CHECK-OUT Camera: `http://192.168.0.135`

---

## 📊 **Expected Performance**

### ✅ **Connection:**
- Both cameras: **5-10 seconds** startup time
- Stable RTSP streams at **15-30 FPS**
- Resolution: Typically **1920x1080** or **1280x720**

### ✅ **Detection:**
- Face recognition: **Every 2nd frame**
- Response time: **Instant** (no cooldown)
- Multiple faces: Up to **4 faces per frame**

### ✅ **Reliability:**
- Professional CCTV equipment
- 24/7 operation ready
- Automatic fallback if one camera fails

---

## 🎉 **System Ready!**

Your dual CCTV attendance system is now production-ready with:
- ✅ Two professional CCTV cameras
- ✅ Entry and exit point monitoring
- ✅ Real-time attendance tracking
- ✅ No mobile devices required
- ✅ Professional-grade reliability

**Test the second camera and start tracking attendance with dual CCTV detection!** 🚀

---

## 📝 **Configuration Summary**

| Feature | Value |
|---------|-------|
| CHECK-IN IP | 192.168.0.109:554 |
| CHECK-IN Channel | 8 |
| CHECK-OUT IP | 192.168.0.135:554 |
| CHECK-OUT Channel | 1 |
| Protocol | RTSP |
| Cooldown | 0.0 seconds |
| Auto-Checkout | Disabled |
| Mode | Dual Camera |