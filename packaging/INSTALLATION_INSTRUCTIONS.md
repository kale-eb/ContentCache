# Installing silk.ai (Unsigned Version)

## ⚠️ Important: This app is unsigned for development purposes

When you try to open silk.ai, macOS may show a warning saying the app is "damaged" or from an "unidentified developer." This is normal for unsigned apps.

## Installation Steps:

### Method 1: Right-click to open (Easiest)
1. Download the DMG file
2. Open the DMG by double-clicking
3. **Right-click** on silk.ai.app and select "Open"
4. Click "Open" in the security dialog
5. Drag silk.ai to Applications folder

### Method 2: Terminal method (If right-click doesn't work)
1. Download and mount the DMG
2. Open Terminal
3. Run: `sudo xattr -rd com.apple.quarantine /Volumes/silk.ai*/silk.ai.app`
4. Run: `sudo spctl --add /Volumes/silk.ai*/silk.ai.app`
5. Now you can open the app normally

### Method 3: System Preferences (Alternative)
1. Try to open the app (it will fail)
2. Go to System Preferences → Security & Privacy → General
3. Click "Open Anyway" next to the silk.ai message
4. Confirm by clicking "Open"

## ✅ You only need to do this once!
After the first successful open, macOS will remember that you trust this app.

---

**Note:** In the future, we'll provide a properly signed version that won't require these steps. 