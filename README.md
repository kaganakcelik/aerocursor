# AeroCursor (macOS)

A real-time **hand-tracking mouse controller for macOS** built with **MediaPipe**, **OpenCV**, **PyAutoGUI**, and **Quartz**.  
Control your cursor, clicks, dragging, and scrolling using natural hand gestures — no extra hardware required.

> ⚠️ **macOS ONLY (for now)**  
> This project relies on Apple’s **Quartz** framework for smooth, system-level mouse dragging.  
> Windows and Linux are **not supported** in the current version.

---

## Table of Contents
- Overview
- Features
- Gesture Controls
- System Requirements
- Installation
- MediaPipe Model Setup
- Running the Project
- Configuration & Tuning
- Architecture Overview
- macOS Limitation (Why Quartz?)
- Safety & Fail-Safe
- Known Limitations
- Roadmap
- License
- Author

---

## Overview

AeroCursor maps **hand motion to cursor motion** with:
- exponential smoothing
- jitter dead-zones
- inner-region camera mapping
- multithreaded cursor + scroll controllers

The result is **low-latency, stable, product-grade pointer control** suitable for daily use, demos, or research in human-computer interaction.

---

## Features

- 🖱️ **Smooth cursor movement**
  - ~200Hz update loop
  - distance-proportional easing
  - deadband to eliminate tremor noise

- ✌️ **Gesture-based input**
  - Click
  - Drag
  - Scroll

- 🎯 **Inner-area mapping**
  - Only the center region of the camera feed maps to screen space
  - Prevents edge snapping
  - Asymmetric margins tuned for right-handed use

- 🧵 **Multithreaded design**
  - Cursor movement thread
  - Scroll smoothing thread
  - Gesture detection logic in main loop

- 🍎 **Native macOS dragging**
  - Quartz events (`kCGEventLeftMouseDragged`)
  - Much smoother than PyAutoGUI drag emulation

---

## Gesture Controls

| Action | Gesture |
|-----|------|
| Move Cursor | Move hand (ring finger MCP joint) |
| Left Click | Quick pinch (thumb + index) |
| Drag | Hold pinch > 0.2 seconds |
| Scroll | Pinch (thumb + middle) + move vertically |
| Emergency Stop | Slam mouse to any screen corner |

---

## System Requirements

### Hardware
- macOS machine (Intel or Apple Silicon)
- Webcam (60 FPS recommended)

### Software
- macOS **10.15+**
- Python **3.9 – 3.11**

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kaganakcelik/aerocursor.git
cd aerocursor
```

### 2. Create and activate a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install opencv-python mediapipe pyautogui numpy
```

### 4. Running the Project
```bash
python main.py
```
