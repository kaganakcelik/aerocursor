import cv2
import mediapipe as mp
import time
import math
import threading
import pyautogui
import numpy as np
import Quartz # For smooth dragging on Mac

# --- Configuration ---
pyautogui.FAILSAFE = True  # Drag mouse to corner to kill script
pyautogui.PAUSE = 0        # maximize speed

# Mapping: Use center 70% of camera feed for full screen
# Asymmetric X margins for right-handed use (shift active area to right)
SCREEN_MARGIN_LEFT = 0.35
SCREEN_MARGIN_RIGHT = 0.05
SCREEN_MARGIN_Y = 0.2

# Smoothing
# Higher = slower lag but smoother. 
# Prompt suggested 12. With 200Hz loop, 12 might be too fast (1/12th every 5ms).
# Let's increase slightly to ensure smoothness, or rely on the high update rate.
# Actually, prompt example was likely running at frame rate (30-60fps).
# If we run at 200Hz, we need a larger divisor to behave similarly.
# 60fps / 12 = 5 steps to close gap? No, 1/12th decay.
# If we run 3x faster, we need 3x the divisor to have same decay profile per second.
# Let's try 20.0 to be safe against jitter.
SMOOTHING_FACTOR_BASE = 10.0
MIN_STEP = 0.0001
# Deadband: Increase slightly to ignore hand tremors
JITTER_THRESHOLD = 0.004 # Normalized distance (0.4% of screen diagonal)

# Gestures
CLICK_THRESHOLD_RATIO = 0.2  # Distance / HandSize
SCROLL_THRESHOLD_RATIO = 0.2
CURSOR_ACTIVATION_THRESHOLD_RATIO = 0.6 # Distance / HandSize to enable cursor movement
SCROLL_SPEED_MULTIPLIER = 0.03 # Sensitivity (applied before internal scaling)
DRAG_DELAY = 0.4              # Seconds to wait before dragging

# Landmarks
WRIST = 0
INDEX_TIP = 8
THUMB_TIP = 4
MIDDLE_TIP = 12
RING_MCP = 13  # Stable anchor for cursor
MIDDLE_MCP = 9

class CursorController:
    """
    Handles smooth cursor movement in a separate thread.
    """
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        
        # State
        self.target_x = self.screen_w / 2
        self.target_y = self.screen_h / 2
        self.current_x = self.screen_w / 2
        self.current_y = self.screen_h / 2
        self.is_dragging = False # New state
        
        self.running = True
        self.thread = threading.Thread(target=self._move_loop, daemon=True)
        self.thread.start()

    def set_dragging(self, dragging):
        self.is_dragging = dragging

    def update_target(self, norm_x, norm_y):
        """
        Updates the target position based on normalized camera coordinates (0-1).
        Applies 'Inner Area Mapping'.
        """
        # Map [margin_left, 1-margin_right] to [0, 1] for X
        # Map [margin_y, 1-margin_y] to [0, 1] for Y
        
        active_width = 1.0 - (SCREEN_MARGIN_LEFT + SCREEN_MARGIN_RIGHT)
        active_height = 1.0 - (2 * SCREEN_MARGIN_Y)
        
        raw_x = (norm_x - SCREEN_MARGIN_LEFT) / active_width
        raw_y = (norm_y - SCREEN_MARGIN_Y) / active_height
        
        # Clamp to 0-1
        raw_x = max(0, min(1, raw_x))
        raw_y = max(0, min(1, raw_y))
        
        # Convert to pixels
        new_target_x = raw_x * self.screen_w
        new_target_y = raw_y * self.screen_h
        
        # Jitter gating (deadband)
        dist = math.hypot(new_target_x - self.target_x, new_target_y - self.target_y)
        pixel_jitter_limit = JITTER_THRESHOLD * math.hypot(self.screen_w, self.screen_h)

        if dist > pixel_jitter_limit:
            self.target_x = new_target_x
            self.target_y = new_target_y

    def _move_loop(self):
        """
        Continuously moves current position towards target position.
        """
        while self.running:
            dx = self.target_x - self.current_x
            dy = self.target_y - self.current_y
            dist = math.hypot(dx, dy)
            
            if dist < 1.0:
                time.sleep(0.001)
                continue

            # Variable step size: fast catch-up, slow convergence
            # "step = distance / factor" logic implies velocity is proportional to distance
            # This is equivalent to exponential smoothing: current += (target - current) * alpha
            
            # Use a constant easing factor. 
            # If we run at ~200Hz, a factor of 5-10 gives Snappy but Smooth.
            # Factor 12.0 might be a bit slow, but let's stick closer to the prompt's divisor logic first, 
            # bearing in mind we need a fraction here.
            
            # The prompt's "step = dist/12" meant displacement, i.e.,  move dist/12 pixels.
            # Since dx is the full vector, we want (1/12) of dx.
            
            easing_factor = 1.0 / SMOOTHING_FACTOR_BASE
            
            move_x = dx * easing_factor
            move_y = dy * easing_factor

            # Snap to target if very close (avoid micro-drifting for pixel-perfect stops)
            if dist < 0.5:
                move_x = dx
                move_y = dy
            
            self.current_x += move_x
            self.current_y += move_y
            
            try:
                if self.is_dragging:
                    # Use Quartz for smooth dragging on Mac
                    # pyautogui.moveTo often fails to generate correct drag events
                    ev = Quartz.CGEventCreateMouseEvent(
                        None, 
                        Quartz.kCGEventLeftMouseDragged, 
                        (self.current_x, self.current_y), 
                        Quartz.kCGMouseButtonLeft
                    )
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
                else:
                    pyautogui.moveTo(self.current_x, self.current_y, _pause=False)
            except pyautogui.FailSafeException:
                pass # User aborted or hit corner
            
            time.sleep(0.005) # ~200 updates/sec

    def stop(self):
        self.running = False
        self.thread.join()

class ScrollController:
    """
    Handles smooth scrolling using the same easing logic as CursorController.
    """
    def __init__(self):
        self.target_scroll_y = 0.0
        self.current_scroll_y = 0.0
        self.points_per_scroll_click = 1 # Accumulate this much before emitting a scroll event
        self.scroll_accumulator = 0.0
        
        self.running = True
        self.thread = threading.Thread(target=self._scroll_loop, daemon=True)
        self.thread.start()

    def add_scroll_delta(self, delta):
        """
        Adds to the target scroll "virtual position".
        Delta should be scaled (e.g., pixels or lines).
        """
        self.target_scroll_y += delta

    def _scroll_loop(self):
        while self.running:
            diff = self.target_scroll_y - self.current_scroll_y
            
            # Snap if close to avoid endless micro-scrolling
            if abs(diff) < 0.1:
                # If we have residual accumulator, we might want to clear it or wait
                # But basically we are "stopped"
                self.current_scroll_y = self.target_scroll_y
                time.sleep(0.01)
                continue
                
            # Easing
            easing_factor = 1.0 / SMOOTHING_FACTOR_BASE
            step = diff * easing_factor
            
            self.current_scroll_y += step
            
            # Accumulate integer scrolling
            self.scroll_accumulator += step
            
            scroll_clicks = int(self.scroll_accumulator)
            if scroll_clicks != 0:
                pyautogui.scroll(scroll_clicks)
                self.scroll_accumulator -= scroll_clicks
            
            time.sleep(0.005) # ~200Hz

    def stop(self):
        self.running = False
        self.thread.join()

class GestureDetector:
    """
    Handles clicks, drags, and scrolling.
    """
    def __init__(self, scroll_controller, cursor_controller):
        self.scroll_controller = scroll_controller
        self.cursor_controller = cursor_controller
        
        self.is_clicking = False # Treated as "is_dragging" state in new logic
        self.is_pinched = False # Separate tracking for pinch state
        self.pinch_start_time = 0
        self.is_scrolling = False
        self.last_scroll_y = -1
    
    def update(self, landmarks):
        """
        Process landmarks to detect gestures.
        """
        # 1. Estimate hand size
        w = landmarks[WRIST]
        m_mcp = landmarks[MIDDLE_MCP]
        hand_size = math.hypot(w.x - m_mcp.x, w.y - m_mcp.y)
        
        if hand_size < 0.01: return

        # 2. Click Detection (Index + Thumb)
        idx = landmarks[INDEX_TIP]
        thm = landmarks[THUMB_TIP]
        pinch_dist = math.hypot(idx.x - thm.x, idx.y - thm.y)
        
        click_threshold = CLICK_THRESHOLD_RATIO * hand_size
        
        # State machine for Click vs Drag
        if pinch_dist < click_threshold:
            # Pinching
            if not self.is_pinched:
                self.is_pinched = True
                self.pinch_start_time = time.time()
            
            # If pinched for long enough, start dragging
            if self.is_pinched and not self.is_clicking:
                if time.time() - self.pinch_start_time > DRAG_DELAY:
                    pyautogui.mouseDown()
                    self.is_clicking = True # Now effectively dragging
                    self.cursor_controller.set_dragging(True) # Notify cursor
        else:
            # Not pinching
            if self.is_pinched:
                if self.is_clicking:
                    # Was dragging, so release
                    pyautogui.mouseUp()
                    self.is_clicking = False
                    self.cursor_controller.set_dragging(False) # Notify cursor
                else:
                    # Was pinched but didn't drag yet -> Click
                    pyautogui.click()
                
                self.is_pinched = False
                self.pinch_start_time = 0

        # 3. Scroll Detection (Middle + Thumb)
        mid = landmarks[MIDDLE_TIP]
        scroll_pinch_dist = math.hypot(mid.x - thm.x, mid.y - thm.y)
        scroll_threshold = SCROLL_THRESHOLD_RATIO * hand_size
        
        if scroll_pinch_dist < scroll_threshold:
            if not self.is_scrolling:
                self.is_scrolling = True
                self.last_scroll_y = mid.y
            else:
                # Calculate delta
                dy = mid.y - self.last_scroll_y
                self.last_scroll_y = mid.y
                
                # Send to controller
                # dy is normalized (0-1). 
                # Map to scroll units. 
                # Moving hand UP (negative dy) -> Scroll UP (positive value) 
                # (Standard wheel logic: pushing wheel forward/up scrolls page up? No, pushes content down usually)
                # Let's keep existing logic: dy * multiplier
                # Need substantial scaling because dy is small (~0.01)
                
                scroll_delta = dy * SCROLL_SPEED_MULTIPLIER * 5000 
                # 5000 is heuristic: 1% screen move (0.01) * 1.0 * 5000 = 50 clicks. 
                # That's a good "flick". 
                # User had MULTIPLIER=1.0.
                
                self.scroll_controller.add_scroll_delta(scroll_delta)
        else:
            self.is_scrolling = False
            self.last_scroll_y = -1

    def stop(self):
        # No thread to stop directly (handled by controllers)
        pass

def main():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Initialize components
    cursor = CursorController()
    scroll = ScrollController() # New dedicated controller
    gestures = GestureDetector(scroll, cursor) # Updated signature
    
    # Initialize MediaPipe
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60) 
    
    start_time = time.time()
    
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            print("Hand Tracking Started. Press 'q' to quit.")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detection
                ts = int((time.time() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, ts)
                
                if result.hand_landmarks:
                    hand_lms = result.hand_landmarks[0] 
                    
                    # 1. Cursor Movement (Gated by Pinch Distance)
                    # User: "cursor only gets activated when my finger and thumb are within a certain distance"
                    idx = hand_lms[INDEX_TIP]
                    thm = hand_lms[THUMB_TIP]
                    cursor_pinch_dist = math.hypot(idx.x - thm.x, idx.y - thm.y)
                    
                    # Calculate Hand Size for adaptive threshold
                    w = hand_lms[WRIST]
                    m_mcp = hand_lms[MIDDLE_MCP]
                    eff_hand_size = math.hypot(w.x - m_mcp.x, w.y - m_mcp.y)
                    
                    if eff_hand_size > 0.01:
                        activation_thresh = CURSOR_ACTIVATION_THRESHOLD_RATIO * eff_hand_size
                        if cursor_pinch_dist < activation_thresh:
                            ring_mcp = hand_lms[RING_MCP]
                            cursor.update_target(ring_mcp.x, ring_mcp.y)
                    
                    # 2. Gestures
                    gestures.update(hand_lms)
                    
                    # 3. Visualization
                    h, w, _ = frame.shape
                    
                    cx, cy = int(hand_lms[RING_MCP].x * w), int(hand_lms[RING_MCP].y * h)
                    
                    # Visual feedback for activation
                    color = (0, 0, 255) # Red = Stop
                    if eff_hand_size > 0.01 and cursor_pinch_dist < (CURSOR_ACTIVATION_THRESHOLD_RATIO * eff_hand_size):
                        color = (0, 255, 255) # Yellow = Active
                        
                    cv2.circle(frame, (cx, cy), 8, color, -1)
                    
                    if gestures.is_clicking:
                        cv2.putText(frame, "CLICK/DRAG", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if gestures.is_scrolling:
                        cv2.putText(frame, "SCROLL", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Draw Active Region Box
                h, w, _ = frame.shape
                # Coordinates in pixel space
                # x_min = SCREEN_MARGIN_LEFT * w
                # x_max = (1 - SCREEN_MARGIN_RIGHT) * w
                # y_min = SCREEN_MARGIN_Y * h
                # y_max = (1 - SCREEN_MARGIN_Y) * h
                
                start_point = (int(SCREEN_MARGIN_LEFT * w), int(SCREEN_MARGIN_Y * h))
                end_point = (int((1 - SCREEN_MARGIN_RIGHT) * w), int((1 - SCREEN_MARGIN_Y) * h))
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 1)

                cv2.imshow('Product-Grade Hand Track', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        cursor.stop()
        scroll.stop()
        gestures.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()