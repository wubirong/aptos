from __future__ import annotations

import threading
import queue
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .hand_tracker import HandTracker
from .gesture_controller import GestureController


@dataclass
class AppState:
    running: bool = False
    gesture_enabled: bool = False
    capture_index: int = 0


class GestureApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Hand Gesture Control")
        self.geometry("920x700")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.state = AppState()
        self.cap: Optional[cv2.VideoCapture] = None
        self.tracker = HandTracker()
        self.controller = GestureController(pointer_sensitivity=0.6)

        self._build_ui()
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_queue: "queue.Queue[Image.Image]" = queue.Queue(maxsize=2)

    def _build_ui(self) -> None:
        controls = tk.Frame(self)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.toggle_btn = tk.Button(controls, text="Start", command=self.toggle_run)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)

        self.gesture_var = tk.BooleanVar(value=False)
        self.gesture_chk = tk.Checkbutton(
            controls, text="Enable Control", variable=self.gesture_var, command=self._on_toggle_control
        )
        self.gesture_chk.pack(side=tk.LEFT, padx=5)

        self.device_label = tk.Label(controls, text="Camera Index:")
        self.device_label.pack(side=tk.LEFT, padx=(15, 0))
        self.device_entry = tk.Entry(controls, width=5)
        self.device_entry.insert(0, "0")
        self.device_entry.pack(side=tk.LEFT)

        self.info_label = tk.Label(controls, text="Raise index finger to move. Pinch to click.")
        self.info_label.pack(side=tk.LEFT, padx=15)

        self.canvas = tk.Label(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def toggle_run(self) -> None:
        if not self.state.running:
            try:
                self.state.capture_index = int(self.device_entry.get())
            except Exception:
                self.state.capture_index = 0
            self.cap = cv2.VideoCapture(self.state.capture_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Retry without CAP_DSHOW for Linux/macOS compatibility
                self.cap = cv2.VideoCapture(self.state.capture_index)
            if not self.cap.isOpened():
                self.info_label.config(text="Failed to open camera")
                return
            self.state.running = True
            self.toggle_btn.config(text="Stop")
            self._capture_thread = threading.Thread(target=self._loop, daemon=True)
            self._capture_thread.start()
            # Start UI update loop on main thread
            self.after(10, self._update_canvas)
        else:
            self.state.running = False
            self.toggle_btn.config(text="Start")
            # Clear any stale frames
            try:
                while True:
                    self._frame_queue.get_nowait()
            except queue.Empty:
                pass

    def _on_toggle_control(self) -> None:
        self.state.gesture_enabled = self.gesture_var.get()

    def _loop(self) -> None:
        pinch_down = False
        while self.state.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            vis, hands = self.tracker.process_frame(frame)

            if hands:
                hand = hands[0]
                # Index finger tip (8), thumb tip (4) as normalized coords
                idx = hand.landmarks[8]
                thb = hand.landmarks[4]
                if self.state.gesture_enabled:
                    self.controller.move_pointer(idx[0], idx[1])
                    # Pinch detection
                    dx = (idx[0] - thb[0])
                    dy = (idx[1] - thb[1])
                    dist = np.hypot(dx, dy)
                    if dist < 0.05 and not pinch_down:
                        self.controller.left_click()
                        pinch_down = True
                    elif dist >= 0.07:
                        pinch_down = False

                # Draw pinch distance
                h, w = vis.shape[:2]
                p1 = (int(idx[0] * w), int(idx[1] * h))
                p2 = (int(thb[0] * w), int(thb[1] * h))
                cv2.line(vis, p1, p2, (0, 255, 255), 2)
                cv2.circle(vis, p1, 6, (0, 255, 0), -1)
                cv2.circle(vis, p2, 6, (0, 128, 255), -1)

            # Prepare PIL image for UI thread and enqueue latest frame only
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (900, int(900 * rgb.shape[0] / rgb.shape[1])))
            image = Image.fromarray(resized)
            try:
                # Keep only the most recent frame to avoid backlog
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put_nowait(image)
            except queue.Full:
                pass

            time.sleep(0.01)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self) -> None:
        self.state.running = False
        self.destroy()

    def _update_canvas(self) -> None:
        """Update the Tkinter canvas image from the latest queued frame.

        Runs on the main thread via `after` to keep UI thread-safe.
        """
        try:
            image = self._frame_queue.get_nowait()
        except queue.Empty:
            image = None

        if image is not None:
            imgtk = ImageTk.PhotoImage(image=image)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        if self.state.running:
            self.after(10, self._update_canvas)


def main() -> None:
    app = GestureApp()
    app.mainloop()


if __name__ == "__main__":
    main()
