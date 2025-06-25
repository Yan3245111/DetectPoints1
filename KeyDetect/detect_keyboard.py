import time
import keyboard
import threading
from typing import Callable

# sudo usermod -aG input $USER

class DetectKeyBoard:
    def __init__(self, callback: Callable):
        self._callback = callback
        # ATTR
        self._left_pressed = False
        self._right_pressed = False
        self._left_last_time = 0
        self._right_last_time = 0
        # THREAD
        self._thread_switch = True
        self._key_event_thread = threading.Thread(target=self._key_event, daemon=True)
        self._key_event_thread.start()
        self._timeout_thread = threading.Thread(target=self._timeout, daemon=True)
        self._timeout_thread.start()

    def _key_event(self):
        while self._thread_switch:
            if keyboard.is_pressed('a'):
                if not self._left_pressed:
                    self._left_pressed = True
                    self._callback(btn="a", is_pressed=True)
                    print("left pressed")
                self._left_last_time = time.time()
            if keyboard.is_pressed('s'):
                if not self._right_pressed:
                    self._right_pressed = True
                    self._callback(btn="s", is_pressed=True)
                    print("right pressed")
                self._right_last_time = time.time()
            time.sleep(0.01)  # 降低 CPU 占用

    def _timeout(self):
        while self._thread_switch:
            now = time.time()
            if self._left_pressed and (now - self._left_last_time > 0.3):
                self._left_pressed = False
                self._callback(btn="a", is_pressed=False)
                print("left released")
            if self._right_pressed and (now - self._right_last_time > 0.3):
                self._right_pressed = False
                self._callback(btn="s", is_pressed=False)
                print("right released")
            time.sleep(0.05)  # 检查间隔更短，但不会占用太多 CPU

    def stop(self):
        self._thread_switch = False
        self._key_event_thread.join()
        self._timeout_thread.join()

if __name__ == '__main__':
    def __callback(btn: str, is_pressed: bool):
        print(f"btn={btn}, is_pressed={is_pressed}")
    
    detect = DetectKeyBoard(callback=__callback)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detect.stop()