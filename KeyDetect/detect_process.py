import time
import keyboard
import multiprocessing as mp
from typing import Callable


class DetectPedal:
    def __init__(self, callback: Callable):
        self._callback = callback
        self._state = mp.Manager().dict(
            left_pressed=False,
            right_pressed=False,
            left_last_time=0.0,
            right_last_time=0.0,
            running=True
        )

        # 单一进程
        self._proc = mp.Process(target=self._loop, args=(self._state,))
        self._proc.start()

    def _loop(self, state):
        while state["running"]:
            now = time.time()

            # 检查键盘
            if keyboard.is_pressed('a'):
                if not state["left_pressed"]:
                    state["left_pressed"] = True
                    self._callback(btn="a", is_pressed=True)
                state["left_last_time"] = now

            if keyboard.is_pressed('s'):
                if not state["right_pressed"]:
                    state["right_pressed"] = True
                    self._callback(btn="s", is_pressed=True)
                state["right_last_time"] = now

            # 检查超时释放
            if state["left_pressed"] and (now - state["left_last_time"] > 0.3):
                state["left_pressed"] = False
                self._callback(btn="a", is_pressed=False)

            if state["right_pressed"] and (now - state["right_last_time"] > 0.3):
                state["right_pressed"] = False
                self._callback(btn="s", is_pressed=False)

            time.sleep(0.01)  # 循环间隔，兼顾检测频率和 CPU 占用

    def stop(self):
        self._state["running"] = False
        self._proc.join()


if __name__ == '__main__':
    def __callback(btn: str, is_pressed: bool):
        print(f"btn={btn}, is_pressed={is_pressed}")

    detect = DetectPedal(callback=__callback)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detect.stop()
        print("程序已退出")
