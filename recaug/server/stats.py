import time

import cv2

class FPSTracker(object):
    def __init__(self, rate=1):
        self._start = None
        self._rate = rate
        self._count = 0
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._format = 'FPS: {:.1f}'
        self._fps = 0

    def start(self):
        self._start = time.time()

    def update_count(self):
        self._assert_started()
        self._count += 1

        elapsed = time.time() - self._start
        if elapsed >= self._rate:
            self._fps = self._count / elapsed
            self._count = 0
            self._start = time.time() 
 
    def visualize(self, frame):
        self._assert_started()
        display_str = self._format.format(self.fps)
        cv2.putText(frame, display_str, (10,30), self._font, 0.75, (155, 0, 255), 1, cv2.LINE_AA)

    @property
    def fps(self):
        self._assert_started()
        return self._fps

    def _assert_started(self):
        if self._start is None:
            raise AssertionError("FPSCounter has not been started yet")

class NetStatsTracker(object):
    def __init__(self, check_rate=900):
        self._check_rate = check_rate
        self._check_idx = 0
        self._jpg_running_total = 0
        self._avg_jpg_size = 0

    def update(self, jpg_size):
        self._check_idx += 1
        self._jpg_running_total += jpg_size
        if self._check_idx % self._check_rate == 0:
            self._avg_jpg_size = self._jpg_running_total / self._check_rate
            self._check_idx = 0
            self._jpg_running_total = 0
            print("[NetStats] Avg JPG Size: {:.2f} bytes".format(self._avg_jpg_size))
 
    @property
    def avg_jpg_size(self):
        return self._avg_jpg_size