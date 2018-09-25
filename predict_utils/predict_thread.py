import sys
from threading import Thread

sys.path.append('ssd_keras')
from trained_models.coco_300 import Coco300

class PredictThread(Thread):
    def __init__(self, frame_queue):
        Thread.__init__(self)
        self._frame_queue = frame_queue
        self._model = Coco300() 
        self.daemon = True
        self.start

    def run(self):
        while True:
            frame = self._frame_queue.get(True)

            predictions = self._model.predict(frame)
            predictions.visualize(frame)
            predicted_points = predictions.get_predicted_points()
            