import abc

class Filter:
    @abc.abstractmethod
    def filter(self, predictions):
        pass

class ConfidenceThreshold(Filter):
    """Filter all predictions below a certain threshold"""
    def __init__(self, threshold):
        assert threshold <= 1.0 and threshold >= 0.0, "Invalid threshold"
        self.threshold = threshold

    def filter(self, predictions):
        return list(
            filter(lambda x: x.confidence > self.threshold, predictions))

class HighestConfidence(Filter):
    """Filter all but the highest confidence instance of each class"""
    def filter(self, predictions):
        confidence_map = {}
        for p in predictions:
            if p.class_name not in confidence_map.keys():
                confidence_map[p.class_name] = (p.confidence, p)
            else:
                if p.confidence > confidence_map[p.class_name][0]:
                    confidence_map[p.class_name] = (p.confidence, p)
        return [x[1] for x in confidence_map.values()]