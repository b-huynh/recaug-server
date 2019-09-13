class Event:
    def __init__(self):
        self.handlers = []

    def add(self, callback):
        self.handlers.append(callback)
        return self

    def remove(self, callback):
        self.handlers.remove(callback)
        return self

    def fire(self, *args, **keywargs):
        for handler in self.handlers:
            handler(*args, **keywargs)

    __iadd__ = add
    __isub__ = remove