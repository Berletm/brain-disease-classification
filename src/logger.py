class Tee:
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for stream in self.streams:
            if stream:
                stream.write(data)
    
    def flush(self):
        for stream in self.streams:
            if stream:
                stream.flush()
    
    def isatty(self):
        return any(getattr(s, 'isatty', lambda: False)() for s in self.streams if s)
    
    def close(self):
        for stream in self.streams:
            if stream and hasattr(stream, 'close'):
                stream.close()