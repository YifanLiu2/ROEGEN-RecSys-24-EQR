class DataParser:
    def __init__(self):
        self.path = None
    
    @property
    def path(self):
        return self.path
    
    @path.setter
    def path(self, data_path: str):
        self.path = data_path