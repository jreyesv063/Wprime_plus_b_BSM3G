from wprime_plus_b.utils.configs.config import Config

class DatasetConfig(Config):
    
    def __init__(self, name: str, nsplit: int):
        super().__init__(name=name)
        
        self.name = name
        self.nsplit = nsplit