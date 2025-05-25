class TargetValueMapping:
    def __init__(self):
        self.FALSE:int = 0
        self.TRUE:int = 1
    def _asdict(self):
        return self.dict
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    