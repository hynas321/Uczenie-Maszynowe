from typing import List


class LabelEncoder:
    def __init__(self):
        self.classes_: List[str] = []

    def fit(self, data: List[str]) -> None:
        self.classes_ = sorted(set(data))

    def transform(self, data: List[str]) -> List[int]:
        class_to_index = {cls: i for i, cls in enumerate(self.classes_)}
        return [class_to_index[d] for d in data]