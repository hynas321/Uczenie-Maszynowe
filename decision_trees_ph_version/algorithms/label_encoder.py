from typing import List


class LabelEncoder:
    def __init__(self):
        self.classes: List[str] = []

    def fit(self, data: List[str]) -> None:
        self.classes = sorted(set(data))

    def transform(self, data: List[str]) -> List[int]:
        class_to_index = {element: i for i, element in enumerate(self.classes)}
        return [class_to_index[d] for d in data]