from collections import OrderedDict
from dataclasses import fields
from typing import Any


class ModelOutput(OrderedDict):
    def keys(self) -> Any:
        for field in fields(self):
            yield field.name

    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Any:
        yield from self.keys()

    def values(self) -> Any:
        for field in fields(self):
            yield getattr(self, field.name)

    def items(self) -> Any:
        for field in fields(self):
            yield field.name, getattr(self, field.name)