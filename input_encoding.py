
from typing import List

from enumerations import ClassificationFormat
from framework_component import FunctionalComponent


class BaseInputEncoding(FunctionalComponent):
    def apply(self, X:List["keras.Input"], prefix: str = None):
        raise NotImplementedError("Please override this with a custom implementation")

    @property
    def required_input_format(self) -> ClassificationFormat:
        raise NotImplementedError("Please override this with a custom implementation")
