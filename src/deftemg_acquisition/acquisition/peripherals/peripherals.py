from abc import ABC, abstractmethod

class Peripheral(ABC):
    """
    Abstract class for peripherals
    """

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    