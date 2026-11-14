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

    @abstractmethod
    def start_acquisition(self) -> None:
        pass

    @abstractmethod
    def stop_acquisition(self) -> None:
        pass
