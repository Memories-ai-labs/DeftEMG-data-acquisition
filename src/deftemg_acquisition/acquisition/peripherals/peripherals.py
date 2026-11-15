from abc import ABC, abstractmethod
from pathlib import Path


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

    def save_files(self, session_dir: Path) -> None:
        """Write files to keep with the session, beside ``session.json``: called once, after
        ``open``, with the session folder made. None by default."""
        return
