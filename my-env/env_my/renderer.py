from abc import ABC, abstractmethod

class Renderer(ABC):
    @property
    @abstractmethod
    def render_frame(self, info: dict, mode='human'):
        raise NotImplementedError()

    @abstractmethod
    def render_all(self, info: dict, title=None):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def save_rendering(self, filepath):
        raise NotImplementedError()

    @abstractmethod
    def pause_rendering(self):
        raise NotImplementedError()
    