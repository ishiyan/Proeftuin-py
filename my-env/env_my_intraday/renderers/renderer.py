from abc import ABC, abstractmethod
from numbers import Real

from ..accounts.account import Account
from ..providers.provider import Provider
from ..processors.processor import Processor
from ..frame import Frame 

class Renderer(ABC):

    @abstractmethod
    def reset(self, episode_number: int, account: Account, provider: Provider, processor: Processor, frame: Frame):
        raise NotImplementedError()

    @abstractmethod
    def step(self, frame: Frame, reward: Real):
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    def save_rendering(self, filepath):
        raise NotImplementedError()
    