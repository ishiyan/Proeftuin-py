from copy import deepcopy
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.error import DependencyNotInstalled

class HumanRendering(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs):
    """
    Allows human like rendering for environments that support `rgb_array`
    rendering.

    This wrapper is particularly useful when you have implemented an
    environment that can produce RGB images but haven't implemented any
    code to render the images to the screen.
    
    If you want to use this wrapper with your environments, remember to
    specify `render_fps` in the metadata of your environment.

    The `render_mode` of the wrapped environment must be `rgb_array`.

    No vector version of the wrapper exists.
    """
    def __init__(
        self,
        env: gym.Env[ObsType, ActType]):
        """
        Args:
            env:
                The environment that will be wrapped.
        """
        gym.Wrapper.__init__(self, env)
        if env.render_mode != 'rgb_array':
            raise ValueError(
                f'Render mode is {env.render_mode}, which is incompatible ',
                'with RecordAnimatedGIF. Initialize your environment with ',
                'the "rgb_array" render_mode.')

        if not 'render_fps' in self.env.metadata:
            raise ValueError(
                'The base environment must specify "render_fps" to be used with the HumanRendering wrapper')

        if 'human' not in self.metadata['render_modes']:
            self.metadata = deepcopy(self.env.metadata)
            self.metadata['render_modes'].append('human')

        self.screen_size = None
        self.window = None
        self.clock = None

    @property
    def render_mode(self):
        """Always returns `'human'`."""
        return 'human'

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the base environment and render a frame to the screen."""
        result = super().reset(seed=seed, options=options)
        self.render_frame()
        return result

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        """Perform a step in the base environment and render a frame to the screen."""
        result = super().step(action)
        self.render_frame()
        return result

    def render(self) -> None:
        """This method doesn't do much, actual rendering is performed in :meth:`step` and :meth:`reset`."""
        return None

    def close(self):
        """Close the rendering window."""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
        super().close()

    def render_frame(self):
        """Fetch the last frame from the base environment and render it to the screen."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            )

        last_rgb_array = self.env.render()
        if not isinstance(last_rgb_array, np.ndarray):
            raise ValueError('The render function of the environment should return '
                             f'a numpy array, not a {type(last_rgb_array)}.')
        rgb_array = np.transpose(last_rgb_array, axes=(1, 0, 2))

        if self.screen_size is None:
            self.screen_size = rgb_array.shape[:2]

        if self.screen_size != rgb_array.shape[:2]:
            raise ValueError(
                'The shape of the rgb array has changed from '
                f'{self.screen_size} to {rgb_array.shape[:2]}')

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # rgb_array includes an alpha channel (RGBA format),
        # which is not supported directly by make_surface.
        # Slice the rgb_array to exclude the alpha channel,
        # keeping only the first three channels (RGB).
        # print(f'Shape {last_rgb_array.shape} -> {rgb_array.shape}')
        surf = pygame.surfarray.make_surface(rgb_array[:, :, :3])
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
