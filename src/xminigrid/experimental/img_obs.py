# jit-compatible RGB observations. Currently experimental!
# if it proves useful and necessary in the future, I will consider rewriting env.render in such style also
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt  # 🔧ADDED for show_img




# 🔧ADDED show_img for visualization
def show_img(img, dpi=32):
    plt.figure(dpi=dpi)
    plt.axis('off')
    plt.imshow(img)
    plt.show()



from ..benchmarks import load_bz2_pickle, save_bz2_pickle
from ..core.constants import NUM_COLORS, NUM_LAYERS, TILES_REGISTRY
from ..rendering.rgb_render import render_tile
from ..wrappers import Wrapper

CACHE_PATH = os.environ.get("XLAND_MINIGRID_CACHE", os.path.expanduser("~/.xland_minigrid"))
FORCE_RELOAD = os.environ.get("XLAND_MINIGRID_RELOAD_CACHE", False)


def build_cache(tiles: np.ndarray, tile_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    cache = np.zeros((tiles.shape[0], tiles.shape[1], tile_size, tile_size, 3), dtype=np.uint8)
    agent_cache = np.zeros((tiles.shape[0], tiles.shape[1], tile_size, tile_size, 3), dtype=np.uint8)

    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            # rendering tile
            tile_img = render_tile(
                tile=tuple(tiles[y, x]),
                agent_direction=None,
                highlight=False,
                tile_size=int(tile_size),
            )
            cache[y, x] = tile_img

            # rendering agent on top
            tile_w_agent_img = render_tile(
                tile=tuple(tiles[y, x]),
                agent_direction=0,
                highlight=False,
                tile_size=int(tile_size),
            )
            agent_cache[y, x] = tile_w_agent_img

    return cache, agent_cache


# building cache of pre-rendered tiles
TILE_SIZE = 32

cache_path = os.path.join(CACHE_PATH, "render_cache")

if not os.path.exists(cache_path) or FORCE_RELOAD:
    os.makedirs(CACHE_PATH, exist_ok=True)
    print("Building rendering cache, may take a while...")
    TILE_CACHE, TILE_W_AGENT_CACHE = build_cache(np.asarray(TILES_REGISTRY), tile_size=TILE_SIZE)
    TILE_CACHE = jnp.asarray(TILE_CACHE).reshape(-1, TILE_SIZE, TILE_SIZE, 3)
    TILE_W_AGENT_CACHE = jnp.asarray(TILE_W_AGENT_CACHE).reshape(-1, TILE_SIZE, TILE_SIZE, 3)

    print(f"Done. Cache is saved to {cache_path} and will be reused on consequent runs.")
    save_bz2_pickle({"tile_cache": TILE_CACHE, "tile_agent_cache": TILE_W_AGENT_CACHE}, cache_path)

TILE_CACHE = load_bz2_pickle(cache_path)["tile_cache"]
TILE_W_AGENT_CACHE = load_bz2_pickle(cache_path)["tile_agent_cache"]


def _render_obs(timestep) -> jax.Array:
    grid = timestep.state.grid
    agent_pos = timestep.state.agent.position   # JAX array [2]
    agent_dir = timestep.state.agent.direction  # JAX scalar

    H, W = grid.shape[:2]
    obs_flat_idxs = grid[:, :, 0] * NUM_COLORS + grid[:, :, 1]

    # Render background tiles
    rendered_obs = jnp.take(TILE_CACHE, obs_flat_idxs, axis=0)

    # Use pre-rendered agent tile (assuming only one direction supported here)
    agent_y, agent_x = agent_pos[0], agent_pos[1]
    idx = obs_flat_idxs[agent_y, agent_x]
    agent_tile = TILE_W_AGENT_CACHE[idx]  # Direction-specific cache not handled here

    # Set agent tile in rendered image
    rendered_obs = rendered_obs.at[agent_y, agent_x].set(agent_tile)

    # Flatten to RGB image
    final_img = rendered_obs.transpose((0, 2, 1, 3, 4)).reshape(H * TILE_SIZE, W * TILE_SIZE, 3)

    return final_img


class RGBImgObservationWrapper(Wrapper):
    def observation_shape(self, params):
        new_shape = (params.view_size * TILE_SIZE, params.view_size * TILE_SIZE, 3)

        base_shape = self._env.observation_shape(params)
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {**base_shape, **{"img": new_shape}}
        else:
            obs_shape = new_shape

        return obs_shape

    def __convert_obs(self, timestep):
        rendered_img = _render_obs(timestep)
        return timestep.replace(observation=rendered_img)
    
    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__convert_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__convert_obs(timestep)
        return timestep
    
    def step_doorkey_deterministic(self, params, timestep, action):
        timestep = self._env.step_doorkey_deterministic(params, timestep, action)
        timestep = self.__convert_obs(timestep)
        return timestep
    
    def reset_doorkey_deterministic(self, params, key):
        timestep = self._env.reset_doorkey_deterministic(params, key)
        timestep = self.__convert_obs(timestep)
        return timestep
