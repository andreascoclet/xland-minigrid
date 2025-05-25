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
    agent_cache = np.zeros((tiles.shape[0], tiles.shape[1], 4, tile_size, tile_size, 3), dtype=np.uint8)

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

            # render 4 agent orientations
            for direction in range(4):
                tile_w_agent_img = render_tile(
                    tile=tuple(tiles[y, x]),
                    agent_direction=direction,
                    highlight=False,
                    tile_size=tile_size,
                )
                agent_cache[y, x, direction] = tile_w_agent_img

    return cache, agent_cache


# building cache of pre-rendered tiles
TILE_SIZE = 32

cache_path = os.path.join(CACHE_PATH, "render_cache")

if not os.path.exists(cache_path) or FORCE_RELOAD:
    os.makedirs(CACHE_PATH, exist_ok=True)
    print("Building rendering cache, may take a while...")
    TILE_CACHE, TILE_W_AGENT_CACHE = build_cache(np.asarray(TILES_REGISTRY), tile_size=TILE_SIZE)
    TILE_CACHE = jnp.asarray(TILE_CACHE).reshape(-1, TILE_SIZE, TILE_SIZE, 3)
    TILE_W_AGENT_CACHE = jnp.asarray(TILE_W_AGENT_CACHE).reshape(-1, 4, TILE_SIZE, TILE_SIZE, 3)

    print(f"Done. Cache is saved to {cache_path} and will be reused on consequent runs.")
    save_bz2_pickle({"tile_cache": TILE_CACHE, "tile_agent_cache": TILE_W_AGENT_CACHE}, cache_path)

TILE_CACHE = load_bz2_pickle(cache_path)["tile_cache"]
TILE_W_AGENT_CACHE = load_bz2_pickle(cache_path)["tile_agent_cache"]


# 🔧 Rotate agent position (used for symbolic rendering)
def rotate_pos(pos, direction, H, W):
    y, x = pos[0], pos[1]
    return jax.lax.switch(
        direction,
        (
            lambda: jnp.array([y, x]),
            lambda: jnp.array([ W - 1 - x, y]),
            lambda: jnp.array([H - 1- y, W - 1 - x]),
            lambda: jnp.array([x, H - 1 - y]),
        ),
    )

def rotate_grid(grid, direction):
    """
    Rotate a [H, W, …] grid 0–3 steps CCW by remapping indices,
    exactly like rotate_pos does for a single (y,x).
    """
    H, W = grid.shape[:2]

    def rot0():
        # no change
        return grid

    def rot1():
        # new shape = [W, H, …]
        # new_i in [0..W), new_j in [0..H)
        i = jnp.arange(W)[:, None]    # shape (W,1)
        j = jnp.arange(H)[None, :]    # shape (1,H)
        # invert the single‐point mapping: orig_y = j, orig_x = W-1-i
        orig_y = jnp.broadcast_to(j, (W, H))
        orig_x = jnp.broadcast_to(W - 1 - i, (W, H))
        # advanced‐indexing yields shape (W, H, …)
        return grid[orig_y, orig_x]

    def rot2():
        # new shape = [H, W, …]
        i = jnp.arange(H)[:, None]    # (H,1)
        j = jnp.arange(W)[None, :]    # (1,W)
        # invert 180°: orig_y = H-1-i, orig_x = W-1-j
        orig_y = jnp.broadcast_to(H - 1 - i, (H, W))
        orig_x = jnp.broadcast_to(W - 1 - j, (H, W))
        return grid[orig_y, orig_x]

    def rot3():
        # new shape = [W, H, …]
        i = jnp.arange(W)[:, None]    # (W,1)
        j = jnp.arange(H)[None, :]    # (1,H)
        # invert 270° CCW: orig_y = H-1-j, orig_x = i
        orig_y = jnp.broadcast_to(H - 1 - j, (W, H))
        orig_x = jnp.broadcast_to(i, (W, H))
        return grid[orig_y, orig_x]

    return jax.lax.switch(direction, (rot0, rot1, rot2, rot3))

# ✅ Core rendering logic
def _render_obs(timestep, rotated: bool = False) -> jax.Array:
    #print(f"using new render")
    grid = timestep.state.grid
    agent_pos = timestep.state.agent.position
    agent_dir = timestep.state.agent.direction

    H, W = grid.shape[:2]

    # if rotated:
    #     # 1. Rotate symbolic grid
    #     rotated_grid = rotate_grid(grid, agent_dir)

    #     # 2. Rotate agent position
    #     ry, rx = rotate_pos(agent_pos, agent_dir, H, W)
    # else:
    #     # Use unrotated grid and agent position
    #     rotated_grid = grid
    #     ry, rx = agent_pos[0], agent_pos[1]

    # 3. Render background
    idx = grid[:, :, 0] * NUM_COLORS + grid[:, :, 1]
    rendered_obs = jnp.take(TILE_CACHE, idx, axis=0)

    # 4. Render agent (always facing UP)
    agent_tile = TILE_W_AGENT_CACHE[idx[agent_pos[0], agent_pos[1]], agent_dir]
    rendered_obs = rendered_obs.at[agent_pos[0], agent_pos[1]].set(agent_tile)
    # 5. Flatten to RGB
    rgb_img = rendered_obs.transpose((0, 2, 1, 3, 4)).reshape(H * TILE_SIZE, W * TILE_SIZE, 3) / 255
    return rgb_img

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
