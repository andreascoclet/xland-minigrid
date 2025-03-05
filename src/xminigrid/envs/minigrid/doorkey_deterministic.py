from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


from typing import Optional
from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal
from ...core.grid import coordinates_mask, room, sample_coordinates, sample_direction, vertical_line
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State
from .doorkey import DoorKey


_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


class DoorKeyDeterministic(DoorKey):
    """
    A deterministic variant of DoorKey where the environment is reset
    using a predefined position (`pos`).
    """

    def __init__(self, pos: Optional[jax.Array] = None, **kwargs):
        """
        Initializes the environment with an optional deterministic `pos`.

        Args:
            pos (Optional[jax.Array]): Predefined positions (door, wall, key).
            kwargs: Additional parameters (ignored for now).
        """
        self.pos = pos  # Store deterministic position

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        """Generates the problem using a predefined `pos` instead of random sampling."""
        if self.pos is None:
            #raise ValueError("The position attribute 'pos' is not set. Provide `pos` when creating the environment.")
            return super()._generate_problem(params, key)  # ✅ Use normal random behavior if pos isn't set

        # Unpack the predefined positions
        door_pos, wall_pos, key_x, key_y = self.pos[:4]
        #print(type(door_pos))

        # Handle PRNG key correctly
        # seed = jax.random.PRNGKey(jnp.asarray(key, dtype=jnp.uint32))
        # seeds = jax.random.split(seed, num=2)  # For randomizing agent position and direction
        try:
            # Ensure door is between 1 and height-1
            if door_pos <= 0 or door_pos >= params.height - 1:
                raise ValueError(f"Door position must be between 1 and {params.height - 2} for grid of height {params.height} and width {params.width}.")

            # Ensure wall is between 2 and width-2
            if wall_pos <= 1 or wall_pos >= params.width - 2:
                raise ValueError(f"Wall position must be between 2 and {params.width - 3} for grid of height {params.height} and width {params.width}.")

            # Ensure key_x is between 1 and the wall position (key should be left of wall)
            if key_x <= 0 or key_x >= wall_pos:
                raise ValueError(f"Key position must be between 1 and the wall position = {wall_pos} for grid of height {params.height} and width {params.width}.")

            # Ensure key_y is between 1 and height-1 (within grid bounds)
            if key_y <= 0 or key_y >= params.height - 1:
                raise ValueError(f"Key position must be between 1 and {params.height - 2} for grid of height {params.height} and width {params.width}.")

        except ValueError as e:
            raise ValueError(f"Invalid predefined positions: {e}")
        key, _key = jax.random.split(key)
        keys = jax.random.split(_key, num=2)

        # Initialize the grid and place objects
        grid = room(params.height, params.width)
        grid = vertical_line(grid, wall_pos, 0, params.height, tile=TILES_REGISTRY[Tiles.WALL, Colors.GREY])
        grid = grid.at[door_pos, wall_pos].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, Colors.YELLOW])
        grid = grid.at[key_y, key_x].set(TILES_REGISTRY[Tiles.KEY, Colors.YELLOW])
        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        # Mask positions after the wall so the agent starts on the opposite side of the goal
        mask = coordinates_mask(grid, (params.height, wall_pos), comparison_fn=jnp.less)
        agent_coords = sample_coordinates(keys[0], grid, num=1, mask=mask)[0]

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[1]))

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state