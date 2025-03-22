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


class DoorKeyFullyDeterministic(DoorKey):
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
        # self.pos = pos  # Store deterministic position

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        """Generates the problem using a predefined `pos` instead of random sampling."""
        if params.det_positions is None:
            #raise ValueError("The position attribute 'pos' is not set. Provide `pos` when creating the environment.")
            return super()._generate_problem(params, key)  # ✅ Use normal random behavior if pos isn't set
        # Unpack the predefined positions
        start_y, start_x, direction_start, door_pos, wall_pos, key_x, key_y = params.det_positions
        #print(type(door_pos))

        try:
            # Validate predefined positions
            if start_y <= 0 or start_y >= params.height - 1:
                raise ValueError(f"Agent start_y -coordinate must be between 1 and {params.height - 2}.")

            if start_x <= 0 or start_x >= wall_pos:
                raise ValueError(f"Agent start_y -coordinate must be between 1 and {wall_pos - 1} (to the left of the wall).")

            if door_pos <= 0 or door_pos >= params.height - 1:
                raise ValueError(f"Door position must be between 1 and {params.height - 2}.")

            if wall_pos <= 1 or wall_pos >= params.width - 2:
                raise ValueError(f"Wall position must be between 2 and {params.width - 3}.")
    
            if key_y <= 0 or key_y >= params.height - 1:
                raise ValueError(f"Key y-coordinate must be between 1 and {params.height - 2}.")

            if key_x <= 0 or key_x >= wall_pos:
                raise ValueError(f"Key x-coordinate must be between 1 and {wall_pos - 1} (to the left of the wall).")
            
            # Ensure agent's start position is not the same as the key position
            if (start_y == key_y) and (start_x == key_x):
                raise ValueError(f"Agent start position ({start_y}, {start_x}) cannot be the same as key position ({key_y}, {key_x}).")

            if direction_start not in [0, 1, 2, 3]:
                raise ValueError(f"Invalid agent direction: {direction_start}. Must be 0 (right), 1 (down), 2 (left), or 3 (up).")
        except ValueError as e:
            raise ValueError(f"Invalid predefined positions: {e}")


        # Initialize grid and place objects
        grid = room(params.height, params.width)
        grid = vertical_line(grid, wall_pos, 0, params.height, tile=TILES_REGISTRY[Tiles.WALL, Colors.GREY])
        grid = grid.at[door_pos, wall_pos].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, Colors.YELLOW])
        grid = grid.at[key_y, key_x].set(TILES_REGISTRY[Tiles.KEY, Colors.YELLOW])
        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        # Set the fully deterministic agent position and direction
        agent = AgentState(position=jnp.array([start_y, start_x]), direction=jnp.array(direction_start))

        # Return the fully deterministic state
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