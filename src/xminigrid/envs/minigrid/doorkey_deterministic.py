from __future__ import annotations

import jax
import jax.numpy as jnp

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
    def _generate_problem(self, params: EnvParams, key: jax.Array)-> State[EnvCarry]:
        if key.shape[0] == 1:  # If key has only one element, behave like DoorKey
            return super()._generate_problem(params, key)
        
        door_pos, wall_pos, key_x, key_y, seed = key

        seed = jax.random.PRNGKey(seed.astype(jnp.uint32))
        seed, _seed = jax.random.split(seed)
        seeds = jax.random.split(_seed, num=2)  # For randomizing agent position and direction

        grid = room(params.height, params.width)
        grid = vertical_line(grid, wall_pos, 0, params.height, tile=TILES_REGISTRY[Tiles.WALL, Colors.GREY])
        grid = grid.at[door_pos, wall_pos].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, Colors.YELLOW])
        grid = grid.at[key_y, key_x].set(TILES_REGISTRY[Tiles.KEY, Colors.YELLOW])
        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        # Mask positions after the wall so the agent starts on the opposite side of the goal
        mask = coordinates_mask(grid, (params.height, wall_pos), comparison_fn=jnp.less)
        agent_coords = sample_coordinates(seeds[0], grid, num=1, mask=mask)[0]

        agent = AgentState(position=agent_coords, direction=sample_direction(seeds[1]))
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