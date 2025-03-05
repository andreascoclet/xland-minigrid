# DoorKeyDeterministic Environment in MiniGrid

## Overview

The DoorKeyDeterministic environment is part of the xminigrid library. It is a deterministic version of the DoorKey environment, where an agent needs to find a key, unlock a door, and exit. The environment allows setting specific positions for the door, wall, and key.

## How to Use
	### 1.	Set Up Environment
Ensure you have the required libraries installed, including jax and xminigrid.
        pip install...
	### 2.	Initialize JAX Key
A random key is used for seeding and splitting.
    import jax
    import xminigrid

    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
	### 3.	Define Positions
You can specify the positions for the door, wall, and key. The pos should always be a list containing four elements, in the format [door_pos, wall_pos, key_x, key_y]. If not specified, the environment defaults to behavior similar to the original DoorKey environment.
    door_pos = 8
    wall_pos = 8
    key_x = 4
    key_y = 14
    pos = [door_pos, wall_pos, key_x, key_y]  # Always a list
	### 4.	Create the Environment
Use the xminigrid.make function to create the environment and assign the defined positions.
    env, env_params = xminigrid.make("MiniGrid-DoorKeyDet-16x16")
    env.pos = pos
	### 5.	JIT-Compatible Reset and Step Methods
Both reset and step methods are JIT-compatible for efficient computation. The environment can be reset using a JAX-compiled function.
    timestep = jax.jit(env.reset)(env_params, reset_key)
    show_img(env.render(env_params, timestep), dpi=64)

## Default Behavior

If you don’t provide the pos argument, the environment will default to behavior similar to the DoorKey environment, where positions of the door, wall, and key are automatically assigned.
