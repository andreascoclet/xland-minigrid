

---

# DoorKeyDeterministic Environment in MiniGrid

## Overview

The `DoorKeyDeterministic` environment is part of the `xminigrid` library. It is a deterministic version of the `DoorKey` environment, where an agent must find a key, unlock a door, and exit. The environment allows you to specify the positions of the door, wall, and key, making it deterministic and customizable.

---

## Installation

To use the `DoorKeyDeterministic` environment, ensure you have the required libraries installed. Run the following command to install `jax` and `xminigrid`:

```bash
pip install .....
```

---

## Usage

### Step 1: Initialize JAX Key

Start by initializing a JAX random key. This key is used for seeding and splitting. Use the `jax.random.key` function to create the key, and then split it into two keys: one for resetting the environment and one for further use.

---

### Step 2: Define Positions

Next, define the positions for the door, wall, and key. The `pos` argument should always be a list containing four elements in the format `[door_pos, wall_pos, key_x, key_y]`. If you don't specify the `pos` argument, the environment will default to behavior similar to the original `DoorKey` environment, where the positions are automatically assigned.

---

### Step 3: Create the Environment

Use the `xminigrid.make` function to create the `DoorKeyDeterministic` environment. Pass the environment name (e.g., `"MiniGrid-DoorKeyDet-16x16"`) to the `make` function. After creating the environment, assign the `pos` list (defined in Step 2) to the `env.pos` attribute.

---

### Step 4: Reset the Environment

Reset the environment using the `reset` method. This method is JIT-compatible, meaning it can be optimized for efficient computation. Use the `jax.jit` function to compile the `reset` method, and pass the environment parameters and reset key to initialize the environment.

---

### Step 5: Render the Environment

To visualize the environment, use the `render` method. This method generates an image of the current state of the environment. You can display this image using a plotting library like `matplotlib`. Define a helper function (e.g., `show_img`) to render and display the image.

---

### Step 6: Perform Actions

Interact with the environment using the `step` method. Pass the environment parameters, the current timestep, and an action (e.g., move forward) to the `step` method. The environment will return a new timestep, which you can render and display using the `render` method.

---

## Default Behavior

If you don’t provide the `pos` argument when creating the environment, it will default to behavior similar to the original `DoorKey` environment. In this case, the positions of the door, wall, and key are automatically assigned.

---



Example Code

Here’s a complete example of how to use the DoorKeyDeterministic environment:

python
import jax
import xminigrid
from matplotlib import pyplot as plt

# Step 1: Initialize JAX key
key = jax.random.key(0)
key, reset_key = jax.random.split(key)

# Step 2: Define positions
door_pos = 8
wall_pos = 8
key_x = 4
key_y = 14
pos = [door_pos, wall_pos, key_x, key_y]  # Always a list

# Step 3: Create the environment
env, env_params = xminigrid.make("MiniGrid-DoorKeyDet-16x16")
env.pos = pos

# Step 4: Reset the environment
timestep = jax.jit(env.reset)(env_params, reset_key)

# Step 5: Render the environment
def show_img(img, dpi=64):
    plt.figure(dpi=dpi)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

show_img(env.render(env_params, timestep), dpi=64)

# Step 6: Perform actions
action = 2  # Example action (e.g., move forward)
timestep = jax.jit(env.step)(env_params, timestep, action)
show_img(env.render(env_params, timestep), dpi=64)
