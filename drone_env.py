import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

WIDTH = 600
HEIGHT = 400
DRONE_SPEED = 2
COLLISION_DISTANCE = 25

class DroneEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, width, height, obstacles):
        super().__init__()

        self.width = width
        self.height = height
        self.obstacles = obstacles

        self.observation_space = spaces.Box(
            low=0,
            high=max(width, height),
            shape=(4,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = np.random.randint(50, self.width - 50)
        self.y = np.random.randint(50, self.height - 50)

        self.target = (
            np.random.randint(50, self.width - 50),
            np.random.randint(50, self.height - 50)
        )

        self.steps = 0

        obs = np.array(
            [self.x, self.y, self.target[0], self.target[1]],
            dtype=np.float32
        )

        return obs, {}

    def step(self, action):

        moves = [
            (0, -1), (0, 1),
            (-1, 0), (1, 0),
            (-1, -1), (1, -1),
            (-1, 1), (1, 1)
        ]

        dx, dy = moves[int(action)]

        norm = np.hypot(dx, dy) if (dx != 0 or dy != 0) else 1

        dx = dx / norm * DRONE_SPEED
        dy = dy / norm * DRONE_SPEED

        self.x += dx
        self.y += dy

        self.x = np.clip(self.x, 0, self.width)
        self.y = np.clip(self.y, 0, self.height)

        dist = np.hypot(self.x - self.target[0], self.y - self.target[1])

        reward = -dist / 100

        done = False

        for obs in self.obstacles:

            ox, oy, w, h = obs

            if ox <= self.x <= ox + w and oy <= self.y <= oy + h:
                reward -= 100
                done = True

        if dist < 5:
            reward += 100
            done = True

        self.steps += 1

        if self.steps > 500:
            done = True

        obs = np.array(
            [self.x, self.y, self.target[0], self.target[1]],
            dtype=np.float32
        )

        return obs, reward, done, False, {}
        

class Drone:

    def __init__(self, drone_id, env, model):
        self.id = drone_id
        self.env = env
        self.model = model
        self.reset()

    def reset(self):
        obs, _ = self.env.reset()

        self.x = obs[0]
        self.y = obs[1]

        self.target = (obs[2], obs[3])

        self.collision = False
        self.path_length = 0

    def move(self, drones):

        obs_arr = np.array(
            [self.x, self.y, self.target[0], self.target[1]],
            dtype=np.float32
        )

        action, _ = self.model.predict(obs_arr, deterministic=True)

        new_obs, _, done, _, _ = self.env.step(int(action))

        self.x = new_obs[0]
        self.y = new_obs[1]

        self.collision = False

        for d in drones:

            if d.id != self.id:

                dist = math.hypot(self.x - d.x, self.y - d.y)

                if dist < COLLISION_DISTANCE:
                    self.collision = True

        self.path_length += 1

        if done:
            self.reset()