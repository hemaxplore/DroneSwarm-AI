# multi_drone_pytorch.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import math
import time
import gymnasium as gym  # Use Gymnasium
from gymnasium import spaces
from stable_baselines3 import DQN
import torch

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(layout="wide")
st.title("Multi-Drone RL Path Planning (PyTorch + Gymnasium)")

# -------------------------------
# Sidebar Controls
# -------------------------------
NUM_DRONES = st.sidebar.slider("Number of Drones", 1, 5, 3)
NUM_OBSTACLES = st.sidebar.slider("Number of Obstacles", 1, 5, 3)
DRONE_SPEED = st.sidebar.slider("Drone Speed", 1, 5, 2)
FRAME_DELAY = st.sidebar.slider("Frame Delay (s)", 0.05, 1.0, 0.2)

WIDTH, HEIGHT = 600, 400
DRONE_RADIUS = 10
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 80, 80
COLLISION_DISTANCE = 25

# -------------------------------
# RL Environment
# -------------------------------
class DroneEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, width, height, obstacles, drone_speed=DRONE_SPEED):
        super().__init__()
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.drone_speed = drone_speed
        # Observation: x, y, target_x, target_y
        self.observation_space = spaces.Box(low=0, high=max(width,height), shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)
        self.steps = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.random.randint(50, self.width-50)
        self.y = np.random.randint(50, self.height-50)
        self.target = (np.random.randint(50, self.width-50), np.random.randint(50, self.height-50))
        self.steps = 0
        obs = np.array([self.x, self.y, self.target[0], self.target[1]], dtype=np.float32)
        return obs, {}

    def step(self, action):
        moves = [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(1,-1),(-1,1),(1,1)]
        dx, dy = moves[int(action)]  # Convert to scalar
        norm = np.hypot(dx, dy) if (dx != 0 or dy != 0) else 1
        dx, dy = dx / norm * self.drone_speed, dy / norm * self.drone_speed
        self.x += dx
        self.y += dy
        self.x = np.clip(self.x, 0, self.width)
        self.y = np.clip(self.y, 0, self.height)

        # Reward
        dist_to_target = np.hypot(self.x - self.target[0], self.y - self.target[1])
        reward = -dist_to_target / 100.0
        done = False

        # Obstacle collision
        for obs in self.obstacles:
            ox, oy, w, h = obs
            if ox <= self.x <= ox + w and oy <= self.y <= oy + h:
                reward -= 100
                done = True

        # Target reached
        if dist_to_target < 5:
            reward += 100
            done = True

        self.steps += 1
        if self.steps > 500:
            done = True

        obs = np.array([self.x, self.y, self.target[0], self.target[1]], dtype=np.float32)
        return obs, reward, done, False, {}  # Gymnasium: step returns 5-tuple

# -------------------------------
# Drone class
# -------------------------------
class Drone:
    def __init__(self, drone_id, env, model):
        self.id = drone_id
        self.env = env
        self.model = model
        self.reset()

    def reset(self):
        obs, _ = self.env.reset()
        self.x, self.y = obs[0], obs[1]
        self.target = (obs[2], obs[3])
        self.collision = False
        self.path_length = 0
        self.closest_obstacle = None
        self.path = [(self.x, self.y)]

    def move(self, drones, obstacles):
        # Closest obstacle
        min_dist = float('inf')
        closest = None
        for i, obs in enumerate(obstacles):
            ox, oy, w, h = obs
            cx, cy = ox + w/2, oy + h/2
            dist = math.hypot(self.x - cx, self.y - cy)
            if dist < min_dist:
                min_dist = dist
                closest = i
        self.closest_obstacle = closest

        # RL action
        obs_arr = np.array([self.x, self.y, self.target[0], self.target[1]], dtype=np.float32)
        action, _ = self.model.predict(obs_arr, deterministic=True)
        new_obs, reward, done, _, _ = self.env.step(int(action))  # fix TypeError here
        self.x, self.y = new_obs[0], new_obs[1]
        self.path.append((self.x, self.y))

        # Collision with other drones
        self.collision = False
        for d in drones:
            if d.id != self.id and math.hypot(self.x - d.x, self.y - d.y) < COLLISION_DISTANCE:
                self.collision = True

        self.path_length += 1
        if done:
            self.reset()

    def draw(self, frame):
        color = (0,0,255) if self.collision else (255,0,0)
        cv2.circle(frame,(int(self.x), int(self.y)),DRONE_RADIUS,color,-1)
        cv2.circle(frame,(int(self.target[0]), int(self.target[1])),5,(0,255,0),-1)

    def info(self):
        return {"ID": self.id, "X": int(self.x), "Y": int(self.y),
                "Target_X": int(self.target[0]), "Target_Y": int(self.target[1]),
                "Collision": self.collision,
                "Path Length": self.path_length,
                "Closest Object": self.closest_obstacle}

# -------------------------------
# Obstacles & Environment
# -------------------------------
obstacles = [(np.random.randint(50, WIDTH-OBSTACLE_WIDTH-50),
              np.random.randint(50, HEIGHT-OBSTACLE_HEIGHT-50),
              OBSTACLE_WIDTH, OBSTACLE_HEIGHT) for _ in range(NUM_OBSTACLES)]

env_dummy = DroneEnv(WIDTH, HEIGHT, obstacles)

# -------------------------------
# PyTorch DQN Model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if 'model' not in st.session_state:
    model = DQN("MlpPolicy", env_dummy, verbose=0, tensorboard_log=None, device=device)
    model.learn(total_timesteps=5000)  # small training for demo
    st.session_state.model = model
else:
    model = st.session_state.model

# -------------------------------
# Initialize Drones
# -------------------------------
if 'drones' not in st.session_state or st.sidebar.button("Restart Simulation"):
    st.session_state.drones = [Drone(i, DroneEnv(WIDTH, HEIGHT, obstacles), model) for i in range(NUM_DRONES)]

# -------------------------------
# Placeholders
# -------------------------------
frame_placeholder = st.empty()
map_placeholder = st.empty()
info_placeholder = st.empty()
plot_placeholder = st.empty()
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()

# -------------------------------
# Simulation loop
# -------------------------------
while True:
    if time.time() - st.session_state.last_time < FRAME_DELAY:
        time.sleep(0.01)
        continue
    st.session_state.last_time = time.time()

    frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)*255

    # Draw obstacles
    for i, obs in enumerate(obstacles):
        x, y, w, h = obs
        cv2.rectangle(frame, (x, y), (x+w, y+h), (150,150,150), -1)
        cv2.putText(frame, f"O{i}", (x+5,y+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    # Move drones
    for drone in st.session_state.drones:
        drone.move(st.session_state.drones, obstacles)
        drone.draw(frame)

    # Display frame
    frame_placeholder.image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), use_column_width=True)

    # Mini-map
    map_fig = go.Figure()
    for i, obs in enumerate(obstacles):
        x, y, w, h = obs
        map_fig.add_trace(go.Scatter(x=[x,x+w,x+w,x,x], y=[y,y,y+h,y+h,y],
                                     fill="toself", fillcolor="gray", line=dict(color="gray"), name=f"Obj {i}"))
    for drone in st.session_state.drones:
        color = 'red' if drone.collision else 'blue'
        if len(drone.path) > 1:
            xs, ys = zip(*drone.path)
            map_fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=color, dash='dash'), name=f"D{drone.id} Path"))
        map_fig.add_trace(go.Scatter(x=[drone.x], y=[drone.y], mode='markers+text',
                                     marker=dict(color=color,size=12), text=[f"D{drone.id}"], textposition="top center"))
    map_fig.update_layout(xaxis=dict(range=[0,WIDTH]), yaxis=dict(range=[0,HEIGHT], autorange='reversed'),
                          width=600,height=400, showlegend=True)
    map_placeholder.plotly_chart(map_fig,use_container_width=True)

    # Drone info
    info_df = pd.DataFrame([d.info() for d in st.session_state.drones])
    info_placeholder.dataframe(info_df)

    # Collision & Path Length Plot
    plot_fig = go.Figure()
    plot_fig.add_trace(go.Bar(x=[d.id for d in st.session_state.drones], y=[1 if d.collision else 0 for d in st.session_state.drones], name="Collision"))
    plot_fig.add_trace(go.Bar(x=[d.id for d in st.session_state.drones], y=[d.path_length for d in st.session_state.drones], name="Path Length"))
    plot_fig.update_layout(barmode='group', title="Drone Collisions & Path Lengths")
    plot_placeholder.plotly_chart(plot_fig,use_container_width=True)