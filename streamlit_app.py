# streamlit_app.py
import streamlit as st
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
import time
from io import BytesIO

st.set_page_config(layout="wide")
st.title("CartPole — Compare RL Agents")

MODEL_DIR = "models"

agent = st.sidebar.selectbox("Choose agent", 
    ["Random", "SARSA (loaded)", "REINFORCE (loaded)", "DQN (loaded)"])
episodes = st.sidebar.number_input("Episodes to run", min_value=1, max_value=1000, value=5)
render_fps = st.sidebar.slider("FPS", 1, 60, 20)
run_button = st.sidebar.button("Run")

# ------------------------------------------------
# 🔹 Load models (SARSA Q-table, REINFORCE policy, DQN)
# ------------------------------------------------
@st.cache_resource
def load_models():
    res = {}

    # ---------- SARSA ----------
    try:
        res['sarsa'] = np.load(f"{MODEL_DIR}/sarsa_Q.npy", allow_pickle=True)
    except:
        res['sarsa'] = None

    # ---------- REINFORCE ----------
    try:
        class PolicyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(4, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 2),
                    torch.nn.Softmax(dim=-1)
                )

            def forward(self, x):
                return self.layers(x)

        p = PolicyNet()
        p.load_state_dict(torch.load(f"{MODEL_DIR}/reinforce_policy.pth", map_location="cpu"))
        res['reinforce'] = p
    except:
        res['reinforce'] = None

    # ---------- DQN ----------
    try:
        class DQNNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(4,128), torch.nn.ReLU(),
                    torch.nn.Linear(128,128), torch.nn.ReLU(),
                    torch.nn.Linear(128,2)
                )

            def forward(self, x):
                return self.layers(x)

        dqn = DQNNet()
        dqn.load_state_dict(torch.load(f"{MODEL_DIR}/dqn_net.pth", map_location="cpu"))
        res['dqn'] = dqn
    except:
        res['dqn'] = None

    return res

models = load_models()

# ------------------------------------------------
# 🔹 Frame renderer
# ------------------------------------------------
def render_frame(env):
    frame = env.render()
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = Image.new("RGB", (600, 400))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ------------------------------------------------
# 🔹 RUN SIMULATION
# ------------------------------------------------
if run_button:
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    cols = st.columns([2, 1])
    img_holder = cols[0].empty()
    stat_box = cols[1].empty()

    all_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0

        while not done:

            # --- Choose action based on agent ---
            if agent == "Random":
                a = env.action_space.sample()

            elif agent == "SARSA (loaded)":
                Q = models['sarsa']
                if Q is None:
                    a = env.action_space.sample()
                else:
                    low = np.array([-4.8, -5, -0.418, -5])
                    high = np.array([4.8, 5, 0.418, 5])
                    bins = np.array([6, 12, 6, 12])
                    width = (high - low) / bins
                    digit = (np.clip(obs, low, high) - low) / width
                    s = tuple(np.clip(digit.astype(int), 0, bins - 1))
                    a = int(np.argmax(Q[s]))

            elif agent == "REINFORCE (loaded)":
                net = models['reinforce']
                if net is None:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        probs = net(obs_t).numpy()[0]
                    a = int(np.argmax(probs))

            elif agent == "DQN (loaded)":
                net = models['dqn']
                if net is None:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q = net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).numpy()[0]
                    a = int(np.argmax(q))

            # --- Step environment ---
            obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += reward

            # --- Render frame ---
            frame = render_frame(env)
            img_holder.image(frame, use_container_width=True)  # ✅ updated parameter
            stat_box.markdown(
                f"### Episode: {ep+1}/{episodes}\n"
                f"Reward: **{total_reward:.1f}**"
            )

            time.sleep(1.0 / render_fps)

        all_rewards.append(total_reward)

    env.close()

    st.success(f"Finished! Mean Reward = {np.mean(all_rewards):.2f}")
    st.bar_chart(all_rewards)
