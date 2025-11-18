# 🚀 CartPole Game Using Reinforcement Learning

This project demonstrates and compares different Reinforcement Learning (RL) algorithms on the classic **CartPole-v1** environment from OpenAI Gymnasium.  
You can train, evaluate, and visualize agents directly in a Jupyter Notebook, and also run a **Streamlit app** to see the gameplay in real-time.

The project is designed for users who want to:
- Compare different RL agents (Random, SARSA, REINFORCE, DQN)  
- Visualize agent performance  
- Understand reinforcement learning in a practical, interactive way  

---

## 🧩 Key Components

- **Implemented Agents**:
  - Random Policy  
  - SARSA (Tabular)  
  - REINFORCE (Policy Gradient)  
  - DQN (Deep Q-Network)  

- **Streamlit App**: Visualizes real-time gameplay with reward statistics  
- **Jupyter Notebook**: Train and evaluate agents  
- **Saved Models**: Pre-trained models for direct use without retraining  

---

## 🎯 Objective

To build an interactive and visual tool to **compare RL agents**, understand their performance, and learn reinforcement learning concepts practically.  

---

## 🧠 Tech Stack

| Category       | Tools |
|----------------|-------|
| **Language**   | Python 🐍 |
| **Frameworks** | Gymnasium, Streamlit, PyTorch |
| **Libraries**  | numpy, matplotlib |
| **Visualization** | Streamlit / Matplotlib |
| **Environment** | OpenAI Gymnasium CartPole-v1 |

---

## ⚙️ Installation & Setup (Windows)


### Clone the repository
```bash
git clone https://github.com/utkarsh4863/CartPole-Game-Using-Reinforcement-Learning-.git
cd "CartPole game project using RL"
```

### Create a virtual environment
```bash
python -m venv venv
```

### Activate the virtual environment
```bash
venv\Scripts\activate
```

### Install required dependencies
```bash
pip install -r requirements.txt
```
### Run Jupyter Notebook (optional)
```bash
jupyter notebook "CartPole game using  RL.ipynb"
```

### Or run Streamlit app to see real-time gameplay
```bash
streamlit run streamlit_app.py
```
---


## 📓 Usage (Windows)
```

# 1. Jupyter Notebook
# - Train SARSA, REINFORCE, DQN agents
# - Compare their rewards
# - Save trained models in the models/ folder
jupyter notebook "CartPole game using  RL.ipynb"

# 2. Streamlit App
# - Choose agent (Random, SARSA, REINFORCE, DQN)
# - Set number of episodes
# - Adjust FPS for rendering
# - Click 'Run' to start simulation
# - Live reward stats and reward bar chart displayed
streamlit run streamlit_app.py
```
---




### 📁 Folder Structure
```

# CartPole game project using RL/
# │── CartPole game using  RL.ipynb
# │── streamlit_app.py
# │── models/
# │     ├── sarsa_Q.npy
# │     ├── reinforce_policy.pth
# │     └── dqn_net.pth
# │── requirements.txt
# │── .gitignore
# │── README.md
```

---

### ⚡ Notes
```
# The models/ folder contains pre-trained agents for direct use in Streamlit.
# If you want to retrain models, run the notebook and save the models in the same folder.
# Streamlit app depends on the models; if models are missing, the agents will run randomly.
```
---

## 🧩 Author

**Utkarsh Kashyap**



