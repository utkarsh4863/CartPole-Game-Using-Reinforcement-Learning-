# CartPole Game Using Reinforcement Learning

This project demonstrates and compares different Reinforcement Learning (RL) algorithms on the classic **CartPole-v1** environment from OpenAI Gymnasium.  
You can train, evaluate, and visualize agents directly in a Jupyter Notebook, and also run a **Streamlit app** to see the gameplay in real-time.

## 🚀 Features

- Implemented Agents:
  - **Random Policy**  
  - **SARSA (Tabular)**  
  - **REINFORCE (Policy Gradient)**  
  - **DQN (Deep Q-Network)**

- Compare rewards across agents  
- Visualize gameplay in **Streamlit app**  
- Models saved for direct use (no retraining required)  

---


## 🛠 Installation & Setup (Windows)

```bash
# Clone the repository
git clone https://github.com/utkarsh4863/CartPole-Game-Using-Reinforcement-Learning-.git
cd "CartPole game project using RL"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Run Jupyter Notebook (optional)
jupyter notebook "CartPole game using  RL.ipynb"

# Or run Streamlit app to see gameplay
streamlit run streamlit_app.py

## 📓 Usage (Windows)

```bash
# 1. Run Jupyter Notebook to train or evaluate agents
jupyter notebook "CartPole game using  RL.ipynb"

# - Train SARSA, REINFORCE, DQN agents
# - Compare their rewards
# - Save trained models in the models/ folder

# 2. Run Streamlit app to see real-time gameplay
streamlit run streamlit_app.py

# Sidebar options in Streamlit app:
# - Choose agent (Random, SARSA, REINFORCE, DQN)
# - Set number of episodes
# - Adjust FPS for rendering
# Click 'Run' to start the simulation
# Gameplay will display live reward stats and a reward bar chart

## 📁 Folder Structure
CartPole game project using RL/
│── CartPole game using RL.ipynb
│── streamlit_app.py
│── models/
│ ├── sarsa_Q.npy
│ ├── reinforce_policy.pth
│ └── dqn_net.pth
│── requirements.txt
│── .gitignore
│── README.md

💻 License
# This project is open-source for educational purposes.


