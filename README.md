# 🚀 CartPole Game Using Reinforcement Learning

This project showcases and compares several Reinforcement Learning (RL) algorithms using the classic **CartPole-v1** environment from [OpenAI Gymnasium](https://gymnasium.farama.org/). Train, evaluate, and visualize agents easily within a Jupyter Notebook or interactively with a **Streamlit app** for real-time gameplay.

---

## 🧩 Key Features

- **Implemented Agents**
  - 🔹 Random Policy  
  - 🔹 SARSA (Tabular)  
  - 🔹 REINFORCE (Policy Gradient)  
  - 🔹 DQN (Deep Q-Network)  

- **Streamlit App**  
  - Real-time gameplay
  - Live reward statistics & visualizations  

- **Jupyter Notebook**  
  - Train & evaluate agents step-by-step  
  - Save/load trained models easily  

- **Pre-trained Models**  
  - No need to retrain agents every time  
  - Models ready for use in `models/` folder  

---

## 🎯 Objective

Interactive visual playground for **comparing RL agents**, understanding their strengths, and learning reinforcement learning by doing.

---

## 🌐 Live Demo

🎉 **See it in action on Streamlit:**  
[Open the CartPole RL Game live!](https://mvxe7kkcg2fpyiuwpfmpkf.streamlit.app/)

---

## 🧠 Tech Stack

| Category          | Tools                                 |
|-------------------|---------------------------------------|
| **Language**      | Python 🐍                             |
| **Frameworks**    | Gymnasium, Streamlit, PyTorch         |
| **Libraries**     | numpy, matplotlib                     |
| **Visualization** | Streamlit, Matplotlib                 |
| **Environment**   | OpenAI Gymnasium CartPole-v1          |

---

## ⚙️ Installation & Setup (Windows)

1. **Clone the repository**
    ```bash
    git clone https://github.com/utkarsh4863/CartPole-Game-Using-Reinforcement-Learning-.git
    cd "CartPole game project using RL"
    ```

2. **Create a virtual environment**
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**
    ```bash
    venv\Scripts\activate
    ```

4. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Jupyter Notebook for training & evaluation**
    ```bash
    jupyter notebook "CartPole game using  RL.ipynb"
    ```

6. **Or launch the Streamlit app for gameplay visualization**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 📓 How To Use (Windows)

### 1. Jupyter Notebook

- Train SARSA / REINFORCE / DQN agents
- Compare performance and visualize reward curves
- Save trained models to `models/`

    ```bash
    jupyter notebook "CartPole game using  RL.ipynb"
    ```

### 2. Streamlit App

- Select agent: Random, SARSA, REINFORCE, DQN
- Set episodes and adjust FPS for smoother gameplay
- Click 'Run' to start simulation
- See live rewards and dynamic reward bar chart

    ```bash
    streamlit run streamlit_app.py
    ```

---

## 📁 Folder Structure

```
CartPole game project using RL/
│── CartPole game using  RL.ipynb
│── streamlit_app.py
│── models/
│     ├── sarsa_Q.npy
│     ├── reinforce_policy.pth
│     └── dqn_net.pth
│── requirements.txt
│── .gitignore
│── README.md
```

---

## ⚡ Notes

- The `models/` directory includes pre-trained agents ready for use in Streamlit.
- If you wish to retrain agents, run the notebook and overwrite/save models in the same folder.
- If models are absent, Streamlit runs agents with random policy by default.

---

## 🤝 Author

Developed & maintained by **Utkarsh Kashyap**  
Feel free to connect and contribute! 🚀

---
