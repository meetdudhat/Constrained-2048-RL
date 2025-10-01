# Constrained-2048-RL
Reinforcement Learning for 2048 with Constraints: Handling Immovable and Non-Combining cell

# Team Members
| Name | Contact|
|----------|----------|
| Arya Shah |Aryashah@cmail.carleton.ca|
| Ayman Madani | aymanmadani@cmail.carleton.ca |
| Jason Xu | jasonxu@cmail.carleton.ca |
| Meetkumar Dudhat | meetkumardudhat@cmail.carleton.ca |
| Sachin Bansal | sachinbansal@cmail.carleton.ca |

# Milestones

### **Oct 15 – Environment and Baselines (Setup)** ⚙️
- Implement the constrained 2048 environment (with immovable/non-combining tiles).
- Add a few simple tests for key game logic like merging, handling invalid moves, and termination conditions.
- Implement a simple baseline agent (e.g., random moves, or an agent maximizing the immediate merge score).

---

### **Oct 29 – First RL Agent** 🤖
- Implement and train an initial RL agent on the *standard* 2048 game to verify that the learning process works.
- Extend and adapt this agent to handle the unique challenges of the constrained environment.

---

### **Nov 12 – Improvements & Analysis** 📈
- Experiment with different reward structures (e.g., adding penalties for invalid moves, rewarding empty cells).
- Collect performance results from the new agent and compare them against the initial baseline.

---

### **Nov 26 – Final Results & Deliverables** 📦
- Tune hyperparameters and finalize the best-performing agent.
- Run simple robustness checks to test the agent's stability.
- Prepare final deliverables:
    - Final model checkpoint and configuration files.
    - A brief `README.md` explaining how to run the project.
    - Demo Video.
    - A concise final report summarizing the approach and findings.
