# COMP 4010 Intro to Reinforcement Learning

 **Group:** 19

 **Date:** November 15, 2025

---
### Ayman
* **Past Two Weeks:** Implementated the baseline DQN agent using stable-baselines3. Engineered a key feature to normalize the game state (using log2), which significantly improves learning stability. Developed a robust logging solution to track agent performance (score, max_tile) in TensorBoard.
* **Next Two Weeks:** Will begin the Final Analysis. This involves running the long-duration training experiments for all agent configurations and using the resulting data to write the final project report.

---
### Meet
* **Past Two Weeks:** Successfully implemented the first advanced reward structure. This involved modifying the game engine to report tile merges and updating the environment to reward the agent based on the log2 value of newly merged tiles, aligning the agent's reward with its normalized state.
* **Next Two Weeks:** Will support the final analysis phase by assisting with experiment runs and contributing to the final report/presentation.

---
### Sachin
* **Past Two Weeks:** Successfully created a new, standalone evaluate.py script. This essential tool loads a final, saved agent and runs it deterministically for 100+ episodes to gather hard statistics (e.g., average score, max tile, win rate), which are crucial for the final report.
* **Next Two Weeks:** Will support the final analysis phase by assisting with experiment runs and contributing to the final report/presentation.

---
### Jason
* **Past Two Weeks:** Successfully implemented the final, most complex reward structure. This version adds a "potential" factor (rewarding empty cells) and an "efficiency" factor (penalizing non-merge moves) to the agent's logic, teaching it to balance progress with survival.
* **Next Two Weeks:** Will support the final analysis phase by assisting with experiment runs and contributing to the final report/presentation.

---
### Arya
* **Past Two Weeks:**  Successfully implemented the final testing harness. Upgraded the main training script to dynamically support switching between the standard and constrained (blocked tile) game environments, which allows us to test our final agent's adaptability.
* **Next Two Weeks:** Will support the final analysis phase by assisting with experiment runs and contributing to the final report/presentation.

---
### Team Goals for Next Milestone and Current Milestone Status

The baseline agent development and experimental framework milestones are complete. Our team has successfully built a fully modular and robust platform for training, logging, and evaluating multiple sophisticated RL agents on 2048.

The immediate goal for the next milestone is the Final Analysis. The entire team will support the execution of the final, long-duration training experiments. The data from these runs will be collected, analyzed, and synthesized into the final project report and presentation.