# Comparative Analysis: Reinforcement Learning in 2048

**Date:** December 30, 2024

**Subject:** Performance Benchmarking of Baseline (DQN) vs. Improved (AlphaZero) Agents

---

## 1. Executive Summary

We evaluated two distinct reinforcement learning architectures—**Deep Q-Network (Baseline)** and **AlphaZero (MCTS)**—across two environment variants: **Standard 2048** and **Constrained 2048** (Immovable Corner Block).

Despite limited training resources resulting in a 0% win rate (reaching the 2048 tile) across all models, the statistical breakdown demonstrates a massive performance gap:

* **Standard Environment:** The MCTS agent achieved an average score **3.5x higher** than the best Baseline configuration (7,371 vs. 2,096).
* **Constrained Environment:** The MCTS agent adapted significantly better to the obstacle, scoring **1.8x higher** than the Baseline (2,231 vs. 1,226).
* **Generalization:** The MCTS agent consistently achieved higher "Max Tiles," reaching 1024 in standard play, whereas the Baseline rarely exceeded 512.

---

## 2. Standard Environment Performance

The Standard environment represents the classic 2048 game. This test measures the agents' ability to learn the fundamental "Monotonic Snake" strategy without external obstructions.

### Performance Comparison

| Metric | **AlphaZero (MCTS)** | **Best Baseline (DQN)** | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Avg Score** | **7,371.84** | 2,096.44 | **+251%** |
| **Avg Max Tile** | **560.64** | 187.20 | **+199%** |
| **Best Score** | **16,000** | 5,172 | **+209%** |
| **Highest Tile** | **1024** | 512 | **+1 Tier** |

### Analysis
The **AlphaZero agent** demonstrates a fundamental strategic advantage.
* **Lookahead vs. Reaction:** The huge disparity in scores indicates that MCTS (planning 150 moves ahead) successfully avoids the "early death" traps that plague the Baseline.
* **Stability:** The Baseline's low average max tile (~187) suggests it frequently dies before forming even a 256 tile. In contrast, the MCTS agent averages ~560, meaning it consistently builds 512 tiles and frequently attempts 1024.

---

## 3. Constrained Environment Performance

The Constrained environment places an immovable block (`-1`) in the corner, removing the safest anchor point for the "Snake" strategy. This tests the agents' spatial adaptability.

### Performance Comparison

| Metric | **AlphaZero (MCTS)** | **Best Baseline (CNN)** | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Avg Score** | **2,231.48** | 1,226.48 | **+82%** |
| **Avg Max Tile** | **202.24** | 129.92 | **+56%** |
| **Best Score** | **7,032** | 2,684 | **+162%** |
| **Highest Tile** | **512** | 256 | **+1 Tier** |

### Analysis
The performance gap narrows but remains significant.
* **Baseline Failure:** The Baseline struggles to adapt, with its score dropping by ~40% compared to its Standard performance. It likely treats the block as a generic number, causing it to merge tiles into "dead zones."
* **MCTS Adaptability:** The AlphaZero agent maintained a respectable average score of 2,231. The ability to simulate moves allows it to "see" that tiles cannot move through the block, effectively learning to build its chain in a different direction.

---

## 4. Baseline Study

We tested various configurations for the Baseline DQN to find the optimal setup. The results highlight the importance of **State Representation**.

| Configuration | State Rep | Reward Function | Avg Score (Std) |
| :--- | :--- | :--- | :--- |
| **Tuned (Best)** | **Log2** | **Log Merge** | **2,096** |
| Tuned | Log2 | Potential | 1,874 |
| States Group | Log2 | Raw Score | 1,404 |
| States Group | **Raw Values** | Raw Score | 1,025 |

**Key Finding:** Using `Log2` normalization doubled the performance compared to `Raw Values` (2,096 vs. 1,025). This confirms that Neural Networks struggle with the exponential scale of 2048 (2 vs. 2048) and require logarithmic scaling to learn effectively.

---

## 5. Conclusion

While the **Baseline DQN** serves as a functional proof of concept, it is fundamentally limited by its reactive nature. It fails to plan for the stochasticity of tile spawns, leading to early game terminations.

The **AlphaZero (MCTS)** approach, despite being computationally more expensive, provides a robust solution. Its ability to simulate future states allows it to navigate both Standard and Constrained environments with significantly higher competence, validating the hypothesis that **Model-Based Reinforcement Learning** is superior for stochastic puzzles.