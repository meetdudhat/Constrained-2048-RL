import gymnasium
import time
from env_2048 import Constrained2048Env

env = Constrained2048Env(render_mode="human")

print("Environment created.")
print(f"Action space: {env.action_space}")
# print(f"Observation Space: {env.observation_space}")

print("\n--- STARTING RANDOM AGENT DEMO ---\n")
observation, info = env.reset()
terminated = False
truncated = False

total_reward = 0
step_count = 0

while not (terminated or truncated):
    
    # Picks random action
    action = env.action_space.sample()
    
    print(f"\n--- Step {step_count} ---")
    print(f"Action taken: {['Up', 'Right', 'Down', 'Left'][action]}")
    
    # performs the action in the environment
    observation, reward, terminated, truncated, _ = env.step(action)
    
    print(f"Reward received: {reward}")
    total_reward += reward
    step_count += 1
    
    time.sleep(1)
    

print("\n--- EPISODE/GAME FINISHED ---")
print(f"Total steps: {step_count}")
print(f"Total score: {total_reward}")