import gymnasium
from gymnasium import spaces
import numpy as np

class Log2Wrapper(gymnasium.ObservationWrapper):
    """
    Normalizes the observation space by converting raw tile values to their log2 representation.

    Rationale:
        In 2048, values scale exponentially (2^1 to 2^16). Feeding raw values 
        (e.g., 2 vs 4096) into a neural network causes a large magnitude variance, 
        leading to unstable gradients and slow convergence.
        
        This wrapper compresses the state space to a linear scale:
        - 2    -> 1.0
        - 2048 -> 11.0
        This allows the optimizer to treat the difference between 2->4 and 1024->2048
        as numerically equivalent steps.

    Args:
        env (gymnasium.Env): The environment to wrap.
        policy_type (str): 'mlp' or 'cnn'. Determines the output shape.
                           'cnn' enforces a (Channel, Height, Width) format required
                           by PyTorch Conv2d layers.
    """
    def __init__(self, env, policy_type="mlp"):
        super().__init__(env)
        
        if policy_type not in ["mlp", "cnn"]:
            raise ValueError(f"Unknown policy_type: {policy_type}. Must be 'mlp' or 'cnn'.")
        
        self.policy_type = policy_type
        
        # We get the 'low' and 'hight value from the original env
        low = self.observation_space.low.min()
        high = 32.0 # Sufficient upper bound for 2048 (2^32)
        
        # Adjust input shape based on network architecture requirements
        if self.policy_type == "cnn":
            # PyTorch Conv2d requires (Channels, Height, Width)
            self.output_shape = (1, 4, 4)
        else:
            # MLPs expect a flat or standard grid input
            self.output_shape = (4, 4)
        
        self.observation_space = spaces.Box(
            low=low, 
            high=high,
            shape=self.output_shape,
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Transforms the observation matrix.
        
        Logic:
            1. obs > 0  -> log2(obs)
            2. obs == 0 -> 0.0 (Empty tiles remain zero)
            3. obs == -1 -> -1.0 (Preserve special flags/blocking tiles)
        """
        processed_obs = np.zeros_like(obs, dtype=np.float32)

        # Vectorized log2 calculation avoiding div-by-zero errors (log2(0) = -inf)
        positive_mask = (obs > 0)
        processed_obs[positive_mask] = np.log2(obs[positive_mask])
        
        # Explicitly preserve special state markers (e.g., -1 for obstacles)
        # which would otherwise be lost or malformed by log logic.
        processed_obs[obs == -1] = -1.0
        
        # --- TEMPORARY DEBUG PRINT ---
        # prints 1 in 10,000 times to avoid flooding the console
        # if np.random.rand() < 0.0001:
        #     print("\n" + "="*40)
        #     print("   LOG2WRAPPER: SPOTTED A TRANSFORM")
        #     print("--- RAW (from env) ---")
        #     print(obs)
        #     print("--- LOG2 (to agent) ---")
        #     print(processed_obs.astype(np.float32))
        #     print("="*40 + "\n")
        # --- END DEBUG ---
        
        # Ensure correct channel dimensions for the selected policy
        return processed_obs.reshape(self.output_shape).astype(np.float32)