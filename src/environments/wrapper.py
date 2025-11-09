import gymnasium
from gymnasium import spaces
import numpy as np

class Log2Wrapper(gymnasium.ObservationWrapper):
    """
    Wraps the environment to apply log2 to the observation (board).
    This normalization is crucial for the neural network to
    understand the exponential nature of the game (2, 4, 8, ...).
    """
    def __init__(self, env):
        super().__init__(env)
        
        # We get the 'low' value from the original env,
        # which will be 0 for standard and -1 for constrained.
        low = self.env.observation_space.low.min()
        
        # The new observation space will be floats
        self.observation_space = spaces.Box(
            low=low, 
            high=32, # log2(2^32) is more than enough
            shape=(4, 4), 
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Applies the log2 transformation to the observation.
        """
        # We start with a float array of zeros.
        processed_obs = np.zeros_like(obs, dtype=np.float32)

        # Applys log2 only to positive tiles
        # np.log2(0) is -inf, so we use a 'where' clause.
        positive_mask = (obs > 0)
        processed_obs[positive_mask] = np.log2(obs[positive_mask])
        
        # -1 block would have become 0
        # we must explicitly set it back to -1.0
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
        
        return processed_obs.astype(np.float32)