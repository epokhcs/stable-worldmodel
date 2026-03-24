import gymnasium as gym
import numpy as np
import pytest
import stable_worldmodel.envs

@pytest.mark.parametrize("env_id", [
    "swm/FetchReach-v3",
    "swm/FetchPush-v3",
    "swm/FetchSlide-v3",
    "swm/FetchPickAndPlace-v3"
])
def test_fetch_environment_initialization(env_id):
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box), "Observation space must be a flattened Box"
    assert env.observation_space.shape[0] > 0, "Observation space must have positive dimension"
    
    obs, info = env.reset()
    assert getattr(env.unwrapped, "env_name", None) or "env_name" in info, "env_name must be present"
    assert "proprio" in info, "proprio state must be exposed"
    assert "state" in info, "flattened state must be exposed"
    assert "goal_state" in info, "goal state must be exposed"
    
    env.close()

def test_fetch_visual_randomization():
    env = gym.make("swm/FetchPush-v3")
    
    color_target = np.array([0.5, 0.1, 0.9])
    obs, info = env.reset(options={
        "variation_values": {
            "table.color": color_target,
            "background.color": color_target,
            "object.color": color_target
        }
    })
    
    # Assert the variation space tracked the override
    vs = env.get_wrapper_attr("variation_space")
    np.testing.assert_allclose(vs["table"]["color"].value, color_target)
    np.testing.assert_allclose(vs["background"]["color"].value, color_target)
    
    env.close()

def test_fetch_physical_randomization():
    env = gym.make("swm/FetchPush-v3")
    
    target_pos = np.array([1.4, 0.8])
    obs, info = env.reset(options={
        "variation_values": {
            "block.start_position": target_pos
        }
    })
    
    vs = env.get_wrapper_attr("variation_space")
    np.testing.assert_allclose(vs["block"]["start_position"].value, target_pos)
    
    env.close()
