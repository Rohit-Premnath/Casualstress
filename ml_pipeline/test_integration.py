import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent if "__file__" in globals() else "."))
import numpy as np

from generative_engine.scenario_generator import generate_scenarios
from generative_engine_rl.action_space_loader import load_spec
from generative_engine_rl.causal_stress_env import CausalStressEnv
from generative_engine_rl.rollout import make_stub_var_model, make_stub_causal_adjacency

spec = load_spec()
var_model = make_stub_var_model(spec)
causal_adj = make_stub_causal_adjacency(spec)

env = CausalStressEnv(
    var_model=var_model,
    causal_adjacency=causal_adj,
    spec=spec,
    regime_at_episode_start="stressed",
    scenario_fn=generate_scenarios,   # the REAL production generator
    seed=42,
)

obs, info = env.reset(seed=1)
gspc_idx = spec.core_variables.index("^GSPC")
crash_family = spec.family_to_idx["market_crash"]
action = {
    "target_var": gspc_idx,
    "shock_magnitude": np.array([-3.0], dtype=np.float32),
    "event_family": crash_family,
}
obs2, reward, term, trunc, info = env.step(action)
print(f"Integration test: reward={reward:+.4f}, components={info['reward_breakdown']}")
print("PASSED" if not info["action_rejected"] and np.isfinite(reward) else "FAILED")