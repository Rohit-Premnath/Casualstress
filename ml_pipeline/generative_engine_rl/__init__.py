"""
generative_engine_rl
====================
Phase 2B: RL adversarial scenario generator for CausalStress.

This package is ISOLATED from the canonical generative_engine. It imports
read-only artifacts from the canonical pipeline (VAR fits, causal graphs,
action space spec) but writes to its own database tables and produces
its own backtest outputs.

If Phase 2B training is unstable or the regime-transition stretch goal is
descoped, this entire package can be deleted with zero impact on the
canonical 90.0% test coverage headline result.

Modules:
    causal_stress_env  : Gymnasium environment wrapping the canonical scenario generator
    action_space_loader: Loads action_space.yaml and provides agent-side helpers
    action_wrapper     : MultiDiscrete wrapper that exposes a clean action space to PPO
    env_factory        : Single source of truth for env construction (fast/real modes)
    real_mode_loader   : Production-DB data loading for real-mode training
    diagnose_real_mode : Pre-flight check for real-mode prerequisites
    portfolio_model    : Simple equal-weight portfolio for scaffolding rewards
    rewards            : Multi-component adversarial reward function
    rollout            : Random-policy rollout for environment sanity testing
    train_ppo          : PPO training script with checkpointing and evaluation
"""

__version__ = "0.7.0-deterministic-generator"