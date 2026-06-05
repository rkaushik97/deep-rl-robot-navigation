"""Algorithm registry — the single place where algorithm names map to classes.

Each algorithm lives in its own folder (ddpg/, td3/, sac/) with the network +
update code, a frozen reference-exact config.sh, and an experiments/ playground.
To add a new algorithm: drop a folder here, import its class below, add it to
REGISTRY. Everything else (training loop, evaluation) dispatches through this.
"""
from .ddpg.ddpg import DDPG
from .td3.td3 import TD3
from .sac.sac import SAC

REGISTRY = {
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC,
}


def get_agent_class(name):
    """Return the agent class for an algorithm name (e.g. 'ddpg')."""
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"unknown algorithm '{name}'; supported: {sorted(REGISTRY)}")
    return REGISTRY[key]
