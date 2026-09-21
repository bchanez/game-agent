"""Game registry — the list of games the agent knows how to play.

Add a new game by dropping a module in this package that exposes a `SPEC`
(a `game_env.GameSpec`), then register it below. Scripts select one with
`--game <name>`; nothing else changes.
"""
from games.mario import SPEC as mario
from games.atari import montezuma, breakout

REGISTRY = {spec.name: spec for spec in (mario, montezuma, breakout)}


def get_game(name):
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown game '{name}'. available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]


def get_games(names):
    """Resolve a comma-separated list (or iterable) of game names to specs,
    for training one policy on a mix of games."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    return [get_game(n) for n in names]
