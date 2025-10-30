import os
import json
from ASRCAISim1.policy import StandalonePolicy

def getUserAgentClass(args={}):
    from .SAD2Agent import SAD2Agent
    return SAD2Agent

def getUserAgentModelConfig(args={}):
    with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
        cfg = json.load(f)
    model_type = (args or {}).get("type", "original")
    return cfg.get(model_type, cfg.get("original", {}))

def isUserAgentSingleAsset(args={}):
    return False

class DummyPolicy(StandalonePolicy):
    def step(self, observation, reward, done, info, agentFullName, observation_space, action_space):
        return None

def getBlueInitialState(args={}):
    return [
        {"pos": [-10000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
        {"pos": [10000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
        {"pos": [-20000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
        {"pos": [20000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
    ]

