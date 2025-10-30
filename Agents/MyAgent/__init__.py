# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#ルールベースの初期行動判断モデルのようにobservationやactionを必要としないタイプのモデルの登録方法例
import os
import json
from ASRCAISim1.policy import StandalonePolicy

def getUserAgentClass(args={}):
    from .MyAgent import SADAgent
    return SADAgent

def getUserAgentModelConfig(args={}):
    with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
        cfg = json.load(f)
    model_type = (args or {}).get("type", "original")
    # fallback to "original" when not specified
    return cfg.get(model_type, cfg.get("original", {}))

def isUserAgentSingleAsset(args={}):
    # Centralized agent controlling 4 fighters (ports "0".."3")
    return False

class DummyPolicy(StandalonePolicy):
    def step(self, observation, reward, done, info, agentFullName, observation_space, action_space):
        # This agent decides internally; policy output unused.
        return None

def getUserPolicy(args={}):
    return DummyPolicy()

def getBlueInitialState(args={}):
    # Optional helper for selected initial state (4 fighters)
    return [
        {"pos": [-10000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
        {"pos": [10000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
        {"pos": [-20000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
        {"pos": [20000.0, 100000.0, -10000.0], "vel": 270.0, "heading": 270.0},
    ]
