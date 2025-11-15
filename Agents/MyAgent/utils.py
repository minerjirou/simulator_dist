from math import atan2, cos, sin

import numpy as np

from ASRCAISim1.core import MotionState, Track3D


def heading_for_bracket(self, idx, base_dir):
    # attackers offset opposite sides for multi-axis pressure
    side = -1.0 if (idx % 2 == 0) else 1.0
    az = atan2(base_dir[1], base_dir[0]) + side * self.crankAz
    return np.array([cos(az), sin(az), 0.0])


def _normalize2d(self, v):
    n = float(np.linalg.norm(v[:2]))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return np.array([v[0] / n, v[1] / n, 0.0])


def blend_dir(self, a, b, w_b):
    w_a = 1.0 - w_b
    v = w_a * a + w_b * b
    n = float(np.linalg.norm(v[:2]))
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return np.array([v[0] / n, v[1] / n, v[2] if abs(v[2]) > 1e-6 else 0.0])


def compute_intercept_dir(self, myMotion: MotionState, tgt: Track3D):
    # 2Dリード迎撃: tau ~ |dr|/closing_speed を 2..8s にクリップ
    myp = myMotion.pos()
    myv = myMotion.vel()
    tp = tgt.pos()
    tv = tgt.vel()
    dr = tp - myp
    rel = tv - myv
    closing = max(1.0, -float(np.dot(dr[:2], rel[:2])) / (float(np.linalg.norm(dr[:2])) + 1e-6))
    tau = float(np.clip(float(np.linalg.norm(dr[:2])) / closing, 2.0, 8.0))
    aim = tp + tv * tau
    dirv = aim - myp
    dirv[2] = 0.0
    return _normalize2d(self, dirv)


def soft_center_bias_dir(self, myMotion: MotionState, base_dir):
    # 中央(Y=0)指向の弱いバイアスを付与（境界押し出しを抑制）
    pB = self.teamOrigin.relPtoB(myMotion.pos())
    biasB = np.array([0.0, -float(pB[1]), 0.0])
    biasP = self.teamOrigin.relBtoP(biasB)
    bias_dir = _normalize2d(self, biasP)
    return blend_dir(self, base_dir, bias_dir, 0.15)


def compute_beam_dir(self, current_dir, threat_dir):
    # 脅威LOSに対して±90度のビーム。回頭量の小さい側を選択
    th = _normalize2d(self, threat_dir)
    # 2D 90度回転（左/右）
    left = np.array([-th[1], th[0], 0.0])
    right = np.array([th[1], -th[0], 0.0])
    cur = _normalize2d(self, current_dir)
    # コサイン距離が小さい方（=向きが近い方）を選ぶ
    if float(np.dot(cur[:2], left[:2])) >= float(np.dot(cur[:2], right[:2])):
        return left
    else:
        return right


def detect_close_threat(self, myMotion: MotionState):
    """Return the most dangerous close threat Track3D and helper metrics.
    Criteria:
      - range (2D) < breakRThreat
      - enemy nose toward us (cos >= breakEnemyNose)
      - enemy position roughly behind our nose (cosTail <= breakCosTail)
    """
    if not self.lastTrackInfo:
        return None, None, None, None
    myp = myMotion.pos()
    myv = myMotion.vel()
    vhat = _normalize2d(self, myv)
    best = None
    best_r = 1e18
    best_cosTail = 1.0
    best_cosNose = 0.0
    for t in self.lastTrackInfo:
        dr = t.pos() - myp
        r2 = float(np.linalg.norm(dr[:2]))
        if r2 <= 1.0:
            continue
        # our tail aspect: bandit position relative to our velocity
        cosTail = float(np.dot(vhat[:2], (dr[:2] / r2)))
        # bandit nose toward us
        tv = t.vel()
        sp = float(np.linalg.norm(tv[:2]))
        if sp < 1.0:
            continue
        bnhat = tv[:2] / sp
        toMe = myp[:2] - t.pos()[:2]
        toMeN = float(np.linalg.norm(toMe))
        if toMeN < 1.0:
            continue
        cosNose = float(np.dot(bnhat, toMe / toMeN))
        if r2 < self.breakRThreat and cosTail <= self.breakCosTail and cosNose >= self.breakEnemyNose:
            if r2 < best_r:
                best = t
                best_r = r2
                best_cosTail = cosTail
                best_cosNose = cosNose
    return best, best_r, best_cosTail, best_cosNose

