import numpy as np

from ASRCAISim1.core import MotionState, Track3D
from BasicAgentUtility.util import calcRNorm


def compute_threat(self, myMotion: MotionState, t: Track3D) -> float:
    # Boundary非依存寄りの指標へ調整：距離(近いほど高)、閉塞速度(+)、前方占位(+)、invTTIは弱め
    dr = t.pos() - myMotion.pos()
    d2 = float(np.linalg.norm(dr[:2])) + 1e-6
    rel_v = t.vel() - myMotion.vel()
    closing = max(0.0, -float(np.dot(dr[:2], rel_v[:2])) / d2)
    # 前方占位: 敵速度が自陣方向へ向いている度合い（チーム座標の-x方向を自陣内向きとみなす）
    evel = self.teamOrigin.relPtoB(t.vel())
    front = max(0.0, -float(evel[0]))  # >0 でこちら向き
    # 距離スコアは 1/(d + const) で近距離優先
    w_dist = 1.0
    w_front = 0.8
    w_close = self.w_closure
    w_itti = 0.2 * self.w_invTTI  # 影響を弱める
    dist_score = 1.0 / (d2 + 5000.0)
    # invTTI（弱）
    epos = self.teamOrigin.relPtoB(t.pos())
    xdist = max(0.0, abs(float(epos[0]))) + 1e3
    inv_tti = front / xdist
    return w_dist * dist_score + w_front * front / 300.0 + w_close * np.tanh(closing / 200.0) + w_itti * inv_tti


def compute_vip_threat(self, t: Track3D) -> float:
    """Approximate threat of enemy track to our VIP/cap corridor.
    Uses team base coordinates as a proxy: enemy nose toward us (front),
    small x-distance into our side (xdist), and central corridor preference (|y| small).
    """
    try:
        epos = self.teamOrigin.relPtoB(t.pos())
        evel = self.teamOrigin.relPtoB(t.vel())
    except Exception:
        return 0.0
    xdist = max(0.0, abs(float(epos[0]))) + 1e3
    ycorr = 1.0 / (5000.0 + abs(float(epos[1])))  # near centerline => higher
    front = max(0.0, -float(evel[0]))
    inv_tti = front / xdist
    # Normalize and combine
    return 0.7 * inv_tti + 0.3 * ycorr


def compute_combined_threat(self, myMotion: MotionState, t: Track3D) -> float:
    s = compute_threat(self, myMotion, t)
    v = compute_vip_threat(self, t)
    return float(self.threat_w_self * s + self.threat_w_vip * v)


def select_targets(self):
    # attackers: prioritize enemy VIP (via VIP-threat) when competitive, else combined threat; defenders: VIP-centric threat
    if not self.lastTrackInfo:
        return {}
    targets = {}
    sorted_ports = sorted(self.parents.keys(), key=lambda k: int(k))

    # precompute global ranking once
    comb_scores = []
    vip_scores = []
    my0 = None
    for m in self.ourMotion:
        if np.linalg.norm(m.vel()) > 1e-3:
            my0 = m
            break
    if my0 is None and self.ourMotion:
        my0 = self.ourMotion[0]
    for idx, t in enumerate(self.lastTrackInfo):
        try:
            comb = compute_combined_threat(self, my0, t) if my0 is not None else 0.0
        except Exception:
            comb = 0.0
        try:
            vip = compute_vip_threat(self, t)
        except Exception:
            vip = 0.0
        comb_scores.append((comb, idx, t))
        vip_scores.append((vip, idx, t))
    comb_scores.sort(key=lambda x: x[0], reverse=True)
    vip_scores.sort(key=lambda x: x[0], reverse=True)

    # load-aware distribution using DL current_target (previous tick)
    try:
        prev_map = dict(self.datalink.get("current_target", {}))
    except Exception:
        prev_map = {}
    load_count = {}
    for k, v in prev_map.items():
        if v is None:
            continue
        load_count[v] = load_count.get(v, 0) + 1

    def under_cap(truth):
        if truth is None:
            return True
        cap = getattr(self, 'maxAttackersPerTarget', 2)
        return load_count.get(truth, 0) < cap

    # attackers
    attack_picks = []
    for i, port in enumerate(sorted_ports[:2]):
        # choose VIP if competitive vs combined best
        pick = comb_scores[0] if comb_scores else None
        vip_pick = vip_scores[0] if vip_scores else None
        cand = vip_pick if (vip_pick and pick and vip_pick[0] >= self.vipPickRatio * pick[0]) else pick
        # deconflict two attackers by taking next best combined if same index
        if i == 1 and cand and attack_picks and cand[1] == attack_picks[0][1] and len(comb_scores) > 1:
            cand = comb_scores[1]
        # enforce load cap (distribution)
        if cand:
            truth_try = getattr(cand[2], "truth", None)
            if not under_cap(truth_try):
                # find first alternative not exceeding cap
                alt = None
                for s in comb_scores:
                    tt = getattr(s[2], "truth", None)
                    if tt != truth_try and under_cap(tt):
                        alt = s
                        break
                if alt is not None:
                    cand = alt
        if cand:
            p = self.parents[port]
            ai = self.actionInfos[p.getFullName()]
            prev_truth = ai.currentTruth
            prev_score = ai.currentScore
            new_truth = getattr(cand[2], "truth", None)
            new_score = cand[0]
            # hysteresis: only switch if significantly better than previous
            if prev_truth is not None and any(getattr(t, "truth", None) == prev_truth for _, _, t in comb_scores):
                if new_score <= prev_score * (1.0 + self.targetSwitchHyst):
                    # keep previous target if still in list
                    kept = False
                    for s in comb_scores:
                        if getattr(s[2], "truth", None) == prev_truth:
                            targets[p.getFullName()] = s[2]
                            kept = True
                            break
                    if not kept:
                        targets[p.getFullName()] = cand[2]
                        ai.currentTruth = new_truth
                        ai.currentScore = new_score
            else:
                targets[p.getFullName()] = cand[2]
                ai.currentTruth = new_truth
                ai.currentScore = new_score
            else:
                targets[p.getFullName()] = cand[2]
                ai.currentTruth = new_truth
                ai.currentScore = new_score
            attack_picks.append(cand)
            # update load map for subsequent assignments this tick
            if new_truth is not None:
                load_count[new_truth] = load_count.get(new_truth, 0) + 1

    # defenders: guard VIP by picking max VIP threat
    for port in sorted_ports[2:4]:
        p = self.parents[port]
        best = vip_scores[0][2] if vip_scores else None
        if best is not None:
            truth_try = getattr(best, "truth", None)
            # enforce load cap for defenders too
            if not under_cap(truth_try):
                # choose next vip-threat if cap exceeded
                alt = None
                for s in vip_scores[1:]:
                    tt = getattr(s[2], "truth", None)
                    if under_cap(tt):
                        alt = s[2]
                        truth_try = tt
                        break
                targets[p.getFullName()] = alt if alt is not None else best
            else:
                targets[p.getFullName()] = best
            load_count[truth_try] = load_count.get(truth_try, 0) + 1

    # publish current target map to DL
    try:
        for port, parent in self.parents.items():
            pf = parent.getFullName()
            t = targets.get(pf, None)
            truth = getattr(t, 'truth', None) if t is not None else None
            self.datalink["current_target"][port] = truth
    except Exception:
        pass
    return targets


def compute_pk(self, parent, myMotion: MotionState, tgt: Track3D) -> float:
    # Simple surrogate of Pk using (1-RNorm), closure(+), aspect alignment
    try:
        r = float(calcRNorm(parent, myMotion, tgt, False))
    except Exception:
        dr = tgt.pos() - myMotion.pos()
        r = min(1.0, max(0.0, np.linalg.norm(dr[:2]) / 80000.0))
    dr = tgt.pos() - myMotion.pos()
    rel_v = tgt.vel() - myMotion.vel()
    closing = max(0.0, -float(np.dot(dr[:2], rel_v[:2])) / (float(np.linalg.norm(dr[:2])) + 1e-6))
    closing_n = np.tanh(closing / 200.0)
    los = dr / (np.linalg.norm(dr) + 1e-6)
    aspect_align = max(0.0, float(los[0]))
    pk = self.pk_w_r * (1.0 - r) + self.pk_w_closure * closing_n + self.pk_w_aspect * aspect_align
    return float(max(0.0, min(1.0, pk)))
