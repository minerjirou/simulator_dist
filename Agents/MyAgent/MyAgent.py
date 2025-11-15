from math import atan2, cos, sin, sqrt
import numpy as np

from ASRCAISim1.core import (
    Agent,
    MotionState,
    Track3D,
    Time,
    TimeSystem,
    StaticCollisionAvoider2D,
    LinearSegment,
    AltitudeKeeper,
)
from gymnasium import spaces
from BasicAgentUtility.util import (
    TeamOrigin,
    sortTrack3DByDistance,
    calcRNorm,
    # optional for richer PK if needed in future
    # calcRHead, calcRTail,
)


class SADAgent(Agent):
    class ActionInfo:
        def __init__(self):
            self.dstDir = np.array([1.0, 0.0, 0.0])
            self.dstAlt = 10000.0
            self.asThrottle = True
            self.dstThrottle = 1.0
            self.keepVel = False
            self.dstV = 270.0
            self.launchFlag = False
            self.target = Track3D()
            self.lastShotTimes = {}
            # FSM state control
            self.state = "OPENING_SPREAD"
            self.stateEnterT = Time(0.0, TimeSystem.TT)
            # jink timer for defensive break
            self.lastJinkFlipT = Time(0.0, TimeSystem.TT)
            # target selection hysteresis bookkeeping
            self.currentTruth = None
            self.currentScore = -1e18
            # target hysteresis
            self.currentTruth = None
            self.currentScore = -1e18

    def initialize(self):
        super().initialize()
        # Parameters (can be overridden by modelConfig)
        cfg = self.modelConfig if isinstance(self.modelConfig, dict) else {}
        self.altMin = float(cfg.get("altMin", 7000.0))
        self.altMax = float(cfg.get("altMax", 14000.0))
        self.nominalAlt = float(cfg.get("nominalAlt", 12000.0))
        self.minimumV = float(cfg.get("minimumV", 240.0))
        self.recoveryV = float(cfg.get("recoveryV", 280.0))
        self.crankAz = float(cfg.get("crankAz", np.deg2rad(35.0)))
        self.beamAz = float(cfg.get("beamAz", np.deg2rad(90.0)))
        self.pitbullDelay = float(cfg.get("pitbullDelay", 6.0))
        self.dragTime = float(cfg.get("dragTime", 6.0))
        self.recommitTime = float(cfg.get("recommitTime", 6.0))
        self.maxSimulShot = int(cfg.get("maxSimulShot", 2))
        self.rShotThreshold = float(cfg.get("rShotThreshold", 0.85))
        self.shotIntervalMin = float(cfg.get("shotIntervalMin", 5.0))
        # Defensive break/jink tuning
        self.breakRThreat = float(cfg.get("breakRThreat", 7000.0))
        self.breakExitR = float(cfg.get("breakExitR", 9000.0))
        self.breakCosTail = float(cfg.get("breakCosTail", -0.5))  # bandit roughly behind us
        self.breakEnemyNose = float(cfg.get("breakEnemyNose", 0.6))  # bandit nose toward us
        self.jinkPeriod = float(cfg.get("jinkPeriod", 2.5))
        self.jinkVz = float(cfg.get("jinkVz", 0.18))
        # Pump/extend when outnumbered or geometry is poor
        self.pumpRThreat = float(cfg.get("pumpRThreat", 15000.0))
        self.pumpExitR = float(cfg.get("pumpExitR", 18000.0))
        self.pumpTime = float(cfg.get("pumpTime", 12.0))
        self.pumpVz = float(cfg.get("pumpVz", 0.1))
        # Boundary limiter defaults (overridden by ruler values where applicable)
        self.dOutLimit = float(cfg.get("dOutLimit", 7500.0))
        self.dOutLimitThreshold = float(cfg.get("dOutLimitThreshold", 15000.0))
        self.dOutLimitStrength = float(cfg.get("dOutLimitStrength", 0.001))
        # Threat/PK weights (for advanced gating and prioritization)
        self.w_invTTI = float(cfg.get("w_invTTI", 1.2))
        self.w_closure = float(cfg.get("w_closure", 0.5))
        self.w_aspect = float(cfg.get("w_aspect", 0.2))
        self.pkThreshold = float(cfg.get("pkThreshold", 0.5))
        self.pk_w_r = float(cfg.get("pk_w_r", 0.6))
        self.pk_w_closure = float(cfg.get("pk_w_closure", 0.3))
        self.pk_w_aspect = float(cfg.get("pk_w_aspect", 0.1))
        self.grinderT = float(cfg.get("grinderT", 12.0))
        # Threat analysis weights
        self.threat_w_self = float(cfg.get("threat_w_self", 0.6))
        self.threat_w_vip = float(cfg.get("threat_w_vip", 0.4))
        # Target switching hysteresis (fractional improvement required to retarget)
        self.targetSwitchHyst = float(cfg.get("targetSwitchHyst", 0.1))
        # Target switching control and VIP priority
        self.targetSwitchHyst = float(cfg.get("targetSwitchHyst", 0.1))
        self.vipPickRatio = float(cfg.get("vipPickRatio", 0.85))
        # Opening spread/volley
        self.openingSpreadT = float(cfg.get("openingSpreadT", 20.0))
        self.spreadD = float(cfg.get("spreadD", 20000.0))
        self.spreadL = float(cfg.get("spreadL", 10000.0))
        self.spreadAlt = float(cfg.get("spreadAlt", 800.0))
        self.engageDetectR = float(cfg.get("engageDetectR", 45000.0))
        self.openingVolleyT = float(cfg.get("openingVolleyT", 40.0))
        self.openRShotThreshold = float(cfg.get("openRShotThreshold", 0.98))
        self.openPkBias = float(cfg.get("openPkBias", 0.1))
        # Datalink/shared targeting
        self.enableDatalink = bool(cfg.get("enableDatalink", True))
        self.dlFocusPk = float(cfg.get("dlFocusPk", 0.35))
        self.dlPrimaryUpdateT = float(cfg.get("dlPrimaryUpdateT", 2.0))
        # Opening altitude/shot preferences
        self.initialClimbVz = float(cfg.get("initialClimbVz", 0.30))
        self.standoffShotR = float(cfg.get("standoffShotR", 45000.0))
        # Energy and phase thresholds
        self.altEnergyMin = float(cfg.get("altEnergyMin", 10000.0))
        self.vEnergyMin = float(cfg.get("vEnergyMin", 250.0))
        self.dFar = float(cfg.get("dFar", 40000.0))
        self.dMid = float(cfg.get("dMid", 20000.0))
        self.mwsInhibitT = float(cfg.get("mwsInhibitT", 2.5))

        # Runtime containers
        self.ourMotion = []
        self.ourObservables = []
        self.lastTrackInfo = []
        self.msls = []
        self.mws = []

        # Per-parent action and last action obs
        self.actionInfos = {}
        self.last_action_obs = {}
        for _, parent in self.parents.items():
            self.actionInfos[parent.getFullName()] = SADAgent.ActionInfo()
            self.last_action_obs[parent.getFullName()] = np.zeros([6], dtype=np.float32)

        # Team origin (set properly in validate())
        self.teamOrigin = None
        self.dOut = 0.0
        self.dLine = 0.0
        self.altitudeKeeper = AltitudeKeeper()
        # simple in-agent datalink (blackboard)
        self.datalink = {
            "primary_truth": None,
            "last_update": Time(0.0, TimeSystem.TT),
        }

        # Minimal gym spaces to satisfy GymManager
        self._observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._action_space = spaces.Discrete(1)

    def validate(self):
        # Pull arena info and set team origin/coordinate helper
        r = self.manager.getRuler()().observables()
        self.dOut = r["dOut"]
        dLine = r["dLine"]
        eastSider = r["eastSider"]
        self.teamOrigin = TeamOrigin(self.getTeam() == eastSider, dLine)
        self.dLine = dLine

        # Setup controller mode and initial heading (toward forward along team axis)
        for _, parent in self.parents.items():
            parent.setFlightControllerMode("fromDirAndVel")
            ai = self.actionInfos[parent.getFullName()]
            # Initial heading: forward along team axis
            ai.dstDir = np.array([0.0, -1.0 if self.getTeam() == eastSider else 1.0, 0.0])
            ai.dstAlt = self.nominalAlt
            ai.state = "OPENING_SPREAD"
            ai.stateEnterT = self.manager.getTime()
            ai.lastShotTimes = {}

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space

    def makeObs(self):
        # Minimal observation matching Box(shape=(1,))
        return np.zeros((1,), dtype=np.float32)

    # --- helpers to extract observables ---
    def extractFriendObservables(self):
        self.ourMotion = []
        self.ourObservables = []
        firstAlive = None
        for _, parent in self.parents.items():
            if parent.isAlive():
                firstAlive = parent
                break
        if firstAlive is None:
            return
        parentFullNames = set()
        for _, parent in self.parents.items():
            parentFullNames.add(parent.getFullName())
            if parent.isAlive():
                self.ourMotion.append(MotionState(parent.observables["motion"]).transformTo(self.getLocalCRS()))
                self.ourObservables.append(parent.observables)
            else:
                self.ourMotion.append(MotionState())
                self.ourObservables.append(firstAlive.observables.at_p("/shared/fighter").at(parent.getFullName()))
        # add friends outside parents for relative calculations if needed
        for fullName, fObs in firstAlive.observables.at_p("/shared/fighter").items():
            if fullName not in parentFullNames:
                if fObs.at("isAlive"):
                    self.ourMotion.append(MotionState(fObs["motion"]).transformTo(self.getLocalCRS()))
                else:
                    self.ourMotion.append(MotionState())
                self.ourObservables.append(fObs)

    def extractEnemyObservables(self):
        self.lastTrackInfo = []
        firstAlive = None
        for _, parent in self.parents.items():
            if parent.isAlive():
                firstAlive = parent
                break
        if firstAlive is None:
            return
        self.lastTrackInfo = [
            Track3D(t).transformTo(self.getLocalCRS()) for t in firstAlive.observables.at_p("/sensor/track")
        ]
        # Sort tracks by proximity to our fighters
        sortTrack3DByDistance(self.lastTrackInfo, self.ourMotion, True)
        # update datalink primary by global best VIP-centric threat every dlPrimaryUpdateT seconds
        if self.enableDatalink and len(self.ourMotion) > 0 and len(self.lastTrackInfo) > 0:
            now = self.manager.getTime()
            if float(now - self.datalink["last_update"]) >= self.dlPrimaryUpdateT:
                best = None
                best_s = -1e18
                for t in self.lastTrackInfo:
                    try:
                        s = self.compute_vip_threat(t)
                        if s > best_s:
                            best_s, best = s, t
                    except Exception:
                        pass
                if best is not None:
                    try:
                        self.datalink["primary_truth"] = best.truth
                    except Exception:
                        self.datalink["primary_truth"] = None
                    self.datalink["last_update"] = now

    def extractFriendMissileObservables(self):
        # flatten missiles sorted by launch time
        def launchedT(m):
            return Time(m["launchedT"]) if m["isAlive"] and m["hasLaunched"] else Time(np.inf, TimeSystem.TT)
        self.msls = sorted(
            sum([[m for m in f.at_p("/weapon/missiles")] for f in self.ourObservables], []),
            key=launchedT,
        )

    def extractEnemyMissileObservables(self):
        # enemy missile tracks per fighter (front-most first). Guarded to avoid native errors.
        self.mws = []
        try:
            from ASRCAISim1.core import Track2D
        except Exception:
            # fallback to empty if Track2D unavailable
            self.mws = [[] for _ in self.ourMotion[: len(self.parents)]]
            return
        for fIdx, fMotion in enumerate(self.ourMotion[: len(self.parents)]):
            mvec = []
            try:
                fObs = self.ourObservables[fIdx]
                if fObs["isAlive"] and fObs.contains_p("/sensor/mws/track"):
                    for mObs in fObs.at_p("/sensor/mws/track"):
                        try:
                            mtrk = Track2D(mObs).transformTo(self.getLocalCRS())
                            mvec.append(mtrk)
                        except Exception:
                            pass
            except Exception:
                pass
            self.mws.append(mvec)

    # --- tactical helpers ---

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
        s = self.compute_threat(myMotion, t)
        v = self.compute_vip_threat(t)
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
                comb = self.compute_combined_threat(my0, t) if my0 is not None else 0.0
            except Exception:
                comb = 0.0
            try:
                vip = self.compute_vip_threat(t)
            except Exception:
                vip = 0.0
            comb_scores.append((comb, idx, t))
            vip_scores.append((vip, idx, t))
        comb_scores.sort(key=lambda x: x[0], reverse=True)
        vip_scores.sort(key=lambda x: x[0], reverse=True)

        # attackers
        attack_picks = []
        for i, port in enumerate(sorted_ports[:2]):
            # choose VIP if competitive vs combined best
            pick = comb_scores[0] if comb_scores else None
            vip_pick = vip_scores[0] if vip_scores else None
            if vip_pick and pick and vip_pick[0] >= self.vipPickRatio * pick[0]:
                cand = vip_pick
            else:
                cand = pick
            # deconflict two attackers by taking next best combined if same index
            if i == 1 and cand and attack_picks and cand[1] == attack_picks[0][1] and len(comb_scores) > 1:
                cand = comb_scores[1]
            if cand:
                p = self.parents[port]
                ai = self.actionInfos[p.getFullName()]
                prev_truth = ai.currentTruth
                prev_score = ai.currentScore
                new_truth = getattr(cand[2], 'truth', None)
                new_score = cand[0]
                # hysteresis: only switch if significantly better than previous
                if prev_truth is not None and any(getattr(t, 'truth', None) == prev_truth for _,_,t in comb_scores):
                    if new_score <= prev_score * (1.0 + self.targetSwitchHyst):
                        # keep previous target if still in list
                        kept = False
                        for s in comb_scores:
                            if getattr(s[2], 'truth', None) == prev_truth:
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

        # defenders: guard VIP by picking max VIP threat
        for port in sorted_ports[2:4]:
            p = self.parents[port]
            best = vip_scores[0][2] if vip_scores else None
            if best is not None:
                targets[p.getFullName()] = best
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

    

    def heading_for_bracket(self, idx, base_dir):
        # attackers offset opposite sides for multi-axis pressure
        side = -1.0 if (idx % 2 == 0) else 1.0
        az = atan2(base_dir[1], base_dir[0]) + side * self.crankAz
        return np.array([cos(az), sin(az), 0.0])

    # --- new geometric helpers ---
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
        myp = myMotion.pos(); myv = myMotion.vel()
        tp = tgt.pos(); tv = tgt.vel()
        dr = tp - myp
        rel = tv - myv
        closing = max(1.0, -float(np.dot(dr[:2], rel[:2])) / (float(np.linalg.norm(dr[:2])) + 1e-6))
        tau = float(np.clip(float(np.linalg.norm(dr[:2])) / closing, 2.0, 8.0))
        aim = tp + tv * tau
        dirv = aim - myp
        dirv[2] = 0.0
        return self._normalize2d(dirv)

    def soft_center_bias_dir(self, myMotion: MotionState, base_dir):
        # 中央(Y=0)指向の弱いバイアスを付与（境界押し出しを抑制）
        pB = self.teamOrigin.relPtoB(myMotion.pos())
        biasB = np.array([0.0, -float(pB[1]), 0.0])
        biasP = self.teamOrigin.relBtoP(biasB)
        bias_dir = self._normalize2d(biasP)
        return self.blend_dir(base_dir, bias_dir, 0.15)

    def compute_beam_dir(self, current_dir, threat_dir):
        # 脅威LOSに対して±90度のビーム。回頭量の小さい側を選択
        th = self._normalize2d(threat_dir)
        # 2D 90度回転（左/右）
        left = np.array([-th[1], th[0], 0.0])
        right = np.array([th[1], -th[0], 0.0])
        cur = self._normalize2d(current_dir)
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
        myp = myMotion.pos(); myv = myMotion.vel()
        vhat = self._normalize2d(myv)
        best = None; best_r = 1e18; best_cosTail = 1.0; best_cosNose = 0.0
        for t in self.lastTrackInfo:
            dr = t.pos() - myp
            r2 = float(np.linalg.norm(dr[:2]))
            if r2 <= 1.0:
                continue
            # our tail aspect: bandit position relative to our velocity
            cosTail = float(np.dot(vhat[:2], (dr[:2] / r2)))
            # bandit nose toward us
            tv = t.vel(); sp = float(np.linalg.norm(tv[:2]))
            if sp < 1.0:
                continue
            bnhat = tv[:2] / sp
            toMe = (myp[:2] - t.pos()[:2])
            toMeN = float(np.linalg.norm(toMe))
            if toMeN < 1.0:
                continue
            cosNose = float(np.dot(bnhat, toMe / toMeN))
            if r2 < self.breakRThreat and cosTail <= self.breakCosTail and cosNose >= self.breakEnemyNose:
                if r2 < best_r:
                    best = t; best_r = r2; best_cosTail = cosTail; best_cosNose = cosNose
        return best, best_r, best_cosTail, best_cosNose

    def deploy(self, action):
        # refresh observables
        self.extractFriendObservables()
        self.extractEnemyObservables()
        self.extractFriendMissileObservables()
        self.extractEnemyMissileObservables()

        targets = self.select_targets()

        # VIP候補（守備ゲートの中心を決める参照）: 先頭機の評価で最大のトラック
        vip = None
        if self.lastTrackInfo and len(self.ourMotion) > 0:
            my0 = self.ourMotion[0]
            smax, vmax = -1e18, None
            for t in self.lastTrackInfo:
                s = self.compute_threat(my0, t)
                if s > smax:
                    smax, vmax = s, t
            vip = vmax

        # determine base forward direction in team coords
        fwd = self.teamOrigin.relBtoP(np.array([1.0, 0.0, 0.0]))
        base_az = atan2(fwd[1], fwd[0])

        sorted_ports = sorted(self.parents.keys(), key=lambda k: int(k))
        for i, port in enumerate(sorted_ports):
            parent = self.parents[port]
            pf = parent.getFullName()
            ai = self.actionInfos[pf]
            if not parent.isAlive():
                continue

            my_idx = list(self.parents.keys()).index(port)
            myMotion = self.ourMotion[my_idx]
            V = np.linalg.norm(myMotion.vel())
            # simple energy check (Phase 0: build band)
            pos = myMotion.pos()
            alt = -float(pos[2]) if len(pos) > 2 else 0.0
            E_ok = (alt >= self.altEnergyMin and V >= self.vEnergyMin)

            # choose target; allow datalink focus for first two (strikers)
            tgt = targets.get(pf, Track3D())
            if self.enableDatalink and i < 2 and self.datalink.get("primary_truth") is not None:
                # try to find matching track by truth
                for t in self.lastTrackInfo:
                    try:
                        if t.truth == self.datalink["primary_truth"]:
                            tgt = t
                            break
                    except Exception:
                        pass

            # FSM transitions (time-based surrogate for pitbull/drag/recommit)
            # Time arithmetic in ASRCAISim1 returns seconds (float)
            now = self.manager.getTime()
            dt = float(now - ai.stateEnterT)

            # High-priority defensive check: close-in threat on our six with nose-on
            close_tgt, close_r, close_cosTail, close_cosNose = self.detect_close_threat(myMotion)
            if close_tgt is not None and ai.state != "DEFENSIVE_BREAK_JINK":
                ai.state = "DEFENSIVE_BREAK_JINK"
                ai.stateEnterT = now
                ai.lastJinkFlipT = now
            # Missile warning preemption (Phase 3): jink + fire inhibit
            myMWS = self.mws[my_idx] if my_idx < len(self.mws) else []
            if len(myMWS) > 0:
                if hasattr(ai, 'lastMwsT'):
                    ai.lastMwsT = now
                if ai.state != "DEFENSIVE_BREAK_JINK":
                    ai.state = "DEFENSIVE_BREAK_JINK"
                    ai.stateEnterT = now
                    ai.lastJinkFlipT = now

            # Outnumbered quick pump/extend: if 2+ enemies within pumpRThreat, go cold to open range
            if ai.state == "HIGH_ATTACK":
                nearCnt = 0
                th_sum = np.zeros(3)
                for t in (self.lastTrackInfo or []):
                    dr = t.pos() - myMotion.pos()
                    dxy = float(np.linalg.norm(dr[:2]))
                    if dxy < self.pumpRThreat:
                        nearCnt += 1
                        th_sum += np.array([dr[0], dr[1], 0.0])
                if nearCnt >= 2:
                    ai.state = "PUMP_EXTEND"
                    ai.stateEnterT = now
                    # store a quick cold direction opposite to average threat bearing
                    if np.linalg.norm(th_sum[:2]) > 1.0:
                        cold = -self._normalize2d(th_sum)
                    else:
                        cold = np.array([cos(base_az + np.pi), sin(base_az + np.pi), 0.0])
                    ai.dstDir = np.array([cold[0], cold[1], self.pumpVz])

            if ai.state == "OPENING_SPREAD":
                # Opening phase: prioritize aggressive climb to nominal altitude band,
                # then transition into lateral spread/formation once energy is built.
                # Current altitude from MotionState (z is negative upward in local CRS)
                pos = myMotion.pos()
                alt = -float(pos[2]) if len(pos) > 2 else 0.0
                # Compute team-forward unit and lateral normal
                fwd = self.teamOrigin.relBtoP(np.array([1.0, 0.0, 0.0]))
                fhat = self._normalize2d(fwd)
                nhat = np.array([-fhat[1], fhat[0], 0.0])

                if alt < 12000.0:
                    # Below 12km: maximum-rate climb, forward component is secondary.
                    # Use pure forward direction with strong upward component.
                    base2d = fhat
                    # Negative z is climb in this CRS; emphasize vertical over horizontal.
                    vz = -abs(self.initialClimbVz)
                    ai.dstDir = np.array([base2d[0] * 0.6, base2d[1] * 0.6, vz])
                    ai.dstAlt = self.nominalAlt
                else:
                    # In band 12–14km: use existing spread/formation logic.
                    sign = -1.0 if (i % 2 == 0) else 1.0
                    gate_pt = myMotion.pos() + fhat * self.spreadD + sign * self.spreadL * nhat
                    dr = gate_pt - myMotion.pos(); dr[2] = 0.0
                    if np.linalg.norm(dr[:2]) > 1.0:
                        base2d = dr / np.linalg.norm(dr)
                    else:
                        base2d = fhat
                    # add gentle climb/level flight in spread phase
                    ai.dstDir = np.array([base2d[0], base2d[1], 0.0])
                    # altitude stack
                    alt_off = (i - 1.5) * 0.5 * self.spreadAlt
                    ai.dstAlt = max(self.altMin, min(self.altMax, self.nominalAlt + alt_off))
                # transition when enemy close or timer expires
                minR = 1e18
                for t in (self.lastTrackInfo or []):
                    d = float(np.linalg.norm((t.pos() - myMotion.pos())[:2]))
                    if d < minR:
                        minR = d
                if dt >= self.openingSpreadT or minR <= self.engageDetectR:
                    ai.state = "HIGH_ATTACK"
                    ai.stateEnterT = now

            elif ai.state == "HIGH_ATTACK":
                # attackers: 予測迎撃ベクトルを基準に±crankのブラケット
                if i < 2:
                    if tgt and not tgt.is_none():
                        idir = self.compute_intercept_dir(myMotion, tgt)
                    else:
                        idir = np.array([cos(base_az), sin(base_az), 0.0])
                    bdir = self.heading_for_bracket(i, idir)
                    ai.dstDir = self.blend_dir(bdir, idir, 0.4)
                else:
                    # defenders: VIPの進行方向前方に可動ゲート（±Lの法線オフセット）
                    D = 15000.0  # 15km ahead
                    L = 7000.0   # 7km lateral
                    if vip is not None and not vip.is_none():
                        v = vip.vel()
                        sp = float(np.linalg.norm(v[:2]))
                        if sp < 30.0:
                            vel_hat = np.array([cos(base_az), sin(base_az), 0.0])
                        else:
                            vel_hat = self._normalize2d(v)
                        center = vip.pos() + vel_hat * D
                        n_hat = np.array([-vel_hat[1], vel_hat[0], 0.0])
                        sign = -1.0 if (i == 2) else 1.0
                        gate_pt = center + sign * L * n_hat
                        dr = gate_pt - myMotion.pos(); dr[2] = 0.0
                        if np.linalg.norm(dr[:2]) > 1.0:
                            ai.dstDir = dr / np.linalg.norm(dr)
                        else:
                            ai.dstDir = vel_hat
                    else:
                        ai.dstDir = np.array([cos(base_az), sin(base_az), 0.0])
                # fire if inside normalized R threshold (with energy/phase gating)
                if tgt and not tgt.is_none():
                    r = calcRNorm(parent, myMotion, tgt, False)
                    flying = 0
                    if self.ourObservables[my_idx].contains_p("/weapon/missiles"):
                        for m in self.ourObservables[my_idx].at_p("/weapon/missiles"):
                            if m.at("isAlive")() and m.at("hasLaunched")():
                                flying += 1
                    ok_interval = True
                    if tgt.truth in ai.lastShotTimes:
                        ok_interval = float(now - ai.lastShotTimes[tgt.truth]) >= self.shotIntervalMin
                    # partner deconfliction: avoid doubling on same tgt unless pk is high
                    ok_partner = True
                    for jport, jparent in self.parents.items():
                        if jport == port:
                            continue
                        jai = self.actionInfos[jparent.getFullName()]
                        if tgt.truth in jai.lastShotTimes:
                            if float(now - jai.lastShotTimes[tgt.truth]) < self.shotIntervalMin:
                                ok_partner = False
                    pk = self.compute_pk(parent, myMotion, tgt)
                    # datalink focus gives slight bias to fire (to maintain pressure)
                    dl_bias = 0.0
                    try:
                        if self.enableDatalink and i < 2 and tgt.truth == self.datalink.get("primary_truth"):
                            dl_bias = 0.05
                    except Exception:
                        pass
                    # base thresholds
                    rThr = self.rShotThreshold
                    pkBias = dl_bias
                    # cheap-shot handling: if energy band not yet achieved, be stricter
                    if not E_ok:
                        rThr = min(1.0, rThr * 0.9)
                        pkBias -= 0.1
                    # altitude-based range compensation: if significantly higher, extend; lower, be conservative
                    try:
                        my_alt = -float(myMotion.pos()[2])
                        tgt_alt = -float(tgt.pos()[2])
                        dalt = my_alt - tgt_alt
                        if dalt > 1000.0:
                            # we are higher: slightly extend range
                            rThr = min(1.0, rThr * 1.03)
                            pkBias += 0.03
                        elif dalt < -1000.0:
                            # we are lower: be a bit more conservative
                            rThr = max(0.8, rThr * 0.97)
                            pkBias -= 0.03
                    except Exception:
                        pass
                    # add VIP-threat-based bias to prioritize high-danger bandits
                    try:
                        vip_th = self.compute_vip_threat(tgt)
                        pkBias += min(0.12, 0.08 * vip_th)
                    except Exception:
                        pass
                    # standoff distance rule: if within standoffShotR, bias to shoot immediately
                    try:
                        dr2 = tgt.pos() - myMotion.pos()
                        rng2d = float(np.linalg.norm(dr2[:2]))
                    except Exception:
                        rng2d = 1e9
                    if rng2d <= self.standoffShotR:
                        rThr = max(rThr, 0.99)
                        pkBias += 0.12
                    if dt <= self.openingVolleyT:
                        rThr = max(rThr, self.openRShotThreshold)
                        pkBias += self.openPkBias
                    if r < rThr and parent.isLaunchableAt(tgt) and flying < self.maxSimulShot and ok_interval and (ok_partner or pk >= max(0.8, self.pkThreshold)) and (pk + pkBias) >= self.pkThreshold:
                        ai.launchFlag = True
                        ai.target = tgt
                        ai.lastShotTimes[tgt.truth] = now
                        # micro nose-pointing toward target at launch for better missile kinematics
                        try:
                            los = tgt.pos() - myMotion.pos()
                            los2d = self._normalize2d(los)
                            aim_dir = np.array([los2d[0], los2d[1], 0.0])
                            ai.dstDir = self.blend_dir(ai.dstDir, aim_dir, 0.3)
                        except Exception:
                            pass
                        ai.state = "PRE_PITBULL_CRANK"
                        ai.stateEnterT = now
                ai.dstAlt = self.nominalAlt

            elif ai.state == "PUMP_EXTEND":
                # Turn cold from the group center of threats, extend and slight climb to regain margin
                th_sum = np.zeros(3)
                minR = 1e18
                for t in (self.lastTrackInfo or []):
                    dr = t.pos() - myMotion.pos()
                    th_sum += np.array([dr[0], dr[1], 0.0])
                    minR = min(minR, float(np.linalg.norm(dr[:2])))
                if np.linalg.norm(th_sum[:2]) > 1.0:
                    cold = -self._normalize2d(th_sum)
                else:
                    cold = np.array([cos(base_az + np.pi), sin(base_az + np.pi), 0.0])
                ai.dstDir = np.array([cold[0], cold[1], self.pumpVz])
                ai.dstAlt = min(self.altMax, self.nominalAlt + 800.0)
                # Exit if distance opened or time elapsed
                if (minR >= self.pumpExitR) or (dt >= self.pumpTime):
                    ai.state = "RECLIMB_RECOMMIT"
                    ai.stateEnterT = now

            elif ai.state == "DEFENSIVE_BREAK_JINK":
                # Missile/close threat defence: choose Crank / Beam / Drag by range
                # Threat direction: from us to closest bandit; if lost, fall back to forward
                if close_tgt is not None:
                    dr = close_tgt.pos() - myMotion.pos()
                elif tgt and not tgt.is_none():
                    dr = tgt.pos() - myMotion.pos()
                else:
                    dr = np.array([cos(base_az), sin(base_az), 0.0])
                th2d = np.array([dr[0], dr[1], 0.0])
                dxy = float(np.linalg.norm(th2d[:2]))
                los2d = self._normalize2d(th2d)

                farR = 40000.0
                midR = 20000.0

                if dxy >= farR:
                    # Far: crank to hold support while offsetting
                    base_dir = los2d
                    crank_dir = self.heading_for_bracket(i, base_dir)
                    ai.dstDir = self.blend_dir(base_dir, crank_dir, 0.6)
                elif dxy >= midR:
                    # Mid: classic beam + gentle descent
                    beam2d = self.compute_beam_dir(myMotion.vel(), th2d)
                    ai.dstDir = np.array([beam2d[0], beam2d[1], -abs(self.jinkVz)])
                else:
                    # Near: drag (turn tail and extend, slight descent)
                    drag2d = -los2d
                    ai.dstDir = np.array([drag2d[0], drag2d[1], -max(0.2, self.jinkVz)])

                # Periodic vertical jink (modulates around chosen horizontal mode)
                if float(now - ai.lastJinkFlipT) >= self.jinkPeriod:
                    ai.lastJinkFlipT = now
                phase = 0.0 if float(now - ai.lastJinkFlipT) < self.jinkPeriod / 2.0 else 1.0
                # Small additional jiggle, keep overall vz downward or small up
                vz_jink = self.jinkVz if phase < 0.5 else -self.jinkVz
                ai.dstDir = self.blend_dir(ai.dstDir, np.array([ai.dstDir[0], ai.dstDir[1], vz_jink]), 0.4)
                ai.dstAlt = max(self.altMin, min(self.altMax, self.nominalAlt))
                # Exit when geometry relaxed or distance opened
                ok_exit = True
                if close_tgt is not None:
                    # recompute metrics
                    dr = close_tgt.pos() - myMotion.pos()
                    r2 = float(np.linalg.norm(dr[:2]))
                    tv = close_tgt.vel(); sp = float(np.linalg.norm(tv[:2]))
                    cosNose = 0.0
                    if sp >= 1.0:
                        cosNose = float(np.dot(tv[:2] / sp, (myMotion.pos()[:2] - close_tgt.pos()[:2]) / max(1.0, float(np.linalg.norm(myMotion.pos()[:2] - close_tgt.pos()[:2])))))
                    ok_exit = (r2 >= self.breakExitR) or (cosNose < 0.3) or (dt >= 8.0)
                if ok_exit:
                    ai.state = "RECLIMB_RECOMMIT"
                    ai.stateEnterT = now

            elif ai.state == "PRE_PITBULL_CRANK":
                # hold offset crank relative to target bearing
                if tgt and not tgt.is_none():
                    dr = tgt.pos() - myMotion.pos()
                    az = atan2(dr[1], dr[0])
                    side = -1.0 if (i % 2 == 0) else 1.0
                    ai.dstDir = np.array([cos(az + side * self.crankAz), sin(az + side * self.crankAz), 0.0])
                else:
                    ai.dstDir = np.array([cos(base_az), sin(base_az), 0.0])
                ai.dstAlt = self.nominalAlt
                # Pitbull: detect by missile flight time when available, otherwise timer
                pitbull = False
                if self.ourObservables[my_idx].contains_p("/weapon/missiles"):
                    latest_ft = -1e9
                    for m in self.ourObservables[my_idx].at_p("/weapon/missiles"):
                        if m.at("isAlive")() and m.at("hasLaunched")():
                            # launchedT may be serialized in missile obs
                            if "launchedT" in m:
                                launchedT = Time(m["launchedT"])
                                ft = float(now - launchedT)
                                latest_ft = max(latest_ft, ft)
                    if latest_ft >= self.pitbullDelay:
                        pitbull = True
                if pitbull or dt >= self.pitbullDelay:
                    ai.state = "PITBULL_BEAM_DRAG_LOW"
                    ai.stateEnterT = now

            elif ai.state == "PITBULL_BEAM_DRAG_LOW":
                # beam perpendicular to LOS and slight descent
                if tgt and not tgt.is_none():
                    dr = tgt.pos() - myMotion.pos()
                    az = atan2(dr[1], dr[0])
                    ai.dstDir = np.array([cos(az + self.beamAz), sin(az + self.beamAz), -0.15])
                else:
                    ai.dstDir = np.array([cos(base_az + self.beamAz), sin(base_az + self.beamAz), -0.15])
                ai.dstAlt = max(self.altMin, self.nominalAlt - 1000.0)
                if dt >= self.dragTime:
                    ai.state = "RECLIMB_RECOMMIT"
                    ai.stateEnterT = now

            elif ai.state == "RECLIMB_RECOMMIT":
                # climb back and point nose in for recommit
                ai.dstDir = np.array([cos(base_az), sin(base_az), 0.2])
                ai.dstAlt = min(self.altMax, self.nominalAlt + 500.0)
                if dt >= self.recommitTime:
                    ai.state = "HIGH_ATTACK"
                    ai.stateEnterT = now

            # 中央指向バイアスを軽く適用
            ai.dstDir = self.soft_center_bias_dir(myMotion, ai.dstDir)

            # Boundary/altitude limiting
            avoider = StaticCollisionAvoider2D()
            c = {
                "p1": np.array([+self.dOut, -5 * self.dLine, 0]),
                "p2": np.array([+self.dOut, +5 * self.dLine, 0]),
                "infinite_p1": True,
                "infinite_p2": True,
                "isOneSide": True,
                "inner": np.array([0.0, 0.0]),
                "limit": self.dOutLimit,
                "threshold": self.dOutLimitThreshold,
                "adjustStrength": self.dOutLimitStrength,
            }
            avoider.borders.append(LinearSegment(c))
            c = c.copy(); c["p1"], c["p2"] = np.array([-self.dOut, -5 * self.dLine, 0]), np.array([-self.dOut, +5 * self.dLine, 0])
            avoider.borders.append(LinearSegment(c))
            c = c.copy(); c["p1"], c["p2"] = np.array([-5 * self.dOut, +self.dLine, 0]), np.array([+5 * self.dOut, +self.dLine, 0])
            avoider.borders.append(LinearSegment(c))
            c = c.copy(); c["p1"], c["p2"] = np.array([-5 * self.dOut, -self.dLine, 0]), np.array([+5 * self.dOut, -self.dLine, 0])
            avoider.borders.append(LinearSegment(c))
            ai.dstDir = avoider(myMotion, ai.dstDir)

            # MWS回避（統合）：LOSに対して90度のビーム＋軽い降下、速度維持
            myMWS = self.mws[my_idx] if my_idx < len(self.mws) else []
            if len(myMWS) > 0:
                th = np.zeros(3)
                for m in myMWS:
                    try:
                        d = m.dir()
                    except Exception:
                        d = np.array([1.0, 0.0, 0.0])
                    th += d
                if np.linalg.norm(th[:2]) > 1e-3:
                    beam2d = self.compute_beam_dir(ai.dstDir, th)
                    ai.dstDir = np.array([beam2d[0], beam2d[1], -0.12])

            # Altitude clamp via AltitudeKeeper while using dstDir
            n = max(1e-6, sqrt(ai.dstDir[0] * ai.dstDir[0] + ai.dstDir[1] * ai.dstDir[1]))
            dstPitch = atan2(-ai.dstDir[2], n)
            bottom = self.altitudeKeeper(myMotion, ai.dstDir, self.altMin)
            minPitch = atan2(-bottom[2], sqrt(bottom[0] * bottom[0] + bottom[1] * bottom[1]))
            top = self.altitudeKeeper(myMotion, ai.dstDir, self.altMax)
            maxPitch = atan2(-top[2], sqrt(top[0] * top[0] + top[1] * top[1]))
            dstPitch = max(minPitch, min(maxPitch, dstPitch))
            cs = cos(dstPitch); sn = sin(dstPitch)
            ai.dstDir = np.array([ai.dstDir[0] / n * cs, ai.dstDir[1] / n * cs, -sn])

            # speed/throttle policy
            ai.asThrottle = True
            ai.dstThrottle = 1.0
            ai.keepVel = False
            if V < self.minimumV:
                ai.asThrottle = False
                ai.keepVel = False
                ai.dstV = self.recoveryV

            # sanitize and finalize commands for this parent
            # convert dstDir to parent coordinates
            originalMyMotion = MotionState(self.ourObservables[my_idx]["motion"]) if self.ourObservables else myMotion
            # ensure finite direction
            vec = ai.dstDir
            if not np.isfinite(vec).all():
                vec = np.array([1.0, 0.0, 0.0])
            dstDir_parent = originalMyMotion.dirAtoP(vec, myMotion.pos(), self.getLocalCRS())
            self.commands[pf] = {
                "motion": {
                    "dstDir": dstDir_parent,
                },
                "weapon": {
                    "launch": ai.launchFlag,
                    "target": ai.target.to_json(),
                },
            }
            # do not send dstAlt unless using altitude command; rely on dstDir+speed control
            if ai.asThrottle:
                self.commands[pf]["motion"]["dstThrottle"] = float(ai.dstThrottle)
            elif ai.keepVel:
                self.commands[pf]["motion"]["dstAccel"] = 0.0
            else:
                v_cmd = float(ai.dstV)
                if not np.isfinite(v_cmd):
                    v_cmd = 260.0
                v_cmd = max(150.0, min(350.0, v_cmd))
                self.commands[pf]["motion"]["dstV"] = v_cmd

            # one-shot
            ai.launchFlag = False

    def control(self):
        # keep simple: reuse deploy decisions each tick
        self.deploy(None)
