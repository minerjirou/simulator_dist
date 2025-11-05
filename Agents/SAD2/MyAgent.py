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
            self.state = "HIGH_ATTACK"
            self.stateEnterT = Time(0.0, TimeSystem.TT)

    def initialize(self):
        super().initialize()
        # Parameters (can be overridden by modelConfig)
        cfg = self.modelConfig if isinstance(self.modelConfig, dict) else {}
        self.altMin = float(cfg.get("altMin", 7000.0))
        self.altMax = float(cfg.get("altMax", 13000.0))
        self.nominalAlt = float(cfg.get("nominalAlt", 10000.0))
        self.minimumV = float(cfg.get("minimumV", 200.0))
        self.recoveryV = float(cfg.get("recoveryV", 240.0))
        self.crankAz = float(cfg.get("crankAz", np.deg2rad(35.0)))
        self.beamAz = float(cfg.get("beamAz", np.deg2rad(90.0)))
        self.pitbullDelay = float(cfg.get("pitbullDelay", 6.0))
        self.dragTime = float(cfg.get("dragTime", 6.0))
        self.recommitTime = float(cfg.get("recommitTime", 6.0))
        self.maxSimulShot = int(cfg.get("maxSimulShot", 2))
        self.rShotThreshold = float(cfg.get("rShotThreshold", 0.85))
        self.shotIntervalMin = float(cfg.get("shotIntervalMin", 5.0))
        # New: speed/altitude band and stack parameters
        self.vBandMin = float(cfg.get("vBandMin", 260.0))
        self.vBandMax = float(cfg.get("vBandMax", 300.0))
        self.bandAltMin = float(cfg.get("bandAltMin", 9000.0))
        self.bandAltMax = float(cfg.get("bandAltMax", 11000.0))
        self.stackAlt = float(cfg.get("stackAlt", 1800.0))
        # New: forward gate geometry
        self.capD = float(cfg.get("capD", 15000.0))
        self.capL = float(cfg.get("capL", 7000.0))
        # New: intercept horizon limits
        self.tauMin = float(cfg.get("tauMin", 2.0))
        self.tauMax = float(cfg.get("tauMax", 8.0))
        # New: bias weight for centrality
        self.centerBiasW = float(cfg.get("centerBiasW", 0.15))
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
            ai.state = "HIGH_ATTACK"
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

    def select_targets(self):
        # attackers: threat-based distinct; defenders: max threat
        if not self.lastTrackInfo:
            return {}
        targets = {}
        sorted_ports = sorted(self.parents.keys(), key=lambda k: int(k))
        # attackers
        attack_scores = []
        for i, port in enumerate(sorted_ports[:2]):
            my_idx = list(self.parents.keys()).index(port)
            myMotion = self.ourMotion[my_idx]
            scores = [(self.compute_threat(myMotion, t), idx, t) for idx, t in enumerate(self.lastTrackInfo)]
            scores.sort(key=lambda x: x[0], reverse=True)
            if scores:
                if i == 0:
                    sel = scores[0]
                    attack_scores = scores
                else:
                    sel = scores[0]
                    if len(scores) > 1 and attack_scores and sel[1] == attack_scores[0][1]:
                        sel = scores[1]
                p = self.parents[port]
                targets[p.getFullName()] = sel[2]
        # defenders
        for port in sorted_ports[2:4]:
            p = self.parents[port]
            my_idx = list(self.parents.keys()).index(port)
            myMotion = self.ourMotion[my_idx]
            best = None
            best_score = -1e18
            for t in self.lastTrackInfo:
                score = self.compute_threat(myMotion, t)
                if score > best_score:
                    best_score = score
                    best = t
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
        # 2Dリード迎撃: tau ~ |dr|/closing_speed を [tauMin..tauMax] にクリップ
        myp = myMotion.pos(); myv = myMotion.vel()
        tp = tgt.pos(); tv = tgt.vel()
        dr = tp - myp
        rel = tv - myv
        closing = max(1.0, -float(np.dot(dr[:2], rel[:2])) / (float(np.linalg.norm(dr[:2])) + 1e-6))
        tau = float(np.clip(float(np.linalg.norm(dr[:2])) / closing, self.tauMin, self.tauMax))
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
        return self.blend_dir(base_dir, bias_dir, float(self.centerBiasW))

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

            # choose target if available
            tgt = targets.get(pf, Track3D())

            # FSM transitions (time-based surrogate for pitbull/drag/recommit)
            # Time arithmetic in ASRCAISim1 returns seconds (float)
            now = self.manager.getTime()
            dt = float(now - ai.stateEnterT)

            if ai.state == "HIGH_ATTACK":
                # attackers: 予測迎撃ベクトルを基準に±crankのブラケット
                if i < 2:
                    if tgt and not tgt.is_none():
                        idir = self.compute_intercept_dir(myMotion, tgt)
                    else:
                        idir = np.array([cos(base_az), sin(base_az), 0.0])
                    bdir = self.heading_for_bracket(i, idir)
                    ai.dstDir = self.blend_dir(bdir, idir, 0.4)
                    # 縦スタック: 攻撃ペアで±stackAltを目標とする
                    try:
                        curAlt = float(myMotion.pos()[2])
                    except Exception:
                        curAlt = self.nominalAlt
                    sign = -1.0 if (i % 2 == 0) else 1.0
                    tgtAlt = self.nominalAlt + sign * self.stackAlt
                    # バンドと併用: 目標を帯域内にクリップ
                    tgtAlt = max(self.bandAltMin, min(self.bandAltMax, tgtAlt))
                    dz = tgtAlt - curAlt
                    if dz > 300.0:
                        ai.dstDir = np.array([ai.dstDir[0], ai.dstDir[1], 0.12])
                    elif dz < -300.0:
                        ai.dstDir = np.array([ai.dstDir[0], ai.dstDir[1], -0.12])
                    else:
                        ai.dstDir = np.array([ai.dstDir[0], ai.dstDir[1], 0.0])
                else:
                    # defenders: VIPの進行方向前方に可動ゲート（±Lの法線オフセット）
                    D = float(self.capD)
                    L = float(self.capL)
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
                        # 守備ペアの縦スタック
                        try:
                            curAlt = float(myMotion.pos()[2])
                        except Exception:
                            curAlt = self.nominalAlt
                        tgtAlt = self.nominalAlt + (sign * self.stackAlt)
                        tgtAlt = max(self.bandAltMin, min(self.bandAltMax, tgtAlt))
                        dz = tgtAlt - curAlt
                        if dz > 300.0:
                            ai.dstDir = np.array([ai.dstDir[0], ai.dstDir[1], 0.1])
                        elif dz < -300.0:
                            ai.dstDir = np.array([ai.dstDir[0], ai.dstDir[1], -0.1])
                        else:
                            ai.dstDir = np.array([ai.dstDir[0], ai.dstDir[1], 0.0])
                    else:
                        ai.dstDir = np.array([cos(base_az), sin(base_az), 0.0])
                # fire if inside normalized R threshold
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
                    # パートナーの事前クランク中は発射抑制（簡易ローテ同期）
                    ok_partner = True
                    if i < 2 and len(sorted_ports) >= 2:
                        partner_port = sorted_ports[1 - i]
                        partner_pf = self.parents[partner_port].getFullName()
                        pai = self.actionInfos.get(partner_pf)
                        if pai is not None and pai.state == "PRE_PITBULL_CRANK":
                            ok_partner = False
                    pk = self.compute_pk(parent, myMotion, tgt)
                    if r < self.rShotThreshold and pk >= self.pkThreshold and parent.isLaunchableAt(tgt) and flying < self.maxSimulShot and ok_interval and ok_partner:
                        ai.launchFlag = True
                        ai.target = tgt
                        ai.lastShotTimes[tgt.truth] = now
                        ai.state = "PRE_PITBULL_CRANK"
                        ai.stateEnterT = now
                ai.dstAlt = self.nominalAlt

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

            # speed/throttle policy with energy band
            ai.asThrottle = True
            ai.dstThrottle = 1.0
            ai.keepVel = False
            if V < self.minimumV:
                ai.asThrottle = False
                ai.keepVel = False
                ai.dstV = self.recoveryV
            else:
                # maintain 260-300 m/s band in general phases
                if ai.state in ("HIGH_ATTACK",) or (ai.state in ("PRE_PITBULL_CRANK",) and i >= 2):
                    if V > self.vBandMax:
                        ai.asThrottle = False
                        ai.keepVel = False
                        ai.dstV = float(self.vBandMax)
                    elif V < self.vBandMin:
                        ai.asThrottle = True
                        ai.dstThrottle = 1.0
                        ai.keepVel = False
                    else:
                        ai.asThrottle = True
                        ai.dstThrottle = 1.0

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
