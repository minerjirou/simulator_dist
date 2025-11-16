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
from .targeting import (
    compute_threat,
    compute_vip_threat,
    compute_combined_threat,
    select_targets,
    compute_pk,
)
from .utils import (
    heading_for_bracket,
    _normalize2d,
    blend_dir,
    compute_intercept_dir,
    soft_center_bias_dir,
    compute_beam_dir,
    detect_close_threat,
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
            # last time MWS detected (for fire inhibit window)
            self.lastMwsT = Time(-1e9, TimeSystem.TT)
            # target selection hysteresis bookkeeping
            self.currentTruth = None
            self.currentScore = -1e18
            # duplicate kept for compatibility
            self.currentTruth = None
            self.currentScore = -1e18
            # role coordination (Shooter/Cover) and cheap-shot policy
            self.role = "SHOOTER"
            self.cheapShotCount = 0
            self.lastCheapShotT = Time(-1e9, TimeSystem.TT)

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
        # Energy gate for BVR shots
        self.altEnergyMin = float(cfg.get("altEnergyMin", 10000.0))
        self.vEnergyMin = float(cfg.get("vEnergyMin", 270.0))
        # Safety-first knobs
        self.mwsInhibitT = float(cfg.get("mwsInhibitT", 5.0))
        self.safetyMarginV = float(cfg.get("safetyMarginV", 15.0))
        self.pumpExitAll = bool(cfg.get("pumpExitAll", True))
        self.defensiveNoFire = bool(cfg.get("defensiveNoFire", True))
        self.conservativeUnderDisadvantage = bool(cfg.get("conservativeUnderDisadvantage", True))
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
        # Range buckets and pre-seeker geometry
        self.dFar = float(cfg.get("dFar", 45000.0))
        self.dMid = float(cfg.get("dMid", 25000.0))
        self.preCrankW = float(cfg.get("preCrankW", 0.6))
        self.earlyBeamEnable = bool(cfg.get("earlyBeamEnable", True))
        self.mwsDirectionalMode = bool(cfg.get("mwsDirectionalMode", True))
        # Main shot gating
        self.mainShotRequireDirect = bool(cfg.get("mainShotRequireDirect", True))
        self.hotAspectCosMin = float(cfg.get("hotAspectCosMin", 0.5))
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
        self.initialClimbVz = float(cfg.get("initialClimbVz", 0.25))
        self.standoffShotR = float(cfg.get("standoffShotR", 45000.0))

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
            "press": {},
            "current_target": {},
            "fighters": {},
            "threat": {},
            "vip_truth": None,
        }

        # Event logger controls
        self.loggingEnabled = bool(cfg.get("loggingEnabled", True))
        self._log_path = None

        def _ensure_log_path():
            if not self.loggingEnabled:
                return None
            if self._log_path is None:
                import os, datetime
                base = os.path.join(os.getcwd(), "simulator_dist", "results", "MyAgent", "logs")
                try:
                    os.makedirs(base, exist_ok=True)
                except Exception:
                    pass
                ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                self._log_path = os.path.join(base, f"fight_{ts}.log")
            return self._log_path
        self._ensure_log_path = _ensure_log_path

        def _logger_event(ev: dict):
            if not self.loggingEnabled:
                return
            try:
                path = self._ensure_log_path()
                if path is None:
                    return
                import json
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except Exception:
                pass
        self.logger_event = _logger_event

        # Role coordination and cheap-shot policy (configurable)
        self.roleCoordinationEnable = bool(cfg.get("roleCoordinationEnable", True))
        self.cheapShotMaxPerEngage = int(cfg.get("cheapShotMaxPerEngage", 1))
        self.cheapShotExtraInterval = float(cfg.get("cheapShotExtraInterval", 3.0))

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
        # Establish simple pairing: (0,1) and (2,3) by port index
        self._pairMate = {}
        port_list = sorted(self.parents.keys(), key=lambda k: int(k))
        if len(port_list) >= 2:
            self._pairMate[port_list[0]] = port_list[1]
            self._pairMate[port_list[1]] = port_list[0]
        if len(port_list) >= 4:
            self._pairMate[port_list[2]] = port_list[3]
            self._pairMate[port_list[3]] = port_list[2]
        for idx, port in enumerate(port_list):
            parent = self.parents[port]
            parent.setFlightControllerMode("fromDirAndVel")
            ai = self.actionInfos[parent.getFullName()]
            # Initial heading and state
            ai.dstDir = np.array([0.0, -1.0 if self.getTeam() == eastSider else 1.0, 0.0])
            ai.dstAlt = self.nominalAlt
            ai.state = "OPENING_SPREAD"
            ai.stateEnterT = self.manager.getTime()
            ai.lastShotTimes = {}
            ai.cheapShotCount = 0
            ai.lastCheapShotT = Time(-1e9, TimeSystem.TT)
            # Alternate roles within pairs: SHOOTER for even, COVER for odd
            ai.role = "SHOOTER" if (idx % 2 == 0) else "COVER"
            ai.press = False
        # reset DL shared maps
        self.datalink["press"] = {}
        self.datalink["current_target"] = {}
        self.datalink["fighters"] = {}
        self.datalink["threat"] = {}

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

            # Safety-first: compute immediate threats and inhibit firing if necessary
            myMWS = self.mws[my_idx] if my_idx < len(self.mws) else []
            inhibitFire = False
            close_tgt, close_r, close_cosTail, close_cosNose = self.detect_close_threat(myMotion)
            # record last MWS detection time
            if len(myMWS) > 0:
                ai.lastMwsT = now
            # MWS inhibit window forces defensive
            if (len(myMWS) > 0) or (float(now - ai.lastMwsT) <= self.mwsInhibitT):
                if ai.state != "DEFENSIVE_BREAK_JINK":
                    ai.state = "DEFENSIVE_BREAK_JINK"
                    ai.stateEnterT = now
                    ai.lastJinkFlipT = now
                inhibitFire = True
            # Close six threat forces defensive
            elif close_tgt is not None:
                if ai.state != "DEFENSIVE_BREAK_JINK":
                    ai.state = "DEFENSIVE_BREAK_JINK"
                    ai.stateEnterT = now
                    ai.lastJinkFlipT = now
                inhibitFire = True
            # Low speed energy floor: prefer defensive/extend to regain energy
            elif V < (self.minimumV + self.safetyMarginV):
                if ai.state != "DEFENSIVE_BREAK_JINK":
                    ai.state = "DEFENSIVE_BREAK_JINK"
                    ai.stateEnterT = now
                    ai.lastJinkFlipT = now
                inhibitFire = True

            # compute and share press flag (lightweight role)
            try:
                alt_now = -float(myMotion.pos()[2]) if len(myMotion.pos()) > 2 else 0.0
            except Exception:
                alt_now = 0.0
            E_ok_press = (alt_now >= self.altEnergyMin) and (V >= self.vEnergyMin)
            ai.press = (len(myMWS) == 0 and close_tgt is None and E_ok_press)
            try:
                self.datalink["press"][port] = bool(ai.press)
            except Exception:
                pass

            # Role coordination: try to keep at least one of the pair pressing if safe
            try:
                if getattr(self, 'roleCoordinationEnable', True) and 'mate_port' in locals():
                    # If I'm defensive/pump -> COVER
                    if ai.state in ("DEFENSIVE_BREAK_JINK", "PUMP_EXTEND"):
                        ai.role = "COVER"
                    # Evaluate mate threat
                    mate_is_threat = False
                    if mate_port in self.parents:
                        m_idx = list(self.parents.keys()).index(mate_port)
                        m_mws = self.mws[m_idx] if m_idx < len(self.mws) else []
                        m_motion = self.ourMotion[m_idx]
                        if len(m_mws) > 0 or self.detect_close_threat(m_motion)[0] is not None:
                            mate_is_threat = True
                    i_safe = (len(myMWS) == 0 and close_tgt is None and ai.state not in ("DEFENSIVE_BREAK_JINK", "PUMP_EXTEND") and V >= self.vEnergyMin)
                    if mate_is_threat and i_safe:
                        ai.role = "SHOOTER"
                    # If both safe and both COVER, lower port becomes SHOOTER
                    if ai.role == "COVER" and mate_ai is not None and getattr(mate_ai, 'role', 'SHOOTER') == "COVER" and i_safe:
                        try:
                            if int(port) < int(mate_port):
                                ai.role = "SHOOTER"
                        except Exception:
                            pass
            except Exception:
                pass

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
                    inhibitFire = True

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
                    ai.cheapShotCount = 0

            elif ai.state == "HIGH_ATTACK":
                # attackers: 予測迎撃ベクトルを基準に±crankのブラケット
                if i < 2:
                    if tgt and not tgt.is_none():
                        idir = self.compute_intercept_dir(myMotion, tgt)
                    else:
                        idir = np.array([cos(base_az), sin(base_az), 0.0])
                    bdir = self.heading_for_bracket(i, idir)
                    # base blend
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
                # Pre-seeker geometry shaping (Phase1/2): early crank/beam/drag before MWS
                if tgt and not tgt.is_none():
                    try:
                        dr2 = tgt.pos() - myMotion.pos()
                        rng2d = float(np.linalg.norm(dr2[:2]))
                    except Exception:
                        rng2d = 1e9
                    # strengthen crank when enemy likely nose-on and within dFar
                    if i < 2 and rng2d <= self.dFar:
                        try:
                            tv = tgt.vel(); sp = float(np.linalg.norm(tv[:2]))
                            toMe = myMotion.pos()[:2] - tgt.pos()[:2]
                            toMeN = float(np.linalg.norm(toMe))
                            cosNose = float(np.dot((tv[:2] / max(sp,1.0)), (toMe / max(toMeN,1.0)))) if sp >= 1.0 and toMeN >= 1.0 else 0.0
                        except Exception:
                            cosNose = 0.0
                        if cosNose > 0.6:
                            # crank bias stronger
                            idir = self.compute_intercept_dir(myMotion, tgt)
                            bdir = self.heading_for_bracket(i, idir)
                            ai.dstDir = self.blend_dir(bdir, idir, self.preCrankW)
                    # early beam near seeker distance
                    if self.earlyBeamEnable and rng2d <= self.dMid:
                        beam2d = self.compute_beam_dir(myMotion.vel(), dr2)
                        ai.dstDir = np.array([beam2d[0], beam2d[1], -0.08])

                # fire if inside normalized R threshold (safety-first gating applied)
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
                    # opening volley: be more permissive in first seconds
                    rThr = self.rShotThreshold
                    pkBias = dl_bias
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
                    # compute hard_defence flag (near or close threat)
                    hard_defence = False
                    try:
                        hard_defence = (rng2d <= max(20000.0, self.dMid)) or (close_tgt is not None)
                    except Exception:
                        pass
                    # do not fire when defensive or inhibited (allow in soft defence)
                    ok_to_fire = True
                    if self.defensiveNoFire and (ai.state in ("DEFENSIVE_BREAK_JINK", "PUMP_EXTEND")):
                        ok_to_fire = (ai.state == "DEFENSIVE_BREAK_JINK") and (not hard_defence)
                    if inhibitFire:
                        ok_to_fire = False
                    # Role gating: COVER does not fire if mate is SHOOTER
                    try:
                        if getattr(self, 'roleCoordinationEnable', True) and ai.role == "COVER" and mate_ai is not None and getattr(mate_ai, 'role', 'SHOOTER') == "SHOOTER":
                            ok_to_fire = False
                    except Exception:
                        pass
                    # main-shot gating using own radar lock and aspect (hot)
                    direct_ok = True
                    if self.mainShotRequireDirect:
                        try:
                            direct_ok = False
                            if self.ourObservables[my_idx].contains_p("/sensor/track"):
                                for tt in self.ourObservables[my_idx].at_p("/sensor/track"):
                                    try:
                                        if getattr(tt, 'truth', None) == getattr(tgt, 'truth', None):
                                            direct_ok = True
                                            break
                                    except Exception:
                                        pass
                        except Exception:
                            direct_ok = True
                    # aspect check (hot/quartering hot)
                    try:
                        vhat = self._normalize2d(myMotion.vel())
                        los = self._normalize2d(tgt.pos() - myMotion.pos())
                        cosHot = float(np.dot(vhat[:2], los[:2]))
                    except Exception:
                        cosHot = 1.0
                    # Energy/altitude-aware gating:
                    # - If below energy band, tighten range and reduce bias.
                    # - If inside energy band, slightly relax range and increase bias.
                    try:
                        pos = myMotion.pos()
                        alt = -float(pos[2]) if len(pos) > 2 else 0.0
                    except Exception:
                        alt = 0.0
                    V = float(np.linalg.norm(myMotion.vel()))
                    E_ok = (alt >= self.altEnergyMin and V >= self.vEnergyMin)
                    if not E_ok:
                        rThr = max(0.8, rThr * 0.97)
                        pkBias -= 0.05
                    else:
                        rThr = min(1.0, rThr * 1.02)
                        pkBias += 0.03
                    # Altitude advantage: if we are >1km above target, allow slightly longer shots;
                    # if we are significantly below, be more conservative.
                    try:
                        tpos = tgt.pos()
                        t_alt = -float(tpos[2]) if len(tpos) > 2 else alt
                        dalt = alt - t_alt
                    except Exception:
                        dalt = 0.0
                    alt_band = 1000.0
                    if dalt > alt_band:
                        rThr = min(1.0, rThr * 1.03)
                        pkBias += 0.02
                    elif dalt < -alt_band:
                        rThr = max(0.75, rThr * 0.97)
                        pkBias -= 0.02
                    # conservative under disadvantage: optionally hard block
                    if self.conservativeUnderDisadvantage and (not E_ok or dalt < -alt_band or len(myMWS) > 0 or close_tgt is not None):
                        ok_to_fire = False
                    # if not direct or not hot/energy, treat as cheap-shot: tighten further and budget per engagement
                    shot_is_cheap = (not direct_ok) or (cosHot < self.hotAspectCosMin) or (not E_ok)
                    if ok_to_fire and shot_is_cheap:
                        rThr = max(0.82, rThr * 0.96)
                        pkBias -= 0.06
                        try:
                            if self.actionInfos[pf].cheapShotCount >= getattr(self, 'cheapShotMaxPerEngage', 1):
                                ok_to_fire = False
                            if ok_to_fire and float(now - self.actionInfos[pf].lastCheapShotT) < (self.shotIntervalMin + getattr(self, 'cheapShotExtraInterval', 3.0)):
                                ok_to_fire = False
                        except Exception:
                            pass
                    if ok_to_fire and r < rThr and parent.isLaunchableAt(tgt) and flying < self.maxSimulShot and ok_interval and (ok_partner or pk >= max(0.8, self.pkThreshold)) and (pk + pkBias) >= self.pkThreshold:
                        ai.launchFlag = True
                        ai.target = tgt
                        ai.lastShotTimes[tgt.truth] = now
                        # event log: shot
                        try:
                            self.logger_event({
                                "type": "shot",
                                "t": float(now),
                                "fighter_port": port,
                                "side": self.getTeam(),
                                "state": ai.state,
                                "target_truth": getattr(tgt, 'truth', None),
                                "rng2d": float(rng2d),
                                "pk": float(pk),
                                "pkBias": float(pkBias),
                                "E_ok": bool(E_ok),
                                "cosHot": float(cosHot),
                                "direct_ok": bool(direct_ok),
                                "shot_type": "main" if (direct_ok and E_ok and cosHot >= self.hotAspectCosMin) else "cheap",
                            })
                        except Exception:
                            pass
                        if 'shot_is_cheap' in locals() and shot_is_cheap:
                            try:
                                self.actionInfos[pf].cheapShotCount += 1
                                self.actionInfos[pf].lastCheapShotT = now
                            except Exception:
                                pass
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
                    else:
                        # event log: inhibit (first occurrence per reason)
                        reason = None
                        if inhibitFire:
                            reason = "INHIBIT_FLAG"
                        elif ai.state in ("DEFENSIVE_BREAK_JINK", "PUMP_EXTEND"):
                            reason = "HARD_DEFENCE" if hard_defence else "DEFENSIVE"
                        elif not direct_ok or cosHot < self.hotAspectCosMin or not E_ok:
                            reason = "GATE_MAIN"
                        if reason and getattr(ai, 'lastInhibitReason', "") != reason:
                            ai.lastInhibitReason = reason
                            try:
                                self.logger_event({
                                    "type": "inhibit",
                                    "t": float(now),
                                    "fighter_port": port,
                                    "side": self.getTeam(),
                                    "reason": reason,
                                    "state": ai.state,
                                })
                            except Exception:
                                pass
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
                # Exit if distance/time criteria satisfied (AND if configured)
                cond_dist = (minR >= self.pumpExitR)
                cond_time = (dt >= self.pumpTime)
                exit_ok = (cond_dist and cond_time) if self.pumpExitAll else (cond_dist or cond_time)
                if exit_ok:
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

                # MWS directional correction: if available and configured
                mdir = None
                if self.mwsDirectionalMode and len(myMWS) > 0:
                    try:
                        mdir = np.zeros(3)
                        for m in myMWS:
                            d = m.dir()
                            mdir += d
                    except Exception:
                        mdir = None
                forward2d = self._normalize2d(myMotion.vel())

                if dxy >= farR:
                    # Far: crank to hold support while offsetting
                    base_dir = los2d
                    crank_dir = self.heading_for_bracket(i, base_dir)
                    ai.dstDir = self.blend_dir(base_dir, crank_dir, 0.6)
                elif dxy >= midR:
                    # Mid: classic beam + gentle descent
                    if mdir is not None and float(np.dot(forward2d[:2], mdir[:2])) < 0.0:
                        # missile from rear/side -> prefer drag
                        drag2d = -los2d
                        ai.dstDir = np.array([drag2d[0], drag2d[1], -max(0.25, self.jinkVz)])
                    else:
                        beam2d = self.compute_beam_dir(myMotion.vel(), th2d)
                        ai.dstDir = np.array([beam2d[0], beam2d[1], -abs(self.jinkVz)])
                else:
                    # Near: drag (turn tail and extend, slight descent with a bit more sink)
                    drag2d = -los2d
                    ai.dstDir = np.array([drag2d[0], drag2d[1], -max(0.25, self.jinkVz)])

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
                # Only recommit if time elapsed AND energy/picture are acceptable
                E_ok = (np.linalg.norm(myMotion.vel()) >= self.vEnergyMin) and ((-float(myMotion.pos()[2])) >= self.altEnergyMin)
                mws_ok = (float(now - ai.lastMwsT) > self.mwsInhibitT)
                no_close = (self.detect_close_threat(myMotion)[0] is None)
                if dt >= self.recommitTime and E_ok and mws_ok and no_close:
                    ai.state = "HIGH_ATTACK"
                    ai.stateEnterT = now
                    ai.cheapShotCount = 0

            # Ally re-separation: if nearest ally too close, push laterally
            try:
                myp2 = myMotion.pos()[:2]
                min_d = 1e18
                for j, om in enumerate(self.ourMotion[: len(self.parents)]):
                    if j == my_idx:
                        continue
                    op = om.pos()[:2]
                    d = float(np.linalg.norm(op - myp2))
                    if d < min_d:
                        min_d = d
                if min_d < self.allyMinSep:
                    fwd2 = self.teamOrigin.relBtoP(np.array([1.0, 0.0, 0.0]))
                    fhat2 = self._normalize2d(fwd2)
                    nhat2 = np.array([-fhat2[1], fhat2[0], 0.0])
                    ai.dstDir = self.blend_dir(ai.dstDir, nhat2, 0.3)
            except Exception:
                pass

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


# Attach helper functions from submodules to SADAgent for better modularity.
SADAgent.compute_threat = compute_threat
SADAgent.compute_vip_threat = compute_vip_threat
SADAgent.compute_combined_threat = compute_combined_threat
SADAgent.select_targets = select_targets
SADAgent.compute_pk = compute_pk
SADAgent.heading_for_bracket = heading_for_bracket
SADAgent._normalize2d = _normalize2d
SADAgent.blend_dir = blend_dir
SADAgent.compute_intercept_dir = compute_intercept_dir
SADAgent.soft_center_bias_dir = soft_center_bias_dir
SADAgent.compute_beam_dir = compute_beam_dir
SADAgent.detect_close_threat = detect_close_threat
