"""UDP receiver — non-blocking listener for JSON messages from the
VisualisationInterface (or any compatible source).

Parses the Morild-protocol JSON messages published by the C++
VisualisationInterface and maintains a current state snapshot.
"""

import json
import socket
import time
from dataclasses import dataclass, field


@dataclass
class WaveSpectrumParams:
    significant_wave_height: float = 0.0
    peak_period: float = 0.0
    direction_deg: float = 0.0
    spreading_factor: float = 1000.0


@dataclass
class GangwayConfigData:
    """Static gangway configuration from posrefs."""
    base_x: float = 0.0    # forward from midship [m] (config body frame)
    base_y: float = 0.0    # starboard from centreline [m] (config body frame)
    base_z: float = 0.0    # downward from keel [m] (config body frame)
    max_height: float = 25.0
    min_length: float = 18.0
    max_length: float = 32.0


@dataclass
class GangwayStateData:
    """Dynamic gangway state from the gangway simulator."""
    total_length: float = 18.0   # boom length from rotation center to tip [m]
    height: float = 0.0          # tower height (rotation center above base) [m]
    slewing_angle: float = 180.0  # horizontal rotation [deg] (0=fwd, 90=stbd, 180=aft)
    boom_angle: float = 0.0      # vertical luffing angle [deg] (positive = up)
    state: int = 0               # 0=Parked, 1=Parking, 2=Moving, 3=Connecting, 4=Connected


@dataclass
class SimulatorState:
    """Latest state received from the dp_simulator."""

    # Simulation time
    sim_time: float = 0.0

    # Vessel state
    vessel_north: float = 0.0
    vessel_east: float = 0.0
    vessel_heading: float = 0.0
    vessel_roll: float = 0.0
    vessel_pitch: float = 0.0
    vessel_heave: float = 0.0

    # Floating platform state
    platform_north: float = 200.0  # default: 200m ahead
    platform_east: float = 0.0
    platform_heading: float = 0.0
    platform_roll: float = 0.0
    platform_pitch: float = 0.0
    platform_heave: float = 0.0

    # Wind
    wind_speed: float = 0.0
    wind_direction: float = 0.0

    # Turbine
    turbine_state: int = 0  # 0=operating, 1=shutdown, 2=idling
    platform_wind_speed: float = 0.0  # wind speed at platform (for rotor RPM)

    # Wave elevation from C++ simulator (at vessel LF position)
    sim_wave_elevation: float = 0.0

    # Drift forces [kN] and yaw moment [kN-m] (from VesselSimulatorForcesTopic)
    drift_surge_kn: float = 0.0
    drift_sway_kn: float = 0.0
    drift_yaw_knm: float = 0.0

    # Wind forces [kN] and yaw moment [kN-m]
    wind_surge_kn: float = 0.0
    wind_sway_kn: float = 0.0
    wind_yaw_knm: float = 0.0

    # Wave parameters
    wave: WaveSpectrumParams = field(default_factory=WaveSpectrumParams)
    swell: WaveSpectrumParams = field(default_factory=WaveSpectrumParams)
    random_seed: int = 42
    frequencies: list[float] = field(default_factory=list)
    directions: list[float] = field(default_factory=list)

    # Flag indicating wave params have changed
    wave_params_updated: bool = False

    # Gangway
    gangway_config: GangwayConfigData = field(default_factory=GangwayConfigData)
    gangway_state: GangwayStateData = field(default_factory=GangwayStateData)
    gangway_config_received: bool = False

    # Timestamps
    last_update: float = 0.0


class UdpReceiver:
    """Non-blocking UDP socket listener that parses Morild-protocol JSON."""

    def __init__(self, port: int = 9000, bind_address: str = "0.0.0.0"):
        self.state = SimulatorState()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind_address, port))
        self._sock.setblocking(False)
        self._buf_size = 65536

    def poll(self) -> bool:
        """Read all pending UDP messages. Returns True if any were received."""
        received = False
        while True:
            try:
                data, _addr = self._sock.recvfrom(self._buf_size)
                self._parse(data)
                received = True
            except BlockingIOError:
                break
            except Exception:
                break
        return received

    def _parse(self, data: bytes):
        """Parse a single JSON message."""
        try:
            msg = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        now = time.time()
        self.state.last_update = now

        # ── Vessel data: {"id":"...", "latlon":{...}, "heave":..., "yaw":..., ...}
        if "latlon" in msg:
            self.state.vessel_heading = msg.get("yaw", self.state.vessel_heading)
            self.state.vessel_roll = msg.get("roll", self.state.vessel_roll)
            self.state.vessel_pitch = msg.get("pitch", self.state.vessel_pitch)
            self.state.vessel_heave = msg.get("heave", self.state.vessel_heave)
            if "ned_north" in msg:
                self.state.vessel_north = msg["ned_north"]
                self.state.vessel_east = msg["ned_east"]
            if "waveElevation" in msg:
                self.state.sim_wave_elevation = msg["waveElevation"]
            if "driftForces" in msg:
                df = msg["driftForces"]
                self.state.drift_surge_kn = df.get("surge", 0.0)
                self.state.drift_sway_kn = df.get("sway", 0.0)
                self.state.drift_yaw_knm = df.get("yaw", 0.0)
            if "windForces" in msg:
                wf = msg["windForces"]
                self.state.wind_surge_kn = wf.get("surge", 0.0)
                self.state.wind_sway_kn = wf.get("sway", 0.0)
                self.state.wind_yaw_knm = wf.get("yaw", 0.0)

        # ── Simulation time: {"OceanSimulationTime": 123.4}
        if "OceanSimulationTime" in msg:
            self.state.sim_time = msg["OceanSimulationTime"]

        # ── Wind: {"WindDirection":..., "WindSpeed":...}
        if "WindDirection" in msg:
            self.state.wind_direction = msg["WindDirection"]
            self.state.wind_speed = msg.get("WindSpeed", self.state.wind_speed)

        # ── Wave parameters: {"frequencies":[...], "directions":[...],
        #                       "spectrums":[{Hs, Tp, dir, spreading}, ...],
        #                       "randomSeed": N}
        if "frequencies" in msg and "spectrums" in msg:
            self.state.frequencies = msg["frequencies"]
            self.state.directions = msg["directions"]
            self.state.random_seed = msg.get("randomSeed", self.state.random_seed)
            spectrums = msg["spectrums"]
            if len(spectrums) >= 1:
                s = spectrums[0]
                self.state.swell = WaveSpectrumParams(
                    significant_wave_height=s.get("significantWaveHeight", 0.0),
                    peak_period=s.get("peakPeriod", 0.0),
                    direction_deg=s.get("dominantDirection", 0.0),
                    spreading_factor=s.get("spreadingFactor", 7.0),
                )
            if len(spectrums) >= 2:
                s = spectrums[1]
                self.state.wave = WaveSpectrumParams(
                    significant_wave_height=s.get("significantWaveHeight", 0.0),
                    peak_period=s.get("peakPeriod", 0.0),
                    direction_deg=s.get("dominantDirection", 0.0),
                    spreading_factor=s.get("spreadingFactor", 2.0),
                )
            self.state.wave_params_updated = True

        # ── Floating platform data: {"platformId":"...", "north":..., "east":..., ...}
        if "platformId" in msg:
            self.state.platform_north = msg.get("north", self.state.platform_north)
            self.state.platform_east = msg.get("east", self.state.platform_east)
            self.state.platform_heading = msg.get("heading", self.state.platform_heading)
            self.state.platform_roll = msg.get("roll", self.state.platform_roll)
            self.state.platform_pitch = msg.get("pitch", self.state.platform_pitch)
            self.state.platform_heave = msg.get("heave", self.state.platform_heave)
            self.state.platform_wind_speed = msg.get("windSpeed", self.state.platform_wind_speed)
            self.state.turbine_state = msg.get("turbineState", self.state.turbine_state)

        # ── Gangway config: {"gangwayConfig":{"index":0, "baseX":..., ...}}
        if "gangwayConfig" in msg:
            gc = msg["gangwayConfig"]
            self.state.gangway_config = GangwayConfigData(
                base_x=gc.get("baseX", 0.0),
                base_y=gc.get("baseY", 0.0),
                base_z=gc.get("baseZ", 0.0),
                max_height=gc.get("maxHeight", 25.0),
                min_length=gc.get("minLength", 18.0),
                max_length=gc.get("maxLength", 32.0),
            )
            self.state.gangway_config_received = True

        # ── Gangway state: {"gangwayState":{"index":0, "state":4, ...}}
        if "gangwayState" in msg:
            gs = msg["gangwayState"]
            self.state.gangway_state = GangwayStateData(
                total_length=gs.get("totalLength", self.state.gangway_state.total_length),
                height=gs.get("height", self.state.gangway_state.height),
                slewing_angle=gs.get("slewingAngle", self.state.gangway_state.slewing_angle),
                boom_angle=gs.get("boomAngle", self.state.gangway_state.boom_angle),
                state=gs.get("state", self.state.gangway_state.state),
            )

    def close(self):
        self._sock.close()
