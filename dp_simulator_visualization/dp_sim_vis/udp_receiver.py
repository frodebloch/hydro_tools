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

    # Wave parameters
    wave: WaveSpectrumParams = field(default_factory=WaveSpectrumParams)
    swell: WaveSpectrumParams = field(default_factory=WaveSpectrumParams)
    random_seed: int = 42
    frequencies: list[float] = field(default_factory=list)
    directions: list[float] = field(default_factory=list)

    # Flag indicating wave params have changed
    wave_params_updated: bool = False

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
            # Note: lat/lon would need conversion to NED offsets relative to a
            # reference point. For now we keep NED as is (mock mode sets directly).

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

        # ── Floating platform data (extension — not yet in C++ VisInterface)
        # {"platformId":"...", "north":..., "east":..., "heading":..., ...}
        if "platformId" in msg:
            self.state.platform_north = msg.get("north", self.state.platform_north)
            self.state.platform_east = msg.get("east", self.state.platform_east)
            self.state.platform_heading = msg.get("heading", self.state.platform_heading)
            self.state.platform_roll = msg.get("roll", self.state.platform_roll)
            self.state.platform_pitch = msg.get("pitch", self.state.platform_pitch)
            self.state.platform_heave = msg.get("heave", self.state.platform_heave)

    def close(self):
        self._sock.close()
