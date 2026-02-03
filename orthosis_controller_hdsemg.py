#!/usr/bin/env python3
"""
FOOT DROP ORTHOSIS - RASPBERRY PI EMG CONTROLLER
=================================================
VERSION: High-Density Surface EMG (HD-sEMG)

This script runs on a Raspberry Pi and handles:
  1. Acquiring EMG data from HD-sEMG array over TA + bipolar surface MG
  2. Processing EMG signals with spatial filtering for electrode shift robustness
  3. Running the trained ML model (trained on Myomatrix, deployed on HD-sEMG)
  4. Sending motor commands to the Arduino
  5. Logging data for analysis

HARDWARE REQUIREMENTS:
  - Raspberry Pi 4 or 5
  - HD-sEMG system (OT Bioelettronica, TMSi SAGA, or similar) via USB
  - 64-channel grid (8x8) over tibialis anterior
  - Bipolar surface electrode on medial gastrocnemius
  - Arduino connected via USB serial

USAGE:
  python orthosis_controller_hdsemg.py --model trained_model.pkl --port /dev/ttyACM0
"""

import argparse
import logging
import pickle
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import laplace
import serial


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Central configuration for all system parameters."""
    
    # HD-sEMG Acquisition
    emg_sample_rate: int = 2048
    grid_rows: int = 8
    grid_cols: int = 8
    emg_channel_mg: int = 1
    
    # Signal Processing
    bandpass_low: float = 20.0
    bandpass_high: float = 450.0
    notch_freq: float = 50.0
    notch_q: float = 30.0
    
    # Spatial Filtering
    use_laplacian: bool = True
    
    # Electrode Quality
    impedance_threshold_kohm: float = 50.0
    min_good_channels: int = 32
    
    # Feature Extraction  
    window_size_ms: int = 150
    window_step_ms: int = 20
    
    # Model
    model_path: str = "trained_model.pkl"
    
    # Arduino Communication
    arduino_port: str = "/dev/ttyACM0"
    arduino_baud: int = 115200
    
    # Control
    activation_threshold: float = 0.1
    activation_smoothing: float = 0.3
    max_activation: float = 1.0
    watchdog_interval_ms: int = 100
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    data_log_file: Optional[str] = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class GaitPhase(Enum):
    UNKNOWN = 0
    STANCE = 1
    SWING = 2


@dataclass
class ArduinoStatus:
    gait_phase: GaitPhase
    foot_drop_angle: float
    current_activation: float
    fault: str
    timestamp: float


@dataclass
class HDEMGSample:
    grid_data: np.ndarray
    mg_channel: float
    timestamp: float
    channel_quality: np.ndarray


@dataclass 
class EMGFeatures:
    ta_rms: float
    ta_mav: float
    ta_wl: float
    ta_zc: int
    ta_ssc: int
    mg_rms: float
    ta_mg_ratio: float
    ta_rms_spatial_std: float
    n_good_channels: int
    timestamp: float


# =============================================================================
# HD-sEMG ACQUISITION
# =============================================================================

class HDsEMGInterface:
    """
    Interface to High-Density surface EMG system.
    
    NOTE: Replace acquisition code with your specific HD-sEMG SDK:
    - OT Bioelettronica: Use OTBio Python SDK
    - TMSi SAGA: Use TMSi Python API
    - Delsys: Use Delsys SDK
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.emg_sample_rate
        self.grid_shape = (config.grid_rows, config.grid_cols)
        self.sample_buffer = deque(maxlen=self.sample_rate * 2)
        self.channel_quality = np.ones(self.grid_shape, dtype=bool)
        self.connected = False
        self.running = False
        self._acquisition_thread: Optional[threading.Thread] = None
        self._device = None
        self.logger = logging.getLogger("HDsEMG")
    
    def connect(self) -> bool:
        """Connect to HD-sEMG system. REPLACE with actual SDK code."""
        self.logger.info("Connecting to HD-sEMG system...")
        # TODO: Replace with actual connection code
        self.connected = True
        self.logger.info("HD-sEMG connected (SIMULATED)")
        return True
    
    def check_impedances(self) -> np.ndarray:
        """Check electrode impedances. REPLACE with actual SDK code."""
        # TODO: Replace with actual impedance measurement
        impedances = np.random.uniform(5, 30, self.grid_shape)
        self.channel_quality = impedances < self.config.impedance_threshold_kohm
        return impedances
    
    def start_acquisition(self):
        if not self.connected:
            raise RuntimeError("Must connect first")
        self.check_impedances()
        self.running = True
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_loop, daemon=True
        )
        self._acquisition_thread.start()
        self.logger.info("HD-sEMG acquisition started")
    
    def stop_acquisition(self):
        self.running = False
        if self._acquisition_thread:
            self._acquisition_thread.join(timeout=1.0)
    
    def _acquisition_loop(self):
        """Background acquisition. REPLACE data reading with actual SDK."""
        sample_interval = 1.0 / self.sample_rate
        next_sample_time = time.time()
        samples_since_check = 0
        
        while self.running:
            now = time.time()
            if now >= next_sample_time:
                # TODO: Replace with actual data reading
                grid_data = np.random.randn(*self.grid_shape) * 0.1
                mg_data = np.random.randn() * 0.05
                
                sample = HDEMGSample(
                    grid_data=grid_data,
                    mg_channel=mg_data,
                    timestamp=now,
                    channel_quality=self.channel_quality.copy()
                )
                self.sample_buffer.append(sample)
                
                samples_since_check += 1
                if samples_since_check >= self.sample_rate * 30:
                    self.check_impedances()
                    samples_since_check = 0
                
                next_sample_time += sample_interval
            else:
                time.sleep(0.0001)
    
    def get_window(self, window_size_samples: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if len(self.sample_buffer) < window_size_samples:
            return None
        recent = list(self.sample_buffer)[-window_size_samples:]
        grid_data = np.array([s.grid_data for s in recent])
        mg_data = np.array([s.mg_channel for s in recent])
        quality = recent[-1].channel_quality
        return grid_data, mg_data, quality
    
    def disconnect(self):
        self.stop_acquisition()
        self.connected = False
        self.logger.info("HD-sEMG disconnected")


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

class HDEMGProcessor:
    """EMG processing with spatial filtering for HD arrays."""
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.emg_sample_rate
        self.grid_shape = (config.grid_rows, config.grid_cols)
        self.logger = logging.getLogger("Processor")
        self._design_filters()
        
        n_ch = config.grid_rows * config.grid_cols
        self._bp_zi = [signal.lfilter_zi(self.bp_b, self.bp_a) * 0 for _ in range(n_ch)]
        self._bp_zi_mg = signal.lfilter_zi(self.bp_b, self.bp_a) * 0
        self._notch_zi = [signal.lfilter_zi(self.notch_b, self.notch_a) * 0 for _ in range(n_ch)]
        self._notch_zi_mg = signal.lfilter_zi(self.notch_b, self.notch_a) * 0
    
    def _design_filters(self):
        nyq = self.sample_rate / 2.0
        low, high = self.config.bandpass_low / nyq, self.config.bandpass_high / nyq
        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')
        self.notch_b, self.notch_a = signal.iirnotch(
            self.config.notch_freq / nyq, self.config.notch_q
        )
    
    def interpolate_bad_channels(self, grid_data: np.ndarray, 
                                  quality_mask: np.ndarray) -> np.ndarray:
        if np.all(quality_mask):
            return grid_data
        result = grid_data.copy()
        rows, cols = self.grid_shape
        for i in range(rows):
            for j in range(cols):
                if not quality_mask[i, j]:
                    neighbors = []
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and quality_mask[ni, nj]:
                            neighbors.append((ni, nj))
                    if neighbors:
                        result[:, i, j] = np.mean(
                            [grid_data[:, ni, nj] for ni, nj in neighbors], axis=0
                        )
                    else:
                        result[:, i, j] = 0
        return result
    
    def apply_laplacian(self, grid_data: np.ndarray) -> np.ndarray:
        result = np.zeros_like(grid_data)
        for t in range(grid_data.shape[0]):
            result[t] = laplace(grid_data[t])
        return result
    
    def filter_window(self, grid_data: np.ndarray, mg_data: np.ndarray,
                      quality_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grid_data = self.interpolate_bad_channels(grid_data, quality_mask)
        rows, cols = self.grid_shape
        filtered_grid = np.zeros_like(grid_data)
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                ch = grid_data[:, i, j]
                filt, self._bp_zi[idx] = signal.lfilter(self.bp_b, self.bp_a, ch, zi=self._bp_zi[idx])
                filt, self._notch_zi[idx] = signal.lfilter(self.notch_b, self.notch_a, filt, zi=self._notch_zi[idx])
                filtered_grid[:, i, j] = filt
        
        filt_mg, self._bp_zi_mg = signal.lfilter(self.bp_b, self.bp_a, mg_data, zi=self._bp_zi_mg)
        filt_mg, self._notch_zi_mg = signal.lfilter(self.notch_b, self.notch_a, filt_mg, zi=self._notch_zi_mg)
        
        if self.config.use_laplacian:
            filtered_grid = self.apply_laplacian(filtered_grid)
        
        return filtered_grid, filt_mg
    
    def extract_features(self, grid_data: np.ndarray, mg_data: np.ndarray,
                         quality_mask: np.ndarray) -> EMGFeatures:
        n_ch = self.grid_shape[0] * self.grid_shape[1]
        flat = grid_data.reshape(grid_data.shape[0], n_ch)
        flat_q = quality_mask.flatten()
        good_idx = np.where(flat_q)[0]
        n_good = len(good_idx)
        
        if n_good == 0:
            return EMGFeatures(0,0,0,0,0,0,0,0,0,time.time())
        
        good = flat[:, good_idx]
        rms_ch = np.sqrt(np.mean(good**2, axis=0))
        ta_rms = np.mean(rms_ch)
        ta_rms_std = np.std(rms_ch)
        ta_mav = np.mean(np.mean(np.abs(good), axis=0))
        ta_wl = np.mean(np.sum(np.abs(np.diff(good, axis=0)), axis=0))
        
        def zc(x): return np.sum(np.diff(np.signbit(x).astype(int)) != 0)
        def ssc(x): d = np.diff(x); return np.sum(d[:-1] * d[1:] < 0) if len(d) > 1 else 0
        
        ta_zc = int(np.mean([zc(good[:, c]) for c in range(n_good)]))
        ta_ssc = int(np.mean([ssc(good[:, c]) for c in range(n_good)]))
        mg_rms = np.sqrt(np.mean(mg_data**2))
        ta_mg_ratio = ta_rms / (mg_rms + 1e-6)
        
        return EMGFeatures(ta_rms, ta_mav, ta_wl, ta_zc, ta_ssc, mg_rms,
                          ta_mg_ratio, ta_rms_std, n_good, time.time())


# =============================================================================
# MODEL AND ARDUINO (simplified versions)
# =============================================================================

class ActivationModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger("Model")
    
    def load(self, path: str):
        self.logger.info(f"Loading model from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data.get('scaler')
        self.logger.info(f"Loaded: {type(self.model).__name__}")
    
    def predict(self, features: EMGFeatures) -> float:
        if not self.model:
            raise RuntimeError("Model not loaded")
        vec = np.array([[features.ta_rms, features.ta_mav, features.ta_wl,
                        features.ta_zc, features.ta_ssc, features.mg_rms,
                        features.ta_mg_ratio]])
        if self.scaler:
            vec = self.scaler.transform(vec)
        pred = self.model.predict(vec)[0]
        return float(np.clip(pred, 0.0, self.config.max_activation))


class ArduinoInterface:
    def __init__(self, config: Config):
        self.config = config
        self.serial: Optional[serial.Serial] = None
        self.last_status: Optional[ArduinoStatus] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer = ""
        self.logger = logging.getLogger("Arduino")
    
    def connect(self) -> bool:
        try:
            self.serial = serial.Serial(self.config.arduino_port, 
                                        self.config.arduino_baud, timeout=0.1)
            time.sleep(2.0)
            self.serial.reset_input_buffer()
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            self.logger.info(f"Connected: {self.config.arduino_port}")
            return True
        except serial.SerialException as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.serial:
            self.send_activation(0.0)
            time.sleep(0.1)
            self.serial.close()
    
    def send_activation(self, val: float):
        if self.serial and self.serial.is_open:
            try:
                self.serial.write(f"A{val:.3f}\n".encode())
            except serial.SerialException:
                pass
    
    def _read_loop(self):
        while self._running:
            if not self.serial or not self.serial.is_open:
                time.sleep(0.1)
                continue
            try:
                if self.serial.in_waiting:
                    self._buffer += self.serial.read(self.serial.in_waiting).decode('ascii', errors='ignore')
                    while '\n' in self._buffer:
                        line, self._buffer = self._buffer.split('\n', 1)
                        self._parse(line.strip())
                else:
                    time.sleep(0.005)
            except:
                time.sleep(0.1)
    
    def _parse(self, line: str):
        try:
            parts = dict(p.split(':') for p in line.split(','))
            gait = {'SWING': GaitPhase.SWING, 'STANCE': GaitPhase.STANCE}.get(
                parts.get('G', ''), GaitPhase.UNKNOWN)
            self.last_status = ArduinoStatus(
                gait, float(parts.get('D', 0)), float(parts.get('A', 0)),
                parts.get('F', ''), time.time())
        except:
            pass


# =============================================================================
# MAIN CONTROLLER
# =============================================================================

class OrthosisController:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("Controller")
        self.emg = HDsEMGInterface(config)
        self.processor = HDEMGProcessor(config)
        self.model = ActivationModel(config)
        self.arduino = ArduinoInterface(config)
        self.running = False
        self.smoothed = 0.0
        self.window_samples = int(config.window_size_ms * config.emg_sample_rate / 1000)
        self.data_log = None
        if config.data_log_file:
            self.data_log = open(config.data_log_file, 'w')
            self.data_log.write("timestamp,ta_rms,mg_rms,ta_mg_ratio,n_good,pred,smooth,gait,sent\n")
    
    def start(self):
        self.logger.info("Starting HD-sEMG controller...")
        self.model.load(self.config.model_path)
        if not self.emg.connect():
            raise RuntimeError("EMG connection failed")
        if not self.arduino.connect():
            raise RuntimeError("Arduino connection failed")
        self.emg.start_acquisition()
        time.sleep(self.config.window_size_ms / 1000 * 1.5)
        self.running = True
        self._loop()
    
    def stop(self):
        self.running = False
        self.arduino.send_activation(0.0)
        time.sleep(0.1)
        self.emg.disconnect()
        self.arduino.disconnect()
        if self.data_log:
            self.data_log.close()
        self.logger.info("Stopped")
    
    def _loop(self):
        interval = self.config.window_step_ms / 1000.0
        next_t = time.time()
        while self.running:
            now = time.time()
            if now >= next_t:
                self._step()
                next_t += interval
            time.sleep(max(0, min(next_t - time.time(), 0.001)))
    
    def _step(self):
        data = self.emg.get_window(self.window_samples)
        if not data:
            return
        grid, mg, quality = data
        if np.sum(quality) < self.config.min_good_channels:
            self.arduino.send_activation(0.0)
            return
        
        grid_f, mg_f = self.processor.filter_window(grid, mg, quality)
        feat = self.processor.extract_features(grid_f, mg_f, quality)
        pred = self.model.predict(feat)
        
        alpha = self.config.activation_smoothing
        self.smoothed = alpha * pred + (1 - alpha) * self.smoothed
        
        gait = self.arduino.last_status.gait_phase if self.arduino.last_status else GaitPhase.UNKNOWN
        send = self.smoothed if (gait == GaitPhase.SWING and 
                                 self.smoothed >= self.config.activation_threshold) else 0.0
        self.arduino.send_activation(send)
        
        if self.data_log:
            self.data_log.write(f"{feat.timestamp:.3f},{feat.ta_rms:.6f},{feat.mg_rms:.6f},"
                               f"{feat.ta_mg_ratio:.4f},{feat.n_good_channels},{pred:.4f},"
                               f"{self.smoothed:.4f},{gait.name},{send:.4f}\n")
            self.data_log.flush()


def main():
    parser = argparse.ArgumentParser(description='HD-sEMG Orthosis Controller')
    parser.add_argument('--model', '-m', default='trained_model.pkl')
    parser.add_argument('--arduino-port', '-p', default='/dev/ttyACM0')
    parser.add_argument('--log-data', '-d', default=None)
    parser.add_argument('--log-level', '-l', default='INFO')
    parser.add_argument('--notch-freq', type=float, default=50.0)
    args = parser.parse_args()
    
    config = Config(model_path=args.model, arduino_port=args.arduino_port,
                   data_log_file=args.log_data, log_level=args.log_level,
                   notch_freq=args.notch_freq)
    
    logging.basicConfig(level=getattr(logging, args.log_level),
                       format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    
    controller = OrthosisController(config)
    try:
        controller.start()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
