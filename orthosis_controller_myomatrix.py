#!/usr/bin/env python3
"""
FOOT DROP ORTHOSIS - RASPBERRY PI EMG CONTROLLER
=================================================
VERSION: Myomatrix Intramuscular EMG

This script runs on a Raspberry Pi and handles:
  1. Acquiring EMG data from Myomatrix intramuscular electrodes
  2. Processing EMG signals (filtering, feature extraction)
  3. Running the trained ML model to predict activation
  4. Sending motor commands to the Arduino
  5. Logging data for analysis

HARDWARE REQUIREMENTS:
  - Raspberry Pi 4 or 5 (Pi 3 may work but with higher latency)
  - Myomatrix EMG system connected via USB
  - Arduino connected via USB serial
  - Surface EMG electrode on medial gastrocnemius (via ADC or Myomatrix aux)

DEPENDENCIES:
  pip install numpy scipy scikit-learn pyserial

USAGE:
  python orthosis_controller_myomatrix.py --model trained_model.pkl --port /dev/ttyUSB0

Author: Chioke Swann
Date: 3 Feb 2026
"""

import argparse
import logging
import pickle
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy import signal
import serial

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Central configuration for all system parameters."""
    
    # EMG Acquisition
    emg_sample_rate: int = 2000          # Hz - Myomatrix typical rate
    emg_channels_ta: int = 32            # Number of TA channels (Myomatrix array)
    emg_channel_mg: int = 1              # Single channel for medial gastrocnemius
    
    # Signal Processing
    bandpass_low: float = 20.0           # Hz - high-pass cutoff
    bandpass_high: float = 450.0         # Hz - low-pass cutoff
    notch_freq: float = 50.0             # Hz - powerline frequency (50 EU, 60 US)
    notch_q: float = 30.0                # Quality factor for notch filter
    
    # Feature Extraction
    window_size_ms: int = 150            # ms - sliding window for features
    window_step_ms: int = 20             # ms - step between windows (50 Hz output)
    
    # Model
    model_path: str = "trained_model.pkl"
    
    # Arduino Communication
    arduino_port: str = "/dev/ttyACM0"
    arduino_baud: int = 115200
    
    # Control
    activation_threshold: float = 0.1    # Minimum activation to send
    activation_smoothing: float = 0.3    # Exponential smoothing factor
    
    # Safety
    max_activation: float = 1.0          # Maximum allowed activation
    watchdog_interval_ms: int = 100      # Send keepalive at least this often
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None       # Set to filename to log to file
    data_log_file: Optional[str] = None  # Set to filename to log raw data


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class GaitPhase(Enum):
    """Gait phase as reported by Arduino."""
    UNKNOWN = 0
    STANCE = 1
    SWING = 2


@dataclass
class ArduinoStatus:
    """Status message from Arduino."""
    gait_phase: GaitPhase
    foot_drop_angle: float
    current_activation: float
    fault: str
    timestamp: float


@dataclass
class EMGSample:
    """A single EMG sample from all channels."""
    ta_channels: np.ndarray    # Shape: (n_channels,)
    mg_channel: float
    timestamp: float


@dataclass 
class EMGFeatures:
    """Extracted features from an EMG window."""
    ta_rms: float              # RMS of TA (mean across channels)
    ta_mav: float              # Mean absolute value
    ta_wl: float               # Waveform length
    ta_zc: int                 # Zero crossings
    ta_ssc: int                # Slope sign changes
    mg_rms: float              # Gastrocnemius RMS
    ta_mg_ratio: float         # TA/MG activation ratio
    timestamp: float


# =============================================================================
# EMG ACQUISITION - MYOMATRIX INTERFACE
# =============================================================================

class MyomatrixInterface:
    """
    Interface to Myomatrix intramuscular EMG system.
    
    NOTE: This is a placeholder implementation. You will need to replace
    the acquisition code with the actual Myomatrix SDK/API calls for your
    specific hardware. The Myomatrix system may use:
      - USB serial communication
      - Custom USB protocol
      - Network streaming
    
    Contact Myomatrix support for their Python API documentation.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.emg_sample_rate
        self.n_channels = config.emg_channels_ta + config.emg_channel_mg
        
        # Buffer for incoming samples
        self.sample_buffer = deque(maxlen=self.sample_rate * 2)  # 2 seconds
        
        # Connection state
        self.connected = False
        self.running = False
        self._acquisition_thread: Optional[threading.Thread] = None
        
        # Placeholder for actual device handle
        self._device = None
        
        self.logger = logging.getLogger("Myomatrix")
    
    def connect(self) -> bool:
        """
        Connect to the Myomatrix system.
        
        REPLACE THIS with actual Myomatrix connection code.
        """
        self.logger.info("Connecting to Myomatrix system...")
        
        # =====================================================================
        # TODO: Replace with actual Myomatrix connection
        # Example (pseudocode):
        #   from myomatrix_sdk import MyomatrixDevice
        #   self._device = MyomatrixDevice()
        #   self._device.connect()
        #   self._device.set_sample_rate(self.sample_rate)
        # =====================================================================
        
        # For now, simulate connection
        self.connected = True
        self.logger.info("Myomatrix connected (SIMULATED - replace with real code)")
        return True
    
    def start_acquisition(self):
        """Start continuous EMG acquisition in background thread."""
        if not self.connected:
            raise RuntimeError("Must connect before starting acquisition")
        
        self.running = True
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_loop,
            daemon=True
        )
        self._acquisition_thread.start()
        self.logger.info("EMG acquisition started")
    
    def stop_acquisition(self):
        """Stop EMG acquisition."""
        self.running = False
        if self._acquisition_thread:
            self._acquisition_thread.join(timeout=1.0)
        self.logger.info("EMG acquisition stopped")
    
    def _acquisition_loop(self):
        """
        Background thread for continuous EMG acquisition.
        
        REPLACE the data reading code with actual Myomatrix SDK calls.
        """
        sample_interval = 1.0 / self.sample_rate
        next_sample_time = time.time()
        
        while self.running:
            now = time.time()
            
            if now >= next_sample_time:
                # =============================================================
                # TODO: Replace with actual Myomatrix data reading
                # Example (pseudocode):
                #   raw_data = self._device.read_samples(n_samples=1)
                #   ta_data = raw_data[:self.config.emg_channels_ta]
                #   mg_data = raw_data[self.config.emg_channels_ta]
                # =============================================================
                
                # SIMULATED DATA - Replace this block
                ta_data = np.random.randn(self.config.emg_channels_ta) * 0.1
                mg_data = np.random.randn() * 0.05
                # END SIMULATED DATA
                
                sample = EMGSample(
                    ta_channels=ta_data,
                    mg_channel=mg_data,
                    timestamp=now
                )
                self.sample_buffer.append(sample)
                
                next_sample_time += sample_interval
            else:
                # Sleep briefly to avoid spinning
                time.sleep(0.0001)
    
    def get_window(self, window_size_samples: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the most recent window of EMG data.
        
        Returns:
            Tuple of (ta_data, mg_data) where:
              - ta_data has shape (window_size, n_channels)
              - mg_data has shape (window_size,)
            Returns None if insufficient data available.
        """
        if len(self.sample_buffer) < window_size_samples:
            return None
        
        # Get most recent samples
        recent_samples = list(self.sample_buffer)[-window_size_samples:]
        
        ta_data = np.array([s.ta_channels for s in recent_samples])
        mg_data = np.array([s.mg_channel for s in recent_samples])
        
        return ta_data, mg_data
    
    def disconnect(self):
        """Disconnect from Myomatrix system."""
        self.stop_acquisition()
        
        # TODO: Add actual disconnect code
        # self._device.disconnect()
        
        self.connected = False
        self.logger.info("Myomatrix disconnected")


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

class EMGProcessor:
    """
    Real-time EMG signal processing.
    
    Applies:
      1. Bandpass filter (20-450 Hz) to remove motion artifacts and high-freq noise
      2. Notch filter to remove powerline interference
      3. Feature extraction for ML model input
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.emg_sample_rate
        self.logger = logging.getLogger("EMGProcessor")
        
        # Design filters
        self._design_filters()
        
        # Filter states for real-time processing (one per channel)
        self.n_ta_channels = config.emg_channels_ta
        self._bp_zi_ta = [signal.lfilter_zi(self.bp_b, self.bp_a) * 0 
                         for _ in range(self.n_ta_channels)]
        self._bp_zi_mg = signal.lfilter_zi(self.bp_b, self.bp_a) * 0
        self._notch_zi_ta = [signal.lfilter_zi(self.notch_b, self.notch_a) * 0
                            for _ in range(self.n_ta_channels)]
        self._notch_zi_mg = signal.lfilter_zi(self.notch_b, self.notch_a) * 0
    
    def _design_filters(self):
        """Design bandpass and notch filters."""
        nyq = self.sample_rate / 2.0
        
        # Bandpass filter (4th order Butterworth)
        low = self.config.bandpass_low / nyq
        high = self.config.bandpass_high / nyq
        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')
        
        # Notch filter for powerline
        notch_freq = self.config.notch_freq / nyq
        self.notch_b, self.notch_a = signal.iirnotch(
            notch_freq, self.config.notch_q
        )
        
        self.logger.debug(f"Filters designed: BP {self.config.bandpass_low}-"
                         f"{self.config.bandpass_high} Hz, Notch {self.config.notch_freq} Hz")
    
    def filter_window(self, ta_data: np.ndarray, mg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply filters to a window of EMG data.
        
        Args:
            ta_data: Shape (window_size, n_channels)
            mg_data: Shape (window_size,)
        
        Returns:
            Filtered (ta_data, mg_data) with same shapes
        """
        filtered_ta = np.zeros_like(ta_data)
        
        # Filter each TA channel
        for ch in range(self.n_ta_channels):
            # Bandpass
            filtered, self._bp_zi_ta[ch] = signal.lfilter(
                self.bp_b, self.bp_a, ta_data[:, ch], zi=self._bp_zi_ta[ch]
            )
            # Notch
            filtered, self._notch_zi_ta[ch] = signal.lfilter(
                self.notch_b, self.notch_a, filtered, zi=self._notch_zi_ta[ch]
            )
            filtered_ta[:, ch] = filtered
        
        # Filter MG channel
        filtered_mg, self._bp_zi_mg = signal.lfilter(
            self.bp_b, self.bp_a, mg_data, zi=self._bp_zi_mg
        )
        filtered_mg, self._notch_zi_mg = signal.lfilter(
            self.notch_b, self.notch_a, filtered_mg, zi=self._notch_zi_mg
        )
        
        return filtered_ta, filtered_mg
    
    def extract_features(self, ta_data: np.ndarray, mg_data: np.ndarray) -> EMGFeatures:
        """
        Extract features from filtered EMG window.
        
        Args:
            ta_data: Filtered TA data, shape (window_size, n_channels)
            mg_data: Filtered MG data, shape (window_size,)
        
        Returns:
            EMGFeatures dataclass with extracted features
        """
        # TA features (compute per channel, then average)
        # RMS - Root Mean Square
        ta_rms_per_ch = np.sqrt(np.mean(ta_data ** 2, axis=0))
        ta_rms = np.mean(ta_rms_per_ch)
        
        # MAV - Mean Absolute Value
        ta_mav_per_ch = np.mean(np.abs(ta_data), axis=0)
        ta_mav = np.mean(ta_mav_per_ch)
        
        # WL - Waveform Length (sum of absolute differences)
        ta_wl_per_ch = np.sum(np.abs(np.diff(ta_data, axis=0)), axis=0)
        ta_wl = np.mean(ta_wl_per_ch)
        
        # ZC - Zero Crossings (average across channels)
        def count_zero_crossings(x):
            return np.sum(np.diff(np.signbit(x).astype(int)) != 0)
        ta_zc = int(np.mean([count_zero_crossings(ta_data[:, ch]) 
                            for ch in range(ta_data.shape[1])]))
        
        # SSC - Slope Sign Changes
        def count_slope_changes(x):
            diff1 = np.diff(x)
            return np.sum(diff1[:-1] * diff1[1:] < 0)
        ta_ssc = int(np.mean([count_slope_changes(ta_data[:, ch])
                             for ch in range(ta_data.shape[1])]))
        
        # MG features
        mg_rms = np.sqrt(np.mean(mg_data ** 2))
        
        # TA/MG ratio (with small epsilon to avoid division by zero)
        epsilon = 1e-6
        ta_mg_ratio = ta_rms / (mg_rms + epsilon)
        
        return EMGFeatures(
            ta_rms=ta_rms,
            ta_mav=ta_mav,
            ta_wl=ta_wl,
            ta_zc=ta_zc,
            ta_ssc=ta_ssc,
            mg_rms=mg_rms,
            ta_mg_ratio=ta_mg_ratio,
            timestamp=time.time()
        )


# =============================================================================
# ML MODEL INTERFACE
# =============================================================================

class ActivationModel:
    """
    Wrapper for the trained ML model that predicts activation from EMG features.
    
    The model should be trained separately using the training script and saved
    as a pickle file. Expected input: feature vector. Expected output: 
    activation magnitude (0.0 to 1.0).
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None  # Optional: feature scaler if used during training
        self.logger = logging.getLogger("Model")
    
    def load(self, model_path: str):
        """
        Load trained model from pickle file.
        
        Expected pickle structure:
            {
                'model': trained sklearn model,
                'scaler': fitted StandardScaler (optional),
                'feature_names': list of feature names,
                'metadata': dict with training info
            }
        """
        self.logger.info(f"Loading model from {model_path}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data.get('scaler', None)
        self.feature_names = data.get('feature_names', [])
        
        self.logger.info(f"Model loaded: {type(self.model).__name__}")
        if self.scaler:
            self.logger.info("Feature scaler loaded")
    
    def predict(self, features: EMGFeatures) -> float:
        """
        Predict activation magnitude from EMG features.
        
        Args:
            features: Extracted EMG features
        
        Returns:
            Activation magnitude (0.0 to 1.0)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded - call load() first")
        
        # Convert features to array
        # Order must match training!
        feature_vector = np.array([
            features.ta_rms,
            features.ta_mav,
            features.ta_wl,
            features.ta_zc,
            features.ta_ssc,
            features.mg_rms,
            features.ta_mg_ratio
        ]).reshape(1, -1)
        
        # Scale if scaler is available
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.model.predict(feature_vector)[0]
        
        # Clamp to valid range
        activation = float(np.clip(prediction, 0.0, self.config.max_activation))
        
        return activation


# =============================================================================
# ARDUINO COMMUNICATION
# =============================================================================

class ArduinoInterface:
    """
    Serial communication with Arduino motor controller.
    
    Protocol:
        To Arduino:   "A0.75\n" - Set activation to 75%
                      "S\n"     - Request status
        From Arduino: "G:SWING,D:-12.5,A:0.65,F:NONE\n"
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.serial: Optional[serial.Serial] = None
        self.logger = logging.getLogger("Arduino")
        
        # Latest status from Arduino
        self.last_status: Optional[ArduinoStatus] = None
        
        # Background reading thread
        self._running = False
        self._read_thread: Optional[threading.Thread] = None
        self._read_buffer = ""
    
    def connect(self) -> bool:
        """Connect to Arduino via serial port."""
        try:
            self.serial = serial.Serial(
                port=self.config.arduino_port,
                baudrate=self.config.arduino_baud,
                timeout=0.1
            )
            
            # Wait for Arduino to reset after serial connection
            time.sleep(2.0)
            
            # Clear any startup messages
            self.serial.reset_input_buffer()
            
            # Start background read thread
            self._running = True
            self._read_thread = threading.Thread(
                target=self._read_loop,
                daemon=True
            )
            self._read_thread.start()
            
            self.logger.info(f"Connected to Arduino on {self.config.arduino_port}")
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino."""
        self._running = False
        if self._read_thread:
            self._read_thread.join(timeout=1.0)
        
        if self.serial:
            # Send stop command before disconnecting
            self.send_activation(0.0)
            time.sleep(0.1)
            self.serial.close()
            self.serial = None
        
        self.logger.info("Disconnected from Arduino")
    
    def send_activation(self, activation: float):
        """
        Send activation command to Arduino.
        
        Args:
            activation: Desired activation magnitude (0.0 to 1.0)
        """
        if not self.serial or not self.serial.is_open:
            return
        
        # Format command
        cmd = f"A{activation:.3f}\n"
        
        try:
            self.serial.write(cmd.encode('ascii'))
        except serial.SerialException as e:
            self.logger.error(f"Failed to send command: {e}")
    
    def request_status(self):
        """Request status update from Arduino."""
        if not self.serial or not self.serial.is_open:
            return
        
        try:
            self.serial.write(b"S\n")
        except serial.SerialException as e:
            self.logger.error(f"Failed to request status: {e}")
    
    def _read_loop(self):
        """Background thread for reading Arduino messages."""
        while self._running:
            if not self.serial or not self.serial.is_open:
                time.sleep(0.1)
                continue
            
            try:
                # Read available data
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting).decode('ascii', errors='ignore')
                    self._read_buffer += data
                    
                    # Process complete lines
                    while '\n' in self._read_buffer:
                        line, self._read_buffer = self._read_buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._parse_status(line)
                else:
                    time.sleep(0.005)  # Brief sleep to avoid spinning
                    
            except serial.SerialException as e:
                self.logger.error(f"Serial read error: {e}")
                time.sleep(0.1)
    
    def _parse_status(self, line: str):
        """Parse status message from Arduino."""
        # Expected format: "G:SWING,D:-12.5,A:0.65,F:NONE"
        try:
            parts = dict(part.split(':') for part in line.split(','))
            
            # Parse gait phase
            gait_str = parts.get('G', 'UNKNOWN')
            if gait_str == 'SWING':
                gait = GaitPhase.SWING
            elif gait_str == 'STANCE':
                gait = GaitPhase.STANCE
            else:
                gait = GaitPhase.UNKNOWN
            
            self.last_status = ArduinoStatus(
                gait_phase=gait,
                foot_drop_angle=float(parts.get('D', 0)),
                current_activation=float(parts.get('A', 0)),
                fault=parts.get('F', 'UNKNOWN'),
                timestamp=time.time()
            )
            
        except (ValueError, KeyError) as e:
            # Not a status message - might be startup message
            self.logger.debug(f"Unparseable message from Arduino: {line}")


# =============================================================================
# MAIN CONTROLLER
# =============================================================================

class OrthosisController:
    """
    Main controller that orchestrates all components.
    
    Control loop:
        1. Get EMG window from Myomatrix
        2. Filter and extract features
        3. Run ML model to get activation
        4. Send activation to Arduino (gated by gait phase)
        5. Log data
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("Controller")
        
        # Initialize components
        self.emg_interface = MyomatrixInterface(config)
        self.emg_processor = EMGProcessor(config)
        self.model = ActivationModel(config)
        self.arduino = ArduinoInterface(config)
        
        # Control state
        self.running = False
        self.smoothed_activation = 0.0
        
        # Calculate window size in samples
        self.window_samples = int(
            config.window_size_ms * config.emg_sample_rate / 1000
        )
        self.step_samples = int(
            config.window_step_ms * config.emg_sample_rate / 1000
        )
        
        # Data logging
        self.data_log_file = None
        if config.data_log_file:
            self.data_log_file = open(config.data_log_file, 'w')
            self.data_log_file.write(
                "timestamp,ta_rms,ta_mav,mg_rms,ta_mg_ratio,prediction,"
                "smoothed,gait_phase,sent_activation\n"
            )
    
    def start(self):
        """Initialize all components and start control loop."""
        self.logger.info("Starting orthosis controller...")
        
        # Load ML model
        self.model.load(self.config.model_path)
        
        # Connect to hardware
        if not self.emg_interface.connect():
            raise RuntimeError("Failed to connect to Myomatrix")
        
        if not self.arduino.connect():
            raise RuntimeError("Failed to connect to Arduino")
        
        # Start EMG acquisition
        self.emg_interface.start_acquisition()
        
        # Wait for buffer to fill
        self.logger.info("Waiting for EMG buffer to fill...")
        time.sleep(self.config.window_size_ms / 1000 * 1.5)
        
        # Run control loop
        self.running = True
        self.logger.info("Starting control loop")
        self._control_loop()
    
    def stop(self):
        """Stop controller and cleanup."""
        self.logger.info("Stopping orthosis controller...")
        self.running = False
        
        # Send stop command to Arduino
        self.arduino.send_activation(0.0)
        time.sleep(0.1)
        
        # Disconnect components
        self.emg_interface.disconnect()
        self.arduino.disconnect()
        
        # Close data log
        if self.data_log_file:
            self.data_log_file.close()
        
        self.logger.info("Controller stopped")
    
    def _control_loop(self):
        """Main control loop running at ~50 Hz."""
        loop_interval = self.config.window_step_ms / 1000.0
        next_loop_time = time.time()
        
        last_watchdog_time = time.time()
        
        while self.running:
            now = time.time()
            
            if now >= next_loop_time:
                try:
                    self._control_step()
                    last_watchdog_time = now
                except Exception as e:
                    self.logger.error(f"Control step error: {e}")
                
                next_loop_time += loop_interval
            
            # Watchdog: send zero activation if control loop stalls
            if now - last_watchdog_time > self.config.watchdog_interval_ms / 1000:
                self.arduino.send_activation(0.0)
                last_watchdog_time = now
            
            # Brief sleep to avoid spinning
            sleep_time = next_loop_time - time.time()
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.001))
    
    def _control_step(self):
        """Single iteration of control loop."""
        # Get EMG window
        emg_data = self.emg_interface.get_window(self.window_samples)
        if emg_data is None:
            return  # Not enough data yet
        
        ta_data, mg_data = emg_data
        
        # Filter
        ta_filtered, mg_filtered = self.emg_processor.filter_window(ta_data, mg_data)
        
        # Extract features
        features = self.emg_processor.extract_features(ta_filtered, mg_filtered)
        
        # Predict activation
        raw_activation = self.model.predict(features)
        
        # Smooth activation (exponential moving average)
        alpha = self.config.activation_smoothing
        self.smoothed_activation = (
            alpha * raw_activation + 
            (1 - alpha) * self.smoothed_activation
        )
        
        # Get gait phase from Arduino
        gait_phase = GaitPhase.UNKNOWN
        if self.arduino.last_status:
            gait_phase = self.arduino.last_status.gait_phase
        
        # Determine activation to send
        # Only activate during swing phase, and only above threshold
        if gait_phase == GaitPhase.SWING and \
           self.smoothed_activation >= self.config.activation_threshold:
            send_activation = self.smoothed_activation
        else:
            send_activation = 0.0
        
        # Send to Arduino
        self.arduino.send_activation(send_activation)
        
        # Log data
        if self.data_log_file:
            self.data_log_file.write(
                f"{features.timestamp:.3f},{features.ta_rms:.6f},"
                f"{features.ta_mav:.6f},{features.mg_rms:.6f},"
                f"{features.ta_mg_ratio:.6f},{raw_activation:.4f},"
                f"{self.smoothed_activation:.4f},{gait_phase.name},{send_activation:.4f}\n"
            )
            self.data_log_file.flush()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def setup_logging(config: Config):
    """Configure logging."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(
        description='Foot Drop Orthosis Controller - Myomatrix EMG Version'
    )
    parser.add_argument(
        '--model', '-m',
        default='trained_model.pkl',
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--arduino-port', '-p',
        default='/dev/ttyACM0',
        help='Arduino serial port'
    )
    parser.add_argument(
        '--log-data', '-d',
        default=None,
        help='Path for data log CSV file'
    )
    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        model_path=args.model,
        arduino_port=args.arduino_port,
        data_log_file=args.log_data,
        log_level=args.log_level
    )
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger("Main")
    
    # Create and run controller
    controller = OrthosisController(config)
    
    try:
        controller.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
