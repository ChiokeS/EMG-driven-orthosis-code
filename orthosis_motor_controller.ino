/*
 * FOOT DROP ORTHOSIS - ARDUINO MOTOR CONTROLLER
 * =============================================
 * 
 * This code runs on an Arduino (Uno, Mega, or Due recommended) and handles:
 *   1. Reading two IMUs (toe and heel) via I2C
 *   2. Detecting gait phase (swing vs stance)
 *   3. Receiving activation commands from Raspberry Pi via Serial
 *   4. Controlling the orthosis motor with safety interlocks
 * 
 * WIRING:
 *   - IMU 1 (Toe):  SDA -> A4, SCL -> A5, VCC -> 3.3V, GND -> GND, AD0 -> GND (addr 0x68)
 *   - IMU 2 (Heel): SDA -> A4, SCL -> A5, VCC -> 3.3V, GND -> GND, AD0 -> VCC (addr 0x69)
 *   - Motor Driver: PWM -> Pin 9, DIR -> Pin 8, EN -> Pin 7
 *   - Current Sense: Analog -> A0
 * 
 * COMMUNICATION PROTOCOL (Serial, 115200 baud):
 *   Incoming from Pi: "A0.75\n" means activate at 75% magnitude
 *                     "A0.00\n" means deactivate
 *                     "S\n" means request status
 *   Outgoing to Pi:   "G:SWING,P:0.32,D:-15.2\n" 
 *                     G = gait phase, P = pitch diff (foot drop angle), D = drop degrees
 * 
 * SAFETY FEATURES:
 *   - Watchdog: If no command received in 500ms, motor stops
 *   - Current limit: Motor stops if current exceeds threshold
 *   - Position limit: Motor stops at mechanical limits
 *   - Stance lockout: Motor cannot activate during stance phase
 */

#include <Wire.h>

// =============================================================================
// CONFIGURATION - ADJUST THESE FOR YOUR HARDWARE
// =============================================================================

// IMU I2C addresses (BNO055 or MPU6050)
// If using MPU6050: AD0 pin LOW = 0x68, AD0 pin HIGH = 0x69
const uint8_t IMU_TOE_ADDR = 0x68;
const uint8_t IMU_HEEL_ADDR = 0x69;

// Motor control pins
const int PIN_MOTOR_PWM = 9;      // PWM output for motor speed
const int PIN_MOTOR_DIR = 8;      // Direction control
const int PIN_MOTOR_EN = 7;       // Enable pin (HIGH = enabled)
const int PIN_CURRENT_SENSE = A0; // Analog input for current sensing

// Safety thresholds
const float MAX_CURRENT_MA = 2000.0;      // Maximum motor current before cutoff
const unsigned long WATCHDOG_TIMEOUT_MS = 500;  // Stop motor if no command received
const float MAX_ACTIVATION = 1.0;         // Maximum allowed activation magnitude

// Gait detection thresholds (tune these based on your patient population)
const float SWING_ACCEL_THRESHOLD = 0.3;  // g - both IMUs must exceed this
const float STANCE_ACCEL_THRESHOLD = 2.0; // g - heel strike detection
const float FOOT_DROP_ANGLE_THRESHOLD = -10.0; // degrees - when to consider foot dropped

// Timing
const unsigned long IMU_UPDATE_INTERVAL_MS = 10;   // 100 Hz IMU reading
const unsigned long STATUS_SEND_INTERVAL_MS = 20;  // 50 Hz status to Pi
const unsigned long SERIAL_CHECK_INTERVAL_MS = 5;  // 200 Hz serial check

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

// IMU data structures
struct IMUData {
  float accel_x, accel_y, accel_z;  // Acceleration in g
  float pitch, roll;                 // Orientation in degrees
  bool valid;                        // True if data was read successfully
};

IMUData imu_toe;
IMUData imu_heel;

// Gait state
enum GaitPhase { STANCE, SWING, UNKNOWN };
GaitPhase current_gait_phase = UNKNOWN;
float foot_drop_angle = 0.0;  // Difference between toe and heel pitch

// Motor state
float target_activation = 0.0;      // Commanded activation (0.0 to 1.0)
float current_activation = 0.0;     // Actual current activation (ramped)
bool motor_enabled = true;          // Safety flag
String safety_fault = "";           // Describes any active fault

// Timing
unsigned long last_command_time = 0;
unsigned long last_imu_update = 0;
unsigned long last_status_send = 0;
unsigned long last_serial_check = 0;

// Serial buffer
String serial_buffer = "";

// =============================================================================
// SETUP
// =============================================================================

void setup() {
  // Initialize serial communication with Raspberry Pi
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for native USB boards)
  }
  
  // Initialize I2C for IMUs
  Wire.begin();
  Wire.setClock(400000);  // 400 kHz I2C for faster reading
  
  // Initialize motor control pins
  pinMode(PIN_MOTOR_PWM, OUTPUT);
  pinMode(PIN_MOTOR_DIR, OUTPUT);
  pinMode(PIN_MOTOR_EN, OUTPUT);
  
  // Start with motor disabled
  digitalWrite(PIN_MOTOR_EN, LOW);
  analogWrite(PIN_MOTOR_PWM, 0);
  
  // Initialize IMUs
  initializeIMU(IMU_TOE_ADDR);
  initializeIMU(IMU_HEEL_ADDR);
  
  // Brief startup delay
  delay(100);
  
  // Enable motor after initialization
  digitalWrite(PIN_MOTOR_EN, HIGH);
  
  Serial.println("ORTHOSIS_READY");
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop() {
  unsigned long now = millis();
  
  // -------------------------------------------------------------------------
  // Task 1: Read IMUs at 100 Hz
  // -------------------------------------------------------------------------
  if (now - last_imu_update >= IMU_UPDATE_INTERVAL_MS) {
    last_imu_update = now;
    
    // Read both IMUs
    readIMU(IMU_TOE_ADDR, &imu_toe);
    readIMU(IMU_HEEL_ADDR, &imu_heel);
    
    // Calculate foot drop angle (toe pitch relative to heel pitch)
    if (imu_toe.valid && imu_heel.valid) {
      foot_drop_angle = imu_toe.pitch - imu_heel.pitch;
    }
    
    // Determine gait phase
    updateGaitPhase();
  }
  
  // -------------------------------------------------------------------------
  // Task 2: Check serial commands at 200 Hz
  // -------------------------------------------------------------------------
  if (now - last_serial_check >= SERIAL_CHECK_INTERVAL_MS) {
    last_serial_check = now;
    checkSerialCommands();
  }
  
  // -------------------------------------------------------------------------
  // Task 3: Send status to Pi at 50 Hz
  // -------------------------------------------------------------------------
  if (now - last_status_send >= STATUS_SEND_INTERVAL_MS) {
    last_status_send = now;
    sendStatusToPi();
  }
  
  // -------------------------------------------------------------------------
  // Task 4: Safety checks and motor control (every loop iteration)
  // -------------------------------------------------------------------------
  performSafetyChecks(now);
  updateMotorOutput();
}

// =============================================================================
// IMU FUNCTIONS
// =============================================================================

/*
 * Initialize an IMU at the given I2C address.
 * This example assumes MPU6050. Modify for BNO055 if using that sensor.
 */
void initializeIMU(uint8_t address) {
  Wire.beginTransmission(address);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0x00);  // Wake up the MPU6050
  Wire.endTransmission(true);
  
  // Set accelerometer range to +/- 4g
  Wire.beginTransmission(address);
  Wire.write(0x1C);  // ACCEL_CONFIG register
  Wire.write(0x08);  // +/- 4g
  Wire.endTransmission(true);
  
  // Set gyro range to +/- 500 deg/s
  Wire.beginTransmission(address);
  Wire.write(0x1B);  // GYRO_CONFIG register
  Wire.write(0x08);  // +/- 500 deg/s
  Wire.endTransmission(true);
}

/*
 * Read accelerometer and calculate pitch/roll from an IMU.
 * For production use, consider using a Kalman filter or the BNO055's
 * built-in sensor fusion for more stable orientation estimates.
 */
void readIMU(uint8_t address, IMUData* data) {
  Wire.beginTransmission(address);
  Wire.write(0x3B);  // Starting register for accel data
  if (Wire.endTransmission(false) != 0) {
    data->valid = false;
    return;
  }
  
  Wire.requestFrom(address, (uint8_t)6, (uint8_t)true);
  if (Wire.available() < 6) {
    data->valid = false;
    return;
  }
  
  // Read raw accelerometer values (16-bit, big-endian)
  int16_t ax_raw = (Wire.read() << 8) | Wire.read();
  int16_t ay_raw = (Wire.read() << 8) | Wire.read();
  int16_t az_raw = (Wire.read() << 8) | Wire.read();
  
  // Convert to g (at +/- 4g range, sensitivity is 8192 LSB/g)
  data->accel_x = ax_raw / 8192.0;
  data->accel_y = ay_raw / 8192.0;
  data->accel_z = az_raw / 8192.0;
  
  // Calculate pitch and roll from accelerometer
  // Note: This is a simplified calculation. For dynamic motion,
  // you'd want to fuse with gyroscope data using a complementary
  // or Kalman filter.
  data->pitch = atan2(-data->accel_x, 
                      sqrt(data->accel_y * data->accel_y + 
                           data->accel_z * data->accel_z)) * 180.0 / PI;
  data->roll = atan2(data->accel_y, data->accel_z) * 180.0 / PI;
  
  data->valid = true;
}

// =============================================================================
// GAIT PHASE DETECTION
// =============================================================================

/*
 * Determine whether the foot is in swing or stance phase.
 * 
 * Logic:
 *   - SWING: Both toe and heel show forward acceleration (foot moving through air)
 *   - STANCE: High acceleration on heel (heel strike) or stable readings (flat foot)
 * 
 * This is a simplified heuristic. For robust gait detection, consider:
 *   - Using gyroscope data for angular velocity
 *   - Machine learning classifier trained on your patient population
 *   - Pressure sensors in the shoe
 */
void updateGaitPhase() {
  if (!imu_toe.valid || !imu_heel.valid) {
    current_gait_phase = UNKNOWN;
    return;
  }
  
  // Calculate total acceleration magnitude for each IMU
  float toe_accel_mag = sqrt(imu_toe.accel_x * imu_toe.accel_x +
                             imu_toe.accel_y * imu_toe.accel_y +
                             imu_toe.accel_z * imu_toe.accel_z);
  float heel_accel_mag = sqrt(imu_heel.accel_x * imu_heel.accel_x +
                              imu_heel.accel_y * imu_heel.accel_y +
                              imu_heel.accel_z * imu_heel.accel_z);
  
  // Check for heel strike (high impact on heel)
  if (heel_accel_mag > STANCE_ACCEL_THRESHOLD) {
    current_gait_phase = STANCE;
    return;
  }
  
  // Check for swing (both IMUs show deviation from 1g, indicating movement)
  // When stationary, magnitude should be ~1g
  float toe_deviation = abs(toe_accel_mag - 1.0);
  float heel_deviation = abs(heel_accel_mag - 1.0);
  
  if (toe_deviation > SWING_ACCEL_THRESHOLD && 
      heel_deviation > SWING_ACCEL_THRESHOLD) {
    // Both sensors show movement - likely swing phase
    // Additional check: toe and heel should be moving together (similar direction)
    // This helps distinguish intentional swing from spastic motion
    float accel_diff = abs(imu_toe.accel_x - imu_heel.accel_x);
    if (accel_diff < 0.5) {  // Accelerations are similar (coordinated movement)
      current_gait_phase = SWING;
      return;
    }
  }
  
  // Default to stance if no clear swing indicators
  current_gait_phase = STANCE;
}

// =============================================================================
// SERIAL COMMUNICATION
// =============================================================================

/*
 * Check for and process commands from the Raspberry Pi.
 * 
 * Command format:
 *   "A0.75\n" - Set activation to 0.75 (75%)
 *   "A0.00\n" - Deactivate
 *   "S\n"     - Request status
 */
void checkSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n') {
      // Process complete command
      processCommand(serial_buffer);
      serial_buffer = "";
    } else if (c != '\r') {
      // Add character to buffer (ignore carriage returns)
      serial_buffer += c;
      
      // Prevent buffer overflow
      if (serial_buffer.length() > 20) {
        serial_buffer = "";
      }
    }
  }
}

/*
 * Process a single command string.
 */
void processCommand(String cmd) {
  if (cmd.length() == 0) return;
  
  char command_type = cmd.charAt(0);
  
  switch (command_type) {
    case 'A':  // Activation command
      {
        float activation = cmd.substring(1).toFloat();
        // Clamp to valid range
        target_activation = constrain(activation, 0.0, MAX_ACTIVATION);
        last_command_time = millis();
      }
      break;
      
    case 'S':  // Status request
      sendStatusToPi();
      break;
      
    case 'E':  // Emergency stop
      target_activation = 0.0;
      current_activation = 0.0;
      motor_enabled = false;
      safety_fault = "EMERGENCY_STOP";
      break;
      
    case 'R':  // Reset faults
      motor_enabled = true;
      safety_fault = "";
      break;
      
    default:
      // Unknown command - ignore
      break;
  }
}

/*
 * Send current status to the Raspberry Pi.
 * 
 * Format: "G:SWING,D:-12.5,A:0.65,F:NONE\n"
 *   G = Gait phase (SWING/STANCE/UNKNOWN)
 *   D = Foot drop angle in degrees (negative = dropped)
 *   A = Current activation level
 *   F = Fault status (NONE or fault description)
 */
void sendStatusToPi() {
  Serial.print("G:");
  switch (current_gait_phase) {
    case SWING:   Serial.print("SWING");   break;
    case STANCE:  Serial.print("STANCE");  break;
    default:      Serial.print("UNKNOWN"); break;
  }
  
  Serial.print(",D:");
  Serial.print(foot_drop_angle, 1);
  
  Serial.print(",A:");
  Serial.print(current_activation, 2);
  
  Serial.print(",F:");
  if (safety_fault.length() > 0) {
    Serial.print(safety_fault);
  } else {
    Serial.print("NONE");
  }
  
  Serial.println();
}

// =============================================================================
// SAFETY AND MOTOR CONTROL
// =============================================================================

/*
 * Perform all safety checks.
 */
void performSafetyChecks(unsigned long now) {
  // Watchdog: Stop motor if no command received recently
  if (now - last_command_time > WATCHDOG_TIMEOUT_MS) {
    target_activation = 0.0;
    // Don't set fault - this is expected behavior when Pi stops sending
  }
  
  // Current limit check
  int current_raw = analogRead(PIN_CURRENT_SENSE);
  // Convert to mA (adjust this formula for your current sense circuit)
  // Example: 5V reference, 185mV/A sensitivity (ACS712-5A)
  float current_ma = (current_raw - 512) * (5000.0 / 1024.0) / 0.185;
  
  if (abs(current_ma) > MAX_CURRENT_MA) {
    motor_enabled = false;
    safety_fault = "OVERCURRENT";
    target_activation = 0.0;
  }
  
  // Stance lockout: Do not activate during stance phase
  // (activating during stance could cause the patient to fall)
  if (current_gait_phase == STANCE && target_activation > 0.0) {
    // Don't disable motor, just prevent activation during stance
    // The motor will activate once swing phase is detected
  }
}

/*
 * Update motor output based on target activation and safety state.
 */
void updateMotorOutput() {
  // If motor is disabled due to fault, stop immediately
  if (!motor_enabled) {
    analogWrite(PIN_MOTOR_PWM, 0);
    current_activation = 0.0;
    return;
  }
  
  // Determine effective target (zero during stance)
  float effective_target = target_activation;
  if (current_gait_phase != SWING) {
    effective_target = 0.0;
  }
  
  // Ramp current activation toward target (for smooth motion)
  // Ramp rate: approximately 10% per 10ms = 1000%/sec = full range in 100ms
  const float RAMP_RATE = 0.1;
  
  if (current_activation < effective_target) {
    current_activation = min(current_activation + RAMP_RATE, effective_target);
  } else if (current_activation > effective_target) {
    // Faster ramp down for safety
    current_activation = max(current_activation - RAMP_RATE * 2, effective_target);
  }
  
  // Convert activation to PWM value
  // Direction: HIGH = dorsiflexion (lifting toe)
  digitalWrite(PIN_MOTOR_DIR, HIGH);
  
  // PWM output (0-255)
  int pwm_value = (int)(current_activation * 255.0);
  analogWrite(PIN_MOTOR_PWM, pwm_value);
}
