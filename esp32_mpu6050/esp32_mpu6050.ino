/*
 * ESP32 + MPU-6050 (6-Axis Accelerometer & Gyroscope)
 * ----------------------------------------------------
 * Built using native ESP32 Wire.h (NO external libraries required!)
 * 
 * Hardware Connections:
 *   MPU-6050 VCC  --> ESP32 3.3V (or 5V if board has 3.3V LDO)
 *   MPU-6050 GND  --> ESP32 GND
 *   MPU-6050 SCL  --> ESP32 GPIO 22 (Default I2C Clock)
 *   MPU-6050 SDA  --> ESP32 GPIO 21 (Default I2C Data)
 *   MPU-6050 AD0  --> GND (sets I2C address to 0x68)
 * 
 * Features:
 *   - Auto-detects MPU6050 at 0x68 or 0x69
 *   - Automatic startup calibration (gyro & accel zeroing)
 *   - Real-time Pitch & Roll tilt angles via Complementary Filter
 *   - Total G-force and vibration magnitude calculation
 *   - High-speed 115200 baud serial telemetry (JSON & human-readable)
 */

#include <Wire.h>
#include <WiFi.h>

// =========================================================================
// Wi-Fi & Direct Web Dashboard Push (Optional: Leave blank for pure USB Serial)
// If pushServerHost is set, ESP32 pushes telemetry directly to your webpage!
// You DO NOT need to know or look up what IP your router gave the ESP32!
// =========================================================================
const char *ssid             = "realme X7 5G";  // Replace with your WiFi SSID
const char *password         = "620720113";     // Replace with your WiFi Password
const char *pushServerHost   = "10.169.71.221"; // Replace with your laptop's Wi-Fi IP (shown on webpage)
const int   pushServerPort   = 5050;             // Port of web_app.py
const bool  enableDirectPush = true;            // Push data directly to webpage (NO ESP32 IP NEEDED!)

// I2C Address (0x68 is default when AD0 is GND or floating)
int mpuAddress = 0x68;

// Raw 16-bit register values
int16_t rawAccX, rawAccY, rawAccZ;
int16_t rawTemp;
int16_t rawGyroX, rawGyroY, rawGyroZ;

// Calibration offsets
float offsetAccX = 0, offsetAccY = 0, offsetAccZ = 0;
float offsetGyroX = 0, offsetGyroY = 0, offsetGyroZ = 0;

// Filtered Orientation angles (degrees)
float pitch = 0.0;
float roll  = 0.0;

// Timing
unsigned long prevTime = 0;

// Sensitivity scales (Default: Accel ±2g -> 16384 LSB/g, Gyro ±250 deg/s -> 131.0 LSB/deg/s)
const float ACC_SCALE  = 16384.0;
const float GYRO_SCALE = 131.0;

// Function prototypes
bool initMPU();
void calibrateMPU();
void readMPU();

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Wait for serial console

  Serial.println();
  Serial.println(F("========================================"));
  Serial.println(F("    ESP32 + MPU-6050 IMU Telemetry      "));
  Serial.println(F("========================================"));

  // Initialize I2C on ESP32 default pins: SDA=21, SCL=22
  Wire.begin(21, 22);
  Wire.setClock(400000); // 400kHz Fast I2C

  // Initialize and wake up MPU-6050
  if (!initMPU()) {
    Serial.println(F("[ERROR] MPU-6050 not detected!"));
    Serial.println(F("Please check wiring: SDA -> GPIO 21, SCL -> GPIO 22, VCC -> 3.3V, GND -> GND"));
    while (true) {
      delay(1000);
    }
  }

  Serial.println(F("[OK] MPU-6050 initialized successfully."));
  Serial.println(F("[INFO] Calibrating sensor... Keep ESP32 flat and still!"));
  
  calibrateMPU();
  
  Serial.println(F("[OK] Calibration complete. Streaming data...\n"));

  // Initialize Wi-Fi for Direct Web Push if configured
  if (strlen(ssid) > 0 && strcmp(ssid, "YOUR_WIFI_SSID") != 0) {
    Serial.print(F("[WiFi] Connecting to "));
    Serial.println(ssid);
    WiFi.begin(ssid, password);
    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 7000) {
      delay(350);
      Serial.print(F("."));
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
      Serial.print(F("[WiFi] Connected! IP: "));
      Serial.println(WiFi.localIP());
      if (enableDirectPush && strlen(pushServerHost) > 0) {
        Serial.print(F("[PUSH] Telemetry streaming directly to: http://"));
        Serial.print(pushServerHost);
        Serial.print(F(":"));
        Serial.print(pushServerPort);
        Serial.println(F("/api/mpu/feed"));
        Serial.println(F("[PUSH] -> No ESP32 IP needed on the dashboard!"));
      }
    } else {
      Serial.println(F("[WiFi] Not connected. Running in pure USB-Serial mode."));
    }
  }

  prevTime = micros();
}

// Push telemetry JSON directly to Laptop Web Server over Wi-Fi
void pushTelemetryToWeb(const char *jsonStr) {
  WiFiClient client;
  client.setTimeout(250);
  if (!client.connect(pushServerHost, pushServerPort)) {
    return;
  }
  client.setNoDelay(true);

  client.print(F("POST /api/mpu/feed HTTP/1.1\r\nHost: "));
  client.print(pushServerHost);
  client.print(F("\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "));
  client.print(strlen(jsonStr));
  client.print(F("\r\n\r\n"));
  client.print(jsonStr);

  unsigned long t0 = millis();
  while (client.connected() && millis() - t0 < 30) {
    if (client.available()) break;
    delay(1);
  }
  client.stop();
}

void loop() {
  // Read sensor registers
  readMPU();

  // Calculate elapsed time (dt in seconds)
  unsigned long currentTime = micros();
  float dt = (currentTime - prevTime) / 1000000.0;
  prevTime = currentTime;

  // Convert raw readings to physical units
  float ax = (rawAccX - offsetAccX) / ACC_SCALE;
  float ay = (rawAccY - offsetAccY) / ACC_SCALE;
  float az = (rawAccZ - offsetAccZ) / ACC_SCALE;

  float gx = (rawGyroX - offsetGyroX) / GYRO_SCALE;
  float gy = (rawGyroY - offsetGyroY) / GYRO_SCALE;
  float gz = (rawGyroZ - offsetGyroZ) / GYRO_SCALE;

  // Temperature in Celsius (MPU-6050 formula)
  float tempC = (rawTemp / 340.0) + 36.53;

  // Compute Accelerometer Angles (degrees)
  float accPitch = atan2(ay, sqrt(ax * ax + az * az)) * 180.0 / PI;
  float accRoll  = atan2(-ax, az) * 180.0 / PI;

  // Complementary Filter: 96% Gyro (smooth, fast) + 4% Accel (eliminates long-term drift)
  pitch = 0.96 * (pitch + gx * dt) + 0.04 * accPitch;
  roll  = 0.96 * (roll + gy * dt) + 0.04 * accRoll;

  // Total dynamic acceleration magnitude (1.0 = resting 1G gravity)
  float totalG = sqrt(ax * ax + ay * ay + az * az);

  // Check vibration / shock threshold
  bool shockDetected = (abs(totalG - 1.0) > 0.45);

  // Build JSON telemetry string
  char jsonBuf[220];
  snprintf(jsonBuf, sizeof(jsonBuf),
           "{\"pitch\":%.2f,\"roll\":%.2f,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,"
           "\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,\"temp\":%.1f,\"shock\":%s}",
           pitch, roll, ax, ay, az, gx, gy, gz, tempC,
           shockDetected ? "true" : "false");

  // Output 1: Serial (115200 baud)
  Serial.println(jsonBuf);

  // Output 2: Direct Wi-Fi Push to Laptop Web Server (NO ESP32 IP NEEDED!)
  if (enableDirectPush && WiFi.status() == WL_CONNECTED && strlen(pushServerHost) > 0) {
    static unsigned long lastPushTime = 0;
    if (millis() - lastPushTime >= 50) { // ~20 Hz
      lastPushTime = millis();
      pushTelemetryToWeb(jsonBuf);
    }
  }

  delay(50); // ~20 Hz update rate
}

/**
 * Initialize and verify MPU-6050 connection
 */
bool initMPU() {
  // Check default address 0x68
  Wire.beginTransmission(0x68);
  if (Wire.endTransmission() == 0) {
    mpuAddress = 0x68;
  } else {
    // Check secondary address 0x69 (if AD0 is pulled HIGH)
    Wire.beginTransmission(0x69);
    if (Wire.endTransmission() == 0) {
      mpuAddress = 0x69;
    } else {
      return false;
    }
  }

  // Wake up MPU-6050: Write 0 to PWR_MGMT_1 register (0x6B)
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x6B);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) return false;

  // Set Accelerometer range to ±2g (Register 0x1C)
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission();

  // Set Gyroscope range to ±250 deg/s (Register 0x1B)
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission();

  return true;
}

/**
 * Read 14 consecutive registers starting at 0x3B:
 * Accel(X,Y,Z) [6 bytes] + Temp [2 bytes] + Gyro(X,Y,Z) [6 bytes]
 */
void readMPU() {
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x3B); // Register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(mpuAddress, 14, true);

  if (Wire.available() >= 14) {
    rawAccX  = (Wire.read() << 8) | Wire.read();
    rawAccY  = (Wire.read() << 8) | Wire.read();
    rawAccZ  = (Wire.read() << 8) | Wire.read();
    rawTemp  = (Wire.read() << 8) | Wire.read();
    rawGyroX = (Wire.read() << 8) | Wire.read();
    rawGyroY = (Wire.read() << 8) | Wire.read();
    rawGyroZ = (Wire.read() << 8) | Wire.read();
  }
}

/**
 * Average 300 samples to find zero-rate offsets
 */
void calibrateMPU() {
  const int samples = 300;
  long sumAx = 0, sumAy = 0, sumAz = 0;
  long sumGx = 0, sumGy = 0, sumGz = 0;

  for (int i = 0; i < samples; i++) {
    readMPU();
    sumAx += rawAccX;
    sumAy += rawAccY;
    sumAz += (rawAccZ - (int16_t)ACC_SCALE); // Leave 1G (16384) on Z axis for gravity
    sumGx += rawGyroX;
    sumGy += rawGyroY;
    sumGz += rawGyroZ;
    delay(5);
  }

  offsetAccX = (float)sumAx / samples;
  offsetAccY = (float)sumAy / samples;
  offsetAccZ = (float)sumAz / samples;

  offsetGyroX = (float)sumGx / samples;
  offsetGyroY = (float)sumGy / samples;
  offsetGyroZ = (float)sumGz / samples;

  // Initialize initial pitch & roll from resting accelerometer
  readMPU();
  float ax = (rawAccX - offsetAccX) / ACC_SCALE;
  float ay = (rawAccY - offsetAccY) / ACC_SCALE;
  float az = (rawAccZ - offsetAccZ) / ACC_SCALE;
  pitch = atan2(ay, sqrt(ax * ax + az * az)) * 180.0 / PI;
  roll  = atan2(-ax, az) * 180.0 / PI;
}
