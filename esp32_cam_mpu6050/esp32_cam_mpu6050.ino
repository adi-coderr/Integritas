/*
 * =========================================================================
 * ESP32-CAM (AI-Thinker) + MPU-6050 Sensor Integration
 * =========================================================================
 * Combines OV2640 camera video streaming with 6-Axis IMU (accelerometer &
 * gyroscope) telemetry for structural crack inspection & orientation tracking.
 *
 * HARDWARE CONNECTIONS:
 * ---------------------
 * MPU-6050 Pin   | ESP32-CAM Pin      | Notes
 * -------------------------------------------------------------------------
 * VCC            | 3.3V (or 5V)       | Power from ESP32-CAM 3.3V pin
 * GND            | GND                | Ground
 * SDA            | GPIO 13            | I2C Data (Pins 21 & 22 are used by
 * camera!) SCL            | GPIO 14            | I2C Clock (Pins 21 & 22 are
 * used by camera!) AD0            | GND (or floating)  | Sets I2C address to
 * 0x68
 *
 * PROGRAMMING / UPLOAD WIRING (via FTDI USB-to-TTL Adapter):
 * -----------------------------------------------------------
 * FTDI 5V/VCC    --> ESP32-CAM 5V
 * FTDI GND       --> ESP32-CAM GND
 * FTDI TX        --> ESP32-CAM U0RX (GPIO 3)
 * FTDI RX        --> ESP32-CAM U0TX (GPIO 1)
 * ESP32-CAM IO0  --> ESP32-CAM GND (CONNECT TO UPLOAD, DISCONNECT TO RUN!)
 *
 * ARDUINO IDE BOARD SETTINGS:
 * ---------------------------
 * Board: "AI Thinker ESP32-CAM" (under Tools > Board > esp32)
 * CPU Frequency: 240MHz (WiFi/BT)
 * Flash Frequency: 80MHz
 * Flash Mode: QIO
 * Partition Scheme: Huge APP (3MB No OTA / 1MB SPIFFS)
 * Upload Speed: 115200 or 921600
 * =========================================================================
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <Wire.h>

// =========================================================================
// 1. WI-FI CONFIGURATION (Optional: Leave blank to run in pure USB-Serial mode)
// =========================================================================
const char *ssid = "realme X7 5G";  // Replace with your WiFi SSID
const char *password = "620720113"; // Replace with your WiFi Password

// =========================================================================
// 1B. DIRECT TELEMETRY PUSH (Push data directly to webpage - NO ESP32 IP NEEDED!)
// =========================================================================
// If pushServerHost is set, ESP32 pushes telemetry directly to your laptop's webpage!
// You DO NOT need to know or look up what IP your router gave the ESP32!
const char *pushServerHost = "10.169.71.17"; // Replace with your laptop's Wi-Fi IP (shown on webpage)
const int   pushServerPort = 5050;             // Port of web_app.py
const bool  enableDirectPush = true;           // Set to true to push data to webpage

// =========================================================================
// 2. MPU-6050 I2C PINS (GPIO 13 & 14 avoid conflicts with OV2640 camera)
// =========================================================================
#define I2C_SDA 13
#define I2C_SCL 14

int mpuAddress = 0x68;
int16_t rawAccX, rawAccY, rawAccZ;
int16_t rawTemp;
int16_t rawGyroX, rawGyroY, rawGyroZ;

float offsetAccX = 0, offsetAccY = 0, offsetAccZ = 0;
float offsetGyroX = 0, offsetGyroY = 0, offsetGyroZ = 0;

float pitch = 0.0;
float roll = 0.0;
float curAx = 0.0, curAy = 0.0, curAz = 1.0;
float curGx = 0.0, curGy = 0.0, curGz = 0.0;
float curTemp = 25.0;
bool shockFlag = false;

unsigned long prevMpuTime = 0;
const float ACC_SCALE = 16384.0; // ±2g
const float GYRO_SCALE = 131.0;  // ±250 deg/s

// =========================================================================
// 3. AI-THINKER ESP32-CAM PIN DEFINITIONS
// =========================================================================
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#define FLASH_LED_PIN 4

httpd_handle_t camera_httpd = NULL;

// Forward declarations
bool initCamera();
bool initMPU();
void calibrateMPU();
void readMPU();
void updateIMU();
void startCameraServer();
void pushTelemetryToWeb(const char *jsonStr);

// =========================================================================
// HTTP SERVER HANDLERS (MJPEG Stream, Snapshot, Telemetry)
// =========================================================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition",
                     "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t telemetry_handler(httpd_req_t *req) {
  char jsonBuf[256];
  snprintf(jsonBuf, sizeof(jsonBuf),
           "{\"pitch\":%.2f,\"roll\":%.2f,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,"
           "\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,\"temp\":%.1f,\"shock\":%s}",
           pitch, roll, curAx, curAy, curAz, curGx, curGy, curGz, curTemp,
           shockFlag ? "true" : "false");
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, jsonBuf, strlen(jsonBuf));
}

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK)
    return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY,
                                  strlen(_STREAM_BOUNDARY));
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (res != ESP_OK) {
      break;
    }
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t capture_uri = {.uri = "/capture",
                             .method = HTTP_GET,
                             .handler = capture_handler,
                             .user_ctx = NULL};

  httpd_uri_t stream_uri = {.uri = "/stream",
                            .method = HTTP_GET,
                            .handler = stream_handler,
                            .user_ctx = NULL};

  httpd_uri_t telem_uri = {.uri = "/telemetry",
                           .method = HTTP_GET,
                           .handler = telemetry_handler,
                           .user_ctx = NULL};

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    httpd_register_uri_handler(camera_httpd, &telem_uri);
    Serial.println(F("[OK] Camera & Telemetry HTTP Server active on Port 80"));
  }
}

// =========================================================================
// SETUP & LOOP
// =========================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println(F("\n=============================================="));
  Serial.println(F("    ESP32-CAM (AI-Thinker) + MPU-6050         "));
  Serial.println(F("=============================================="));

  // 1. Initialize OV2640 Camera
  if (!initCamera()) {
    Serial.println(F("[ERROR] Camera initialization failed!"));
  } else {
    Serial.println(F("[OK] OV2640 Camera initialized successfully."));
  }

  // 2. Initialize I2C for MPU-6050 (SDA=GPIO13, SCL=GPIO14)
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000); // 100kHz standard mode

  if (!initMPU()) {
    Serial.println(F("[WARN] MPU-6050 not detected at 0x68 or 0x69."));
    Serial.println(F("Check wiring: SDA -> GPIO 13, SCL -> GPIO 14, VCC -> "
                     "3.3V, GND -> GND"));
  } else {
    Serial.println(F("[OK] MPU-6050 detected. Calibrating... Keep still!"));
    calibrateMPU();
    Serial.println(F("[OK] MPU-6050 calibration complete."));
  }

  // 3. Connect to Wi-Fi if credentials are provided
  if (strlen(ssid) > 0 && strcmp(ssid, "YOUR_WIFI_SSID") != 0) {
    Serial.print(F("[WiFi] Connecting to "));
    Serial.println(ssid);
    WiFi.begin(ssid, password);

    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 8000) {
      delay(400);
      Serial.print(F("."));
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
      Serial.print(F("[WiFi] Connected! IP Address: http://"));
      Serial.println(WiFi.localIP());
      Serial.println(F("Endpoints:"));
      Serial.print(F("  - Live Video Stream : http://"));
      Serial.print(WiFi.localIP());
      Serial.println(F("/stream"));
      Serial.print(F("  - Still Photo Frame : http://"));
      Serial.print(WiFi.localIP());
      Serial.println(F("/capture"));
      Serial.print(F("  - IMU Telemetry JSON: http://"));
      Serial.print(WiFi.localIP());
      Serial.println(F("/telemetry"));

      startCameraServer();

      // Direct Push Announcement (NO ESP32 IP NEEDED!)
      if (enableDirectPush && strlen(pushServerHost) > 0) {
        Serial.println();
        Serial.println(F(">>> [DIRECT PUSH ACTIVE] ------------------------------"));
        Serial.print(F("Streaming telemetry directly to laptop: http://"));
        Serial.print(pushServerHost);
        Serial.print(F(":"));
        Serial.print(pushServerPort);
        Serial.println(F("/api/mpu/feed"));
        Serial.println(F("You do NOT need to type or know the ESP32 IP on the webpage!"));
        Serial.println(F("-------------------------------------------------------"));
      }
    } else {
      Serial.println(
          F("[WiFi] Connection timed out. Running in USB-Serial Mode."));
    }
  } else {
    Serial.println(
        F("[INFO] No Wi-Fi credentials set. Running in pure USB-Serial Mode."));
  }

  prevMpuTime = micros();
}

void loop() {
  updateIMU();

  // 1. Build structured JSON telemetry payload
  char jsonBuf[220];
  snprintf(jsonBuf, sizeof(jsonBuf),
           "{\"pitch\":%.2f,\"roll\":%.2f,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,"
           "\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,\"temp\":%.1f,\"shock\":%s}",
           pitch, roll, curAx, curAy, curAz, curGx, curGy, curGz, curTemp,
           shockFlag ? "true" : "false");

  // 2. Stream structured JSON telemetry over Serial (115200 baud)
  Serial.println(jsonBuf);

  // 3. Direct Wi-Fi Push to Laptop Webpage (NO ESP32 IP NEEDED!)
  if (enableDirectPush && WiFi.status() == WL_CONNECTED && strlen(pushServerHost) > 0) {
    static unsigned long lastPushTime = 0;
    if (millis() - lastPushTime >= 50) { // 20 Hz push rate
      lastPushTime = millis();
      pushTelemetryToWeb(jsonBuf);
    }
  }

  delay(50); // 20 updates per second
}

// Push telemetry JSON directly to Laptop Web Server over Wi-Fi
void pushTelemetryToWeb(const char *jsonStr) {
  WiFiClient client;
  client.setTimeout(250);
  if (!client.connect(pushServerHost, pushServerPort)) {
    return; // Don't block if server is unreachable
  }
  client.setNoDelay(true);

  client.print(F("POST /api/mpu/feed HTTP/1.1\r\nHost: "));
  client.print(pushServerHost);
  client.print(F("\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "));
  client.print(strlen(jsonStr));
  client.print(F("\r\n\r\n"));
  client.print(jsonStr);

  // Fast drain
  unsigned long t0 = millis();
  while (client.connected() && millis() - t0 < 30) {
    if (client.available()) break;
    delay(1);
  }
  client.stop();
}

// =========================================================================
// CAMERA INITIALIZATION
// =========================================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Optimize quality and buffer based on PSRAM
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 12;          // 10-63 (lower = higher quality)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 16;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  return (err == ESP_OK);
}

// =========================================================================
// MPU-6050 FUNCTIONS
// =========================================================================
bool initMPU() {
  Wire.beginTransmission(0x68);
  if (Wire.endTransmission() == 0) {
    mpuAddress = 0x68;
  } else {
    Wire.beginTransmission(0x69);
    if (Wire.endTransmission() == 0) {
      mpuAddress = 0x69;
    } else {
      return false;
    }
  }

  // Wake up MPU-6050 (write 0 to PWR_MGMT_1)
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x6B);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0)
    return false;

  // Accel ±2g
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission();

  // Gyro ±250 deg/s
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission();

  return true;
}

void readMPU() {
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(mpuAddress, 14, true);

  if (Wire.available() >= 14) {
    rawAccX = (Wire.read() << 8) | Wire.read();
    rawAccY = (Wire.read() << 8) | Wire.read();
    rawAccZ = (Wire.read() << 8) | Wire.read();
    rawTemp = (Wire.read() << 8) | Wire.read();
    rawGyroX = (Wire.read() << 8) | Wire.read();
    rawGyroY = (Wire.read() << 8) | Wire.read();
    rawGyroZ = (Wire.read() << 8) | Wire.read();
  }
}

void calibrateMPU() {
  const int samples = 150;
  long sumAx = 0, sumAy = 0, sumAz = 0;
  long sumGx = 0, sumGy = 0, sumGz = 0;

  for (int i = 0; i < samples; i++) {
    readMPU();
    sumAx += rawAccX;
    sumAy += rawAccY;
    sumAz += (rawAccZ - (int16_t)ACC_SCALE);
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

  readMPU();
  float ax = (rawAccX - offsetAccX) / ACC_SCALE;
  float ay = (rawAccY - offsetAccY) / ACC_SCALE;
  float az = (rawAccZ - offsetAccZ) / ACC_SCALE;
  pitch = atan2(ay, sqrt(ax * ax + az * az)) * 180.0 / PI;
  roll = atan2(-ax, az) * 180.0 / PI;
}

void updateIMU() {
  readMPU();

  unsigned long currentTime = micros();
  float dt = (currentTime - prevMpuTime) / 1000000.0;
  prevMpuTime = currentTime;

  curAx = (rawAccX - offsetAccX) / ACC_SCALE;
  curAy = (rawAccY - offsetAccY) / ACC_SCALE;
  curAz = (rawAccZ - offsetAccZ) / ACC_SCALE;

  curGx = (rawGyroX - offsetGyroX) / GYRO_SCALE;
  curGy = (rawGyroY - offsetGyroY) / GYRO_SCALE;
  curGz = (rawGyroZ - offsetGyroZ) / GYRO_SCALE;

  curTemp = (rawTemp / 340.0) + 36.53;

  float accPitch =
      atan2(curAy, sqrt(curAx * curAx + curAz * curAz)) * 180.0 / PI;
  float accRoll = atan2(-curAx, curAz) * 180.0 / PI;

  // Complementary filter: 96% Gyro + 4% Accel
  pitch = 0.96 * (pitch + curGx * dt) + 0.04 * accPitch;
  roll = 0.96 * (roll + curGy * dt) + 0.04 * accRoll;

  float totalG = sqrt(curAx * curAx + curAy * curAy + curAz * curAz);
  shockFlag = (abs(totalG - 1.0) > 0.45);
}
