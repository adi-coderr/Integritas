/*
  MPU6050 -> WiFi -> your dashboard (push every ~200 ms)
  ------------------------------------------------------
  Sends JSON like:
  {"device":"esp32-mpu6050","mpu_ok":true,"ax":0.01,"ay":-0.02,"az":1.00,
   "gx":0.5,"gy":-0.3,"gz":0.1,"temp":30.5,"pitch":0.8,"roll":-1.1,
   "rssi":-55,"uptime_ms":12345}

  MPU6050 pins are auto-detected:
    SDA=GPIO14 SCL=GPIO15 | SDA=GPIO21 SCL=GPIO22 | SDA=GPIO26 SCL=GPIO27

  Serial Monitor: 115200 baud. It shows WiFi status, the ESP32 IP,
  which server ports answer, and the HTTP code of every push.
*/

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>

// ===================== EDIT THESE =====================
const char* WIFI_SSID = "realme X7 5G";        // must be a 2.4 GHz network
const char* WIFI_PASS = "620720113";

const char* pushServerHost = "10.169.71.221";    // from your dashboard banner
const int   pushServerPort = 3000;               // port your dashboard server runs on
const char* pushServerPath = "/api/sensor";      // route that receives the readings
// ======================================================

const unsigned long READ_INTERVAL_MS  = 100;
const unsigned long POST_INTERVAL_MS  = 200;     // push rate when the server answers
const unsigned long RETRY_INTERVAL_MS = 2000;    // wait after a failed push

struct PinPair {
  int sda;
  int scl;
  const char* name;
};

const PinPair PAIRS[] = {
  {14, 15, "SDA=GPIO14, SCL=GPIO15"},
  {21, 22, "SDA=GPIO21, SCL=GPIO22"},
  {26, 27, "SDA=GPIO26, SCL=GPIO27"}
};
const int NUM_PAIRS = sizeof(PAIRS) / sizeof(PAIRS[0]);

uint8_t mpuAddr = 0x68;
bool connected = false;
int failCount = 0;

float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0, tempC = 0, pitchDeg = 0, rollDeg = 0;
bool mpuOk = false;

WiFiClient client;
HTTPClient http;
int lastHttpCode = 0;
bool lastPushOk = false;
unsigned long lastRead = 0, lastPost = 0, lastPrint = 0, lastSearch = 0, lastWifi = 0;
unsigned long postDelay = POST_INTERVAL_MS;

// ---------------- MPU6050 ----------------
static bool probe(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

static bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool readRegs(uint8_t reg, uint8_t* buf, uint8_t len) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  int n = Wire.requestFrom((int)mpuAddr, (int)len, (int)1);
  if (n != len) return false;
  for (uint8_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

static bool findMpu() {
  for (int i = 0; i < NUM_PAIRS; i++) {
    Wire.end();
    Wire.begin(PAIRS[i].sda, PAIRS[i].scl, 100000);
    delay(20);
    if (probe(0x68)) {
      mpuAddr = 0x68;
    } else if (probe(0x69)) {
      mpuAddr = 0x69;
    } else {
      continue;
    }
    Serial.printf("MPU6050 found at 0x%02X on %s\n", mpuAddr, PAIRS[i].name);
    return true;
  }
  Wire.end();
  return false;
}

static bool initMpu() {
  if (!writeReg(0x6B, 0x00)) return false;   // wake up
  delay(100);
  writeReg(0x1A, 0x03);                      // low-pass filter
  writeReg(0x1B, 0x00);                      // gyro  +-250 deg/s
  writeReg(0x1C, 0x00);                      // accel +-2 g
  return true;
}

static bool readMpu() {
  uint8_t b[14];
  if (!readRegs(0x3B, b, 14)) return false;

  bool allSame = true;
  for (int i = 1; i < 14; i++) {
    if (b[i] != b[0]) { allSame = false; break; }
  }
  if (allSame) return false;

  int16_t rax = (int16_t)((b[0]  << 8) | b[1]);
  int16_t ray = (int16_t)((b[2]  << 8) | b[3]);
  int16_t raz = (int16_t)((b[4]  << 8) | b[5]);
  int16_t rt  = (int16_t)((b[6]  << 8) | b[7]);
  int16_t rgx = (int16_t)((b[8]  << 8) | b[9]);
  int16_t rgy = (int16_t)((b[10] << 8) | b[11]);
  int16_t rgz = (int16_t)((b[12] << 8) | b[13]);

  ax = rax / 16384.0f;
  ay = ray / 16384.0f;
  az = raz / 16384.0f;
  gx = rgx / 131.0f;
  gy = rgy / 131.0f;
  gz = rgz / 131.0f;
  tempC = rt / 340.0f + 36.53f;
  pitchDeg = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / (float)M_PI;
  rollDeg  = atan2f(ay, az) * 180.0f / (float)M_PI;
  return true;
}

// ---------------- JSON ----------------
static String buildJson() {
  char buf[400];
  snprintf(buf, sizeof(buf),
           "{\"device\":\"esp32-mpu6050\",\"mpu_ok\":%s,"
           "\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,"
           "\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,"
           "\"temp\":%.2f,\"pitch\":%.2f,\"roll\":%.2f,"
           "\"rssi\":%d,\"uptime_ms\":%lu}",
           mpuOk ? "true" : "false",
           ax, ay, az, gx, gy, gz, tempC, pitchDeg, rollDeg,
           (int)WiFi.RSSI(), (unsigned long)millis());
  return String(buf);
}

// ---------------- Push to dashboard server ----------------
static bool pushReading() {
  if (WiFi.status() != WL_CONNECTED) {
    lastHttpCode = -100;
    return false;
  }

  String url = String("http://") + pushServerHost + ":" + String(pushServerPort) + pushServerPath;
  http.begin(client, url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(buildJson());
  lastHttpCode = code;

  bool ok = (code >= 200 && code < 300);
  if (!ok) {
    if (code < 0) {
      Serial.printf("PUSH FAILED: %s (cannot reach %s:%d - wrong port, server not running, firewall, or different network)\n",
                    http.errorToString(code).c_str(), pushServerHost, pushServerPort);
    } else if (code == 404) {
      Serial.printf("PUSH FAILED: HTTP 404 - server is reachable but has no route %s\n", pushServerPath);
    } else {
      Serial.printf("PUSH FAILED: HTTP %d - server replied: %s\n", code, http.getString().substring(0, 120).c_str());
    }
  } else if (!lastPushOk) {
    Serial.printf("PUSH OK: server accepted the data (HTTP %d). Check your dashboard now.\n", code);
  }
  http.end();
  lastPushOk = ok;
  return ok;
}

// ---------------- Startup helpers ----------------
static void printNearbyNetworks() {
  int n = WiFi.scanNetworks();
  Serial.printf("2.4 GHz networks this ESP32 can see (%d):\n", n);
  for (int i = 0; i < n && i < 15; i++) {
    Serial.printf("  %s  (signal %d dBm)\n", WiFi.SSID(i).c_str(), WiFi.RSSI(i));
  }
  Serial.println("If your WiFi name is not in this list, it is 5 GHz-only, out of range, or spelled differently.");
}

static void checkServerPorts() {
  Serial.printf("Checking the server at %s ...\n", pushServerHost);

  WiFiClient c;
  if (c.connect(pushServerHost, (uint16_t)pushServerPort, 1500)) {
    Serial.printf("  configured port %d: REACHABLE\n", pushServerPort);
    c.stop();
  } else {
    Serial.printf("  configured port %d: NOT reachable\n", pushServerPort);
  }

  const uint16_t ports[] = {3000, 3001, 4000, 5000, 5173, 8000, 8080};
  Serial.println("  other common ports that answer:");
  bool any = false;
  for (int i = 0; i < 7; i++) {
    WiFiClient t;
    if (t.connect(pushServerHost, ports[i], 600)) {
      Serial.printf("    port %u: OPEN\n", ports[i]);
      any = true;
      t.stop();
    }
  }
  if (!any) Serial.println("    none (server not running, firewall blocking, or ESP32 is on a different network)");
}

// ---------------- Setup / loop ----------------
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("=== MPU6050 -> WiFi -> dashboard ===");

  connected = findMpu() && initMpu();
  Serial.println(connected ? "MPU6050 ready."
                           : "MPU6050 NOT FOUND - will keep searching. Check VCC=3.3V, GND, SDA, SCL.");

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Connecting to WiFi \"%s\"", WIFI_SSID);
  unsigned long startMs = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startMs < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("WiFi FAILED (status %d). Check the name/password and that the network is 2.4 GHz.\n", (int)WiFi.status());
    printNearbyNetworks();
    Serial.println("Restarting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  Serial.println("WiFi connected.");
  Serial.println("ESP32 IP : " + WiFi.localIP().toString());
  Serial.println("Server   : " + String(pushServerHost) + ":" + String(pushServerPort) + String(pushServerPath));
  Serial.println("(The ESP32 IP and the server host should normally share the same first three numbers.)");

  checkServerPorts();

  http.setReuse(true);
  http.setConnectTimeout(1500);
  http.setTimeout(2000);
}

void loop() {
  unsigned long now = millis();

  if (now - lastRead >= READ_INTERVAL_MS) {
    lastRead = now;
    if (connected) {
      if (readMpu()) {
        mpuOk = true;
        failCount = 0;
      } else {
        failCount++;
        if (failCount >= 3) {
          connected = false;
          mpuOk = false;
          Serial.println("Lost MPU6050. Searching again...");
        }
      }
    } else if (now - lastSearch >= 3000) {
      lastSearch = now;
      if (findMpu() && initMpu()) {
        connected = true;
        failCount = 0;
      } else {
        Serial.println("MPU6050 NOT FOUND - check VCC=3.3V, GND, SDA, SCL wires.");
      }
    }
  }

  if (now - lastPost >= postDelay) {
    lastPost = now;
    bool ok = pushReading();
    postDelay = ok ? POST_INTERVAL_MS : RETRY_INTERVAL_MS;
  }

  if (now - lastPrint >= 1000) {
    lastPrint = now;
    Serial.printf("mpu_ok=%d  ax=%.2f ay=%.2f az=%.2f  gx=%.1f gy=%.1f gz=%.1f  temp=%.1fC  | last push HTTP=%d\n",
                  mpuOk, ax, ay, az, gx, gy, gz, tempC, lastHttpCode);
  }

  if (WiFi.status() != WL_CONNECTED && now - lastWifi >= 10000) {
    lastWifi = now;
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }

  delay(1);
}
