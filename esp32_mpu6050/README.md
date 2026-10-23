# ESP32 + MPU-6050 Setup Guide

## 1. Pin Wiring Diagram

| MPU-6050 Pin | ESP32 Pin | Description |
| :--- | :--- | :--- |
| **VCC** | **3.3V** (or 5V if breakout has 3.3V regulator) | Power supply |
| **GND** | **GND** | Ground |
| **SCL** | **GPIO 22** | Hardware I2C Clock |
| **SDA** | **GPIO 21** | Hardware I2C Data |
| **AD0** | **GND** | Sets I2C Address to `0x68` (Leave floating or GND) |
| **INT** | *Not connected* (optional) | Hardware interrupt |

---

## 2. How to Upload in Arduino IDE

1. Open Arduino IDE.
2. Select Board: **Tools > Board > ESP32 Arduino > ESP32 Dev Module** (or your specific ESP32 board).
3. Select Port: **Tools > Port > `/dev/cu.usbserial...`** (or `COM...` on Windows).
4. Open [`esp32_mpu6050.ino`](file:///Users/adi/Downloads/techtrixProject/MyModel1/esp32_mpu6050/esp32_mpu6050.ino).
5. Click **Upload** (Arrow button).
6. Open **Tools > Serial Monitor** and set baud rate to **`115200`**.

---

## 3. What the Code Does
- **Zero Library Dependencies**: Uses the native ESP32 `Wire.h` library, avoiding missing library errors.
- **Auto Startup Calibration**: Keeps flat during the first 2 seconds to zero out drift offsets.
- **Complementary Filter**: Fuses Accelerometer and Gyroscope readings for stable, drift-free Pitch and Roll angles.
- **Output**: Streams real-time structured JSON telemetry at 20 Hz:
  ```json
  {"pitch": 1.25, "roll": -0.84, "ax": 0.012, "ay": -0.015, "az": 0.998, "gx": 0.12, "gy": -0.05, "gz": 0.02, "temp": 28.4, "shock": false}
  ```
