# ESP32-CAM + MPU-6050 Integration Guide

## 1. Hardware Pin Connections

> [!IMPORTANT]
> On a standard ESP32, I2C uses GPIO 21 and 22. **Do NOT use GPIO 21 or 22 on an ESP32-CAM**, because the camera sensor uses GPIO 21 (D3) and GPIO 22 (PCLK)!
> This sketch uses **GPIO 13** (SDA) and **GPIO 14** (SCL).

| MPU-6050 Pin | ESP32-CAM Pin | Notes |
| :--- | :--- | :--- |
| **VCC** | **3.3V** (or 5V) | Power supply |
| **GND** | **GND** | Ground |
| **SDA** | **GPIO 13** | Software/Hardware I2C Data |
| **SCL** | **GPIO 14** | Software/Hardware I2C Clock |
| **AD0** | **GND** | Default I2C address `0x68` |

---

## 2. FTDI USB-to-TTL Adapter Wiring (For Uploading)

ESP32-CAM does not have a built-in USB port, so use an FTDI USB-to-Serial adapter:

| FTDI Adapter Pin | ESP32-CAM Pin | Purpose |
| :--- | :--- | :--- |
| **VCC (5V)** | **5V** | Power |
| **GND** | **GND** | Ground |
| **TX** | **U0RX (GPIO 3)** | Serial Receive |
| **RX** | **U0TX (GPIO 1)** | Serial Transmit |
| **GND** | **GPIO 0 (IO0)** | **MUST BE CONNECTED TO GND DURING UPLOAD!** |

> [!WARNING]
> - **During Upload:** Connect **IO0 to GND**, press the **RST** button on the ESP32-CAM, then click **Upload**.
> - **After Upload:** **Disconnect IO0 from GND**, then press the **RST** button once to run your code!

---

## 3. Arduino IDE Board Settings

In Arduino IDE:
1. Go to **Tools > Board > esp32** and select:
   - **`AI Thinker ESP32-CAM`**
2. **CPU Frequency:** `240MHz (WiFi/BT)`
3. **Flash Frequency:** `80MHz`
4. **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)`
5. **Port:** Select your FTDI port (e.g., `/dev/cu.usbserial-...`).

---

## 4. What the Code Does
- **OV2640 Camera Video:** Hosts a high-speed MJPEG stream at `http://<IP>/stream` and snapshot captures at `http://<IP>/capture`.
- **MPU-6050 Telemetry:** Streams real-time tilt orientation (Pitch, Roll), acceleration ($a_x, a_y, a_z$), and shock detection over:
  - **Serial Monitor** at `115200` baud.
  - **HTTP JSON Endpoint** at `http://<IP>/telemetry`.
  - **Direct Wi-Fi Push** directly to your laptop dashboard (`http://<Laptop_IP>:5050/api/mpu/feed`).

---

## 5. Sending Data to the Webpage: Do You Need the ESP32 IP?

### **Short Answer:**
**NO, you do NOT need the ESP32's IP address!**

### **How the 3 Modes Work:**

1. **Method 1: Direct Wi-Fi Push (Recommended — NO ESP32 IP NEEDED)**
   - The ESP32 acts as the client and **pushes** its sensor telemetry directly to your laptop's IP address.
   - You only need your laptop's IP address (`10.169.71.17:5050`), which is already configured in the `.ino` sketch:
     ```cpp
     const char *pushServerHost = "10.169.71.17"; // Your laptop's LAN IP
     const int   pushServerPort = 5050;
     const bool  enableDirectPush = true;
     ```
   - As soon as the ESP32 connects to your Wi-Fi hotspot/router, it immediately begins sending HTTP POST packets to `http://10.169.71.17:5050/api/mpu/feed`. The web dashboard receives them automatically without you ever needing to know the ESP32's DHCP IP!

2. **Method 2: USB Cable Mode (NO IP NEEDED AT ALL)**
   - Connect the ESP32 via USB cable / FTDI adapter.
   - On the web dashboard, select **USB Cable**, choose the COM / serial port, and click **Connect**.
   - Data streams directly over 115200 baud serial. 100% offline, zero network configuration required.

3. **Method 3: Pull via ESP32 IP (YES, ESP32 IP is needed)**
   - Only needed if you prefer having your browser fetch data from the ESP32 web server (`http://<ESP32_IP>/telemetry`).
   - You can find the ESP32 IP printed in the Arduino IDE Serial Monitor when the ESP32 powers on and connects to Wi-Fi.

