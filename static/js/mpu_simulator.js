/**
 * MPU-6050 Demo & Judge Presentation Simulator (Item D1)
 * -----------------------------------------------------
 * Scripted client-side physics & alert scenario generator.
 * Strictly decoupled from real backend ingestion.
 * Clearly watermarks all generated data as SIMULATED.
 */

(function (window) {
  'use strict';

  class MpuSimulator {
    constructor(onReadingCallback) {
      this.onReading = onReadingCallback;
      this.active = false;
      this.timer = null;
      this.scenario = 'normal';
      this.tick = 0;

      // Internal simulation state
      this.state = {
        pitch: 0.0,
        roll: 0.0,
        ax: 0.0,
        ay: 0.0,
        az: 1.0,
        gx: 0.0,
        gy: 0.0,
        gz: 0.0,
        temp: 28.5,
        mpu_ok: true
      };
    }

    start(scenario = 'normal') {
      this.active = true;
      this.scenario = scenario;
      this.tick = 0;
      if (this.timer) clearInterval(this.timer);
      // Generate simulated readings at 10 Hz (100ms)
      this.timer = setInterval(() => this.step(), 100);
    }

    stop() {
      this.active = false;
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
    }

    setScenario(scenario) {
      this.scenario = scenario;
      this.tick = 0;
    }

    step() {
      if (!this.active) return;
      this.tick++;

      // If scenario is 'offline', stop emitting any samples so watchdog triggers (Item D1)
      if (this.scenario === 'offline') {
        return;
      }

      const t = this.tick * 0.1;
      let pitch = 0, roll = 0, ax = 0, ay = 0, az = 1.0, gx = 0, gy = 0, gz = 0, temp = 28.5, mpu_ok = true;

      switch (this.scenario) {
        case 'normal':
          // Subtle structural micro-vibrations
          pitch = Math.sin(t * 0.5) * 0.8;
          roll = Math.cos(t * 0.4) * 0.5;
          ax = -Math.sin(pitch * Math.PI / 180.0) + (Math.random() - 0.5) * 0.02;
          ay = Math.sin(roll * Math.PI / 180.0) + (Math.random() - 0.5) * 0.02;
          az = 1.00 + (Math.random() - 0.5) * 0.02;
          gx = (Math.random() - 0.5) * 2.0;
          gy = (Math.random() - 0.5) * 2.0;
          gz = (Math.random() - 0.5) * 1.5;
          temp = 28.5 + Math.sin(t * 0.1) * 0.5;
          break;

        case 'tilt':
          // Progressively tilts past warning (15°) and critical (30°)
          const targetTilt = 18.0 + Math.sin(t * 0.8) * 16.0; // oscillates between 2° and 34°
          pitch = targetTilt;
          roll = targetTilt * 0.4;
          ax = -Math.sin(pitch * Math.PI / 180.0);
          ay = Math.sin(roll * Math.PI / 180.0);
          az = Math.cos(pitch * Math.PI / 180.0);
          gx = Math.cos(t * 0.8) * 12.0;
          gy = Math.sin(t * 0.8) * 8.0;
          temp = 32.0;
          break;

        case 'shock':
          // Periodic high-energy impact impulse every 3 seconds
          const isPulse = (this.tick % 30 < 4);
          if (isPulse) {
            ax = (Math.random() > 0.5 ? 1 : -1) * (1.2 + Math.random() * 0.8);
            ay = (Math.random() - 0.5) * 1.4;
            az = 1.0 + (Math.random() - 0.5) * 1.8;
            gx = (Math.random() - 0.5) * 120.0;
            gy = (Math.random() - 0.5) * 140.0;
            pitch = (Math.random() - 0.5) * 20.0;
            roll = (Math.random() - 0.5) * 18.0;
          } else {
            pitch = 0.5; roll = 0.2; ax = 0.02; ay = 0.01; az = 1.01;
            gx = 1.0; gy = 0.5; gz = 0.2;
          }
          temp = 31.0;
          break;

        case 'vibration':
          // Rapid harmonic structural resonance (2s RMS dynamic G > 0.18G)
          const vibAmp = 0.35 * Math.sin(t * 15.0);
          ax = vibAmp;
          ay = vibAmp * 0.7;
          az = 1.0 + vibAmp * 0.9;
          gx = Math.sin(t * 12.0) * 35.0;
          gy = Math.cos(t * 12.0) * 35.0;
          pitch = Math.sin(t * 15.0) * 4.0;
          roll = Math.cos(t * 15.0) * 3.0;
          temp = 34.2;
          break;

        case 'freefall':
          // Gravity collapses below 0.3G for 6 samples every 35 ticks
          const inFreeFall = (this.tick % 35 < 6);
          if (inFreeFall) {
            ax = (Math.random() - 0.5) * 0.08;
            ay = (Math.random() - 0.5) * 0.08;
            az = 0.06 + Math.random() * 0.08; // total G < 0.20 G
            gx = (Math.random() - 0.5) * 15.0;
            gy = (Math.random() - 0.5) * 15.0;
          } else {
            ax = 0.01; ay = 0.01; az = 1.0;
            gx = 0.5; gy = 0.5; gz = 0.1;
          }
          temp = 29.0;
          break;

        case 'rotation':
          // Rapid centrifugal rotation exceeding 150 deg/s
          gx = 165.0 * Math.sin(t * 2.0);
          gy = 95.0 * Math.cos(t * 1.5);
          gz = 180.0 * Math.sin(t * 2.5);
          pitch = Math.sin(t * 2.0) * 25.0;
          roll = Math.cos(t * 2.0) * 25.0;
          ax = 0.3; ay = 0.2; az = 0.95;
          temp = 36.0;
          break;

        case 'overheat':
          // Die thermal runaway ramping past 55°C to 74°C
          temp = Math.min(76.0, 50.0 + (this.tick * 0.15));
          pitch = 1.0; roll = 0.5; ax = 0.02; ay = 0.01; az = 1.0;
          break;

        case 'fault':
          // Hardware freeze or disconnected I2C bus
          mpu_ok = false;
          ax = 0.0; ay = 0.0; az = 0.0;
          gx = 0.0; gy = 0.0; gz = 0.0;
          temp = 0.0;
          break;
      }

      const totalG = Math.sqrt(ax * ax + ay * ay + az * az);

      const simReading = {
        device: 'esp32-simulated',
        simulated: true,
        source: 'SIMULATOR',
        mpu_ok: mpu_ok,
        ax: Number(ax.toFixed(3)),
        ay: Number(ay.toFixed(3)),
        az: Number(az.toFixed(3)),
        gx: Number(gx.toFixed(2)),
        gy: Number(gy.toFixed(2)),
        gz: Number(gz.toFixed(2)),
        pitch: Number(pitch.toFixed(2)),
        roll: Number(roll.toFixed(2)),
        total_g: Number(totalG.toFixed(3)),
        temp: Number(temp.toFixed(1)),
        rssi: -45,
        uptime_ms: this.tick * 100,
        timestamp: Date.now() / 1000
      };

      if (typeof this.onReading === 'function') {
        this.onReading(simReading);
      }
    }
  }

  window.MpuSimulator = MpuSimulator;
})(window);
