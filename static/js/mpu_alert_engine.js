/**
 * MPU-6050 Pure Alert Engine Module
 * ---------------------------------
 * Evaluates raw & calibrated telemetry against configurable threshold rules.
 * Implements anti-false-alarm hysteresis, consecutive sample debounce,
 * 10s alert cooldown, Web Audio alerts, and Notification API.
 */

(function (window) {
  'use strict';

  // Default alert threshold rules (Item B2)
  const DEFAULT_THRESHOLDS = {
    tilt_warning: 15.0,        // degrees
    tilt_critical: 30.0,       // degrees
    shock_warning: 0.50,       // g dynamic accel
    shock_critical: 1.00,      // g dynamic accel
    vibration_warning: 0.05,   // g (2s RMS)
    vibration_critical: 0.15,  // g (2s RMS)
    rotation_warning: 50.0,    // deg/s
    rotation_critical: 150.0,  // deg/s
    freefall_critical: 0.30,   // g total gravity (< 0.3g for >= 2 samples)
    temp_warning: 55.0,        // °C
    temp_critical: 70.0,       // °C
    offline_warning: 3.0,      // seconds without data
    offline_critical: 10.0,    // seconds without data
    webhook_url: ''            // optional external alert delivery (Item B9)
  };

  class MpuAlertEngine {
    constructor() {
      this.thresholds = this.loadThresholds();
      this.alertLog = [];           // Full historical alert log
      this.activeAlerts = new Map(); // key -> active alert state
      this.sampleStreak = new Map(); // key -> { countAbove: 0, countBelow: 0 }
      this.lastNotified = new Map(); // key -> timestamp of last chime/notification (10s cooldown)
      this.consecutiveIdenticalCount = 0;
      this.lastRawSampleStr = '';

      // Audio notification setup
      this.audioCtx = null;
      this.soundEnabled = false;
      this.isMuted = false;

      // Stats
      this.countsToday = { warning: 0, critical: 0 };
      this.lastAlertTime = null;
    }

    loadThresholds() {
      try {
        const saved = localStorage.getItem('mpu_alert_thresholds');
        if (saved) {
          return Object.assign({}, DEFAULT_THRESHOLDS, JSON.parse(saved));
        }
      } catch (e) {
        console.warn('[AlertEngine] Using default thresholds:', e);
      }
      return Object.assign({}, DEFAULT_THRESHOLDS);
    }

    saveThresholds(newThresholds) {
      this.thresholds = Object.assign({}, this.thresholds, newThresholds);
      try {
        localStorage.setItem('mpu_alert_thresholds', JSON.stringify(this.thresholds));
      } catch (e) {
        console.warn('[AlertEngine] Failed to save thresholds:', e);
      }
    }

    resetThresholds() {
      this.thresholds = Object.assign({}, DEFAULT_THRESHOLDS);
      try {
        localStorage.removeItem('mpu_alert_thresholds');
      } catch (e) {}
      return this.thresholds;
    }

    // Initialize Web Audio on user gesture (Item B5)
    initAudio() {
      if (!this.audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
          this.audioCtx = new AudioContext();
        }
      }
      if (this.audioCtx && this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      this.soundEnabled = true;
      return true;
    }

    toggleMute() {
      this.isMuted = !this.isMuted;
      return this.isMuted;
    }

    // Play synthesized alert chimes (no external audio files needed)
    playTone(severity) {
      if (this.isMuted || !this.soundEnabled || !this.audioCtx) return;
      try {
        const now = this.audioCtx.currentTime;
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);

        if (severity === 'critical') {
          // Urgent dual-tone siren
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(880, now);
          osc.frequency.setValueAtTime(660, now + 0.12);
          osc.frequency.setValueAtTime(880, now + 0.24);
          gain.gain.setValueAtTime(0.25, now);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
          osc.start(now);
          osc.stop(now + 0.45);
        } else {
          // Gentle double chime for warning
          osc.type = 'sine';
          osc.frequency.setValueAtTime(587.33, now); // D5
          osc.frequency.setValueAtTime(783.99, now + 0.1); // G5
          gain.gain.setValueAtTime(0.2, now);
          gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
          osc.start(now);
          osc.stop(now + 0.25);
        }
      } catch (e) {
        console.warn('[AlertEngine] Audio playback error:', e);
      }
    }

    // Request Browser Notification permission (Item B8)
    async requestNotificationPermission() {
      if (!('Notification' in window)) return false;
      if (Notification.permission === 'granted') return true;
      const res = await Notification.requestPermission();
      return res === 'granted';
    }

    sendBrowserNotification(title, body) {
      if (!('Notification' in window)) return;
      if (Notification.permission === 'granted' && !this.isMuted) {
        try {
          new Notification(title, {
            body: body,
            icon: '/static/favicon.ico',
            tag: 'mpu-alert'
          });
        } catch (e) {}
      }
    }

    // Optional external webhook delivery (Item B9)
    async dispatchWebhook(alertEvent) {
      const url = this.thresholds.webhook_url;
      if (!url || !url.startsWith('http')) return;
      try {
        fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            event: 'mpu_alert',
            alert: alertEvent,
            timestamp: new Date().toISOString()
          }),
          mode: 'no-cors'
        }).catch(() => {});
      } catch (e) {}
    }

    /**
     * Main evaluation step
     * @param {Object} reading - Sensor readings { ax, ay, az, gx, gy, gz, pitch, roll, total_g, temp, mpu_ok, timestamp }
     * @param {number} rmsVibration - Calculated 2-second dynamic acceleration RMS
     * @param {number} elapsedSinceDataSec - Seconds since last real data arrival
     * @returns {Object} { masterStatus: 'normal'|'warning'|'critical', activeAlerts: Array, newEvents: Array }
     */
    evaluate(reading, rmsVibration, elapsedSinceDataSec) {
      const now = Date.now();
      const newEvents = [];
      const cur = reading || {};

      // 1. Compute dynamic acceleration |total_g - 1.0|
      const ax = typeof cur.ax === 'number' ? cur.ax : 0.0;
      const ay = typeof cur.ay === 'number' ? cur.ay : 0.0;
      const az = typeof cur.az === 'number' ? cur.az : 1.0;
      const totalG = typeof cur.total_g === 'number' ? cur.total_g : Math.sqrt(ax * ax + ay * ay + az * az);
      const dynamicG = Math.abs(totalG - 1.0);

      // 2. Compute tilt
      const pitch = typeof cur.pitch === 'number' ? cur.pitch : 0.0;
      const roll = typeof cur.roll === 'number' ? cur.roll : 0.0;
      const maxTilt = Math.max(Math.abs(pitch), Math.abs(roll));

      // 3. Gyro magnitude
      const gx = typeof cur.gx === 'number' ? cur.gx : 0.0;
      const gy = typeof cur.gy === 'number' ? cur.gy : 0.0;
      const gz = typeof cur.gz === 'number' ? cur.gz : 0.0;
      const maxGyro = Math.max(Math.abs(gx), Math.abs(gy), Math.abs(gz));

      const temp = typeof cur.temp === 'number' ? cur.temp : 25.0;

      // 4. Sensor fault test: identical samples or mpu_ok === false
      const sampleStr = `${ax.toFixed(3)},${ay.toFixed(3)},${az.toFixed(3)},${gx.toFixed(1)},${gy.toFixed(1)},${gz.toFixed(1)}`;
      if (sampleStr === this.lastRawSampleStr && (ax === 0 && ay === 0 && az === 0)) {
        this.consecutiveIdenticalCount++;
      } else {
        this.consecutiveIdenticalCount = 0;
      }
      this.lastRawSampleStr = sampleStr;

      const isFault = (cur.mpu_ok === false) || (this.consecutiveIdenticalCount >= 5);

      // Raw rule definitions: [key, name, isCritical, isWarning, measuredVal, threshCritical, threshWarn, unit, consecutiveNeed]
      const rules = [
        // Free-fall (|a| < 0.3g for >= 2 consecutive samples)
        ['freefall', 'Free-Fall Detected', totalG < this.thresholds.freefall_critical, false, totalG, this.thresholds.freefall_critical, 0, 'G', 2],
        // Sensor Offline / Stale
        ['offline', 'Sensor Offline / Stale', elapsedSinceDataSec >= this.thresholds.offline_critical, elapsedSinceDataSec >= this.thresholds.offline_warning, elapsedSinceDataSec, this.thresholds.offline_critical, this.thresholds.offline_warning, 's', 1],
        // Sensor Fault
        ['fault', 'Sensor Hardware Fault', isFault, false, 0, 1, 0, '', 1],
        // Tilt
        ['tilt', 'Excessive Tilt', maxTilt >= this.thresholds.tilt_critical, maxTilt >= this.thresholds.tilt_warning, maxTilt, this.thresholds.tilt_critical, this.thresholds.tilt_warning, '°', 3],
        // Shock / Impact
        ['shock', 'Shock / High Dynamic G', dynamicG >= this.thresholds.shock_critical, dynamicG >= this.thresholds.shock_warning, dynamicG, this.thresholds.shock_critical, this.thresholds.shock_warning, 'G', 3],
        // Sustained Vibration (2s RMS)
        ['vibration', 'Sustained Vibration', rmsVibration >= this.thresholds.vibration_critical, rmsVibration >= this.thresholds.vibration_warning, rmsVibration, this.thresholds.vibration_critical, this.thresholds.vibration_warning, 'G RMS', 3],
        // Sudden Rotation (Gyro)
        ['rotation', 'Rapid Angular Rotation', maxGyro >= this.thresholds.rotation_critical, maxGyro >= this.thresholds.rotation_warning, maxGyro, this.thresholds.rotation_critical, this.thresholds.rotation_warning, '°/s', 3],
        // Temperature
        ['temp', 'High Die Temperature', temp >= this.thresholds.temp_critical, temp >= this.thresholds.temp_warning, temp, this.thresholds.temp_critical, this.thresholds.temp_warning, '°C', 3]
      ];

      // Evaluate each rule with Anti-False-Alarm hysteresis and cooldown (Item B4)
      for (const [key, name, isCrit, isWarn, measuredVal, critThresh, warnThresh, unit, consecutiveNeed] of rules) {
        const severity = isCrit ? 'critical' : (isWarn ? 'warning' : 'normal');
        let streak = this.sampleStreak.get(key) || { countAbove: 0, countBelow: 0, currentSev: 'normal' };

        if (severity !== 'normal') {
          streak.countAbove++;
          streak.countBelow = 0;
        } else {
          streak.countBelow++;
          streak.countAbove = 0;
        }

        // Fire when condition holds for N consecutive samples (Item B4)
        if (severity !== 'normal' && streak.countAbove >= consecutiveNeed) {
          const active = this.activeAlerts.get(key);
          const lastNotif = this.lastNotified.get(key) || 0;
          const cooldownElapsed = (now - lastNotif) > 10000; // 10s cooldown per alert type

          if (!active || active.severity !== severity || cooldownElapsed) {
            const threshVal = severity === 'critical' ? critThresh : warnThresh;
            const alertEvent = {
              id: 'alt_' + now + '_' + Math.floor(Math.random() * 1000),
              key: key,
              name: name,
              severity: severity,
              measuredValue: Number(measuredVal.toFixed(2)),
              threshold: threshVal,
              unit: unit,
              timestamp: now,
              timeStr: new Date().toLocaleTimeString(),
              status: 'active',
              message: `${name}: ${measuredVal.toFixed(2)} ${unit} (threshold: ${threshVal} ${unit})`
            };

            this.activeAlerts.set(key, alertEvent);
            this.alertLog.unshift(alertEvent);
            if (this.alertLog.length > 300) this.alertLog.pop(); // bound memory
            newEvents.push(alertEvent);

            if (severity === 'critical') this.countsToday.critical++;
            else this.countsToday.warning++;
            this.lastAlertTime = now;

            if (cooldownElapsed) {
              this.playTone(severity);
              this.lastNotified.set(key, now);
              if (severity === 'critical') {
                this.sendBrowserNotification(`CRITICAL: ${name}`, alertEvent.message);
                this.dispatchWebhook(alertEvent);
              }
            }
          }
        } else if (severity === 'normal' && streak.countBelow >= 5) {
          // Clear after 5 consecutive normal samples (Item B4)
          if (this.activeAlerts.has(key)) {
            const resolved = this.activeAlerts.get(key);
            resolved.status = 'resolved';
            resolved.resolvedAt = now;
            this.activeAlerts.delete(key);
          }
        }
        this.sampleStreak.set(key, streak);
      }

      // 5. Determine Master Overall Status Banner (Item B1)
      let masterStatus = 'normal';
      let highestAlert = null;
      for (const alert of this.activeAlerts.values()) {
        if (alert.severity === 'critical') {
          masterStatus = 'critical';
          highestAlert = alert;
          break;
        } else if (alert.severity === 'warning') {
          masterStatus = 'warning';
          highestAlert = alert;
        }
      }

      return {
        masterStatus: masterStatus,
        highestAlert: highestAlert,
        activeAlerts: Array.from(this.activeAlerts.values()),
        newEvents: newEvents,
        alertLog: this.alertLog,
        countsToday: this.countsToday,
        lastAlertTime: this.lastAlertTime
      };
    }

    acknowledgeAlert(alertId) {
      for (const item of this.alertLog) {
        if (item.id === alertId) {
          item.status = 'acknowledged';
          break;
        }
      }
      for (const [key, active] of this.activeAlerts.entries()) {
        if (active.id === alertId) {
          active.status = 'acknowledged';
          break;
        }
      }
    }

    clearResolvedAlerts() {
      this.alertLog = this.alertLog.filter(a => a.status === 'active');
    }

    clearAllAlerts() {
      this.alertLog = [];
      this.activeAlerts.clear();
      this.sampleStreak.clear();
    }
  }

  // Export globally
  window.MpuAlertEngine = MpuAlertEngine;
})(window);
