/**
 * MPU-6050 Rolling History Store & Statistics Calculator
 * ------------------------------------------------------
 * Keeps a memory-bounded client-side buffer of readings (last 30 minutes).
 * Computes 2-second dynamic acceleration RMS, live min/max/avg stats,
 * and handles CSV export of readings and alerts.
 */

(function (window) {
  'use strict';

  class MpuHistoryStore {
    constructor(maxDurationMs = 30 * 60 * 1000) { // 30 minutes window
      this.maxDurationMs = maxDurationMs;
      this.readings = []; // array of { t, ax, ay, az, gx, gy, gz, pitch, roll, totalG, dynamicG, temp, rssi }
      this.rmsWindow2s = []; // array of { t, dynamicG } for last 2000 ms
      this.sessionStartTime = Date.now();
      this.totalSamplesReceived = 0;

      // Cumulative stats
      this.stats = {
        minAx: Infinity, maxAx: -Infinity, sumAx: 0,
        minAy: Infinity, maxAy: -Infinity, sumAy: 0,
        minAz: Infinity, maxAz: -Infinity, sumAz: 0,
        minGx: Infinity, maxGx: -Infinity, sumGx: 0,
        minGy: Infinity, maxGy: -Infinity, sumGy: 0,
        minGz: Infinity, maxGz: -Infinity, sumGz: 0,
        minTemp: Infinity, maxTemp: -Infinity, sumTemp: 0,
        peakShockG: 0,
        maxTiltDeg: 0
      };
    }

    addReading(data) {
      const now = Date.now();
      this.totalSamplesReceived++;

      const ax = typeof data.ax === 'number' ? data.ax : 0.0;
      const ay = typeof data.ay === 'number' ? data.ay : 0.0;
      const az = typeof data.az === 'number' ? data.az : 1.0;
      const gx = typeof data.gx === 'number' ? data.gx : 0.0;
      const gy = typeof data.gy === 'number' ? data.gy : 0.0;
      const gz = typeof data.gz === 'number' ? data.gz : 0.0;
      const temp = typeof data.temp === 'number' ? data.temp : 25.0;
      const pitch = typeof data.pitch === 'number' ? data.pitch : 0.0;
      const roll = typeof data.roll === 'number' ? data.roll : 0.0;
      const totalG = typeof data.total_g === 'number' ? data.total_g : Math.sqrt(ax * ax + ay * ay + az * az);
      const dynamicG = Math.abs(totalG - 1.0);

      const entry = {
        t: now,
        ax: ax,
        ay: ay,
        az: az,
        gx: gx,
        gy: gy,
        gz: gz,
        temp: temp,
        pitch: pitch,
        roll: roll,
        totalG: totalG,
        dynamicG: dynamicG,
        rssi: data.rssi || 0
      };

      this.readings.push(entry);

      // Append to 2-second dynamic RMS window
      this.rmsWindow2s.push({ t: now, dynamicG: dynamicG });
      const cutoff2s = now - 2000;
      while (this.rmsWindow2s.length > 0 && this.rmsWindow2s[0].t < cutoff2s) {
        this.rmsWindow2s.shift();
      }

      // Memory boundary cleanup: prune older than maxDurationMs
      const cutoffHistory = now - this.maxDurationMs;
      while (this.readings.length > 0 && this.readings[0].t < cutoffHistory) {
        this.readings.shift();
      }

      // Update running stats
      this.stats.minAx = Math.min(this.stats.minAx, ax);
      this.stats.maxAx = Math.max(this.stats.maxAx, ax);
      this.stats.sumAx += ax;

      this.stats.minAy = Math.min(this.stats.minAy, ay);
      this.stats.maxAy = Math.max(this.stats.maxAy, ay);
      this.stats.sumAy += ay;

      this.stats.minAz = Math.min(this.stats.minAz, az);
      this.stats.maxAz = Math.max(this.stats.maxAz, az);
      this.stats.sumAz += az;

      this.stats.minGx = Math.min(this.stats.minGx, gx);
      this.stats.maxGx = Math.max(this.stats.maxGx, gx);
      this.stats.sumGx += gx;

      this.stats.minGy = Math.min(this.stats.minGy, gy);
      this.stats.maxGy = Math.max(this.stats.maxGy, gy);
      this.stats.sumGy += gy;

      this.stats.minGz = Math.min(this.stats.minGz, gz);
      this.stats.maxGz = Math.max(this.stats.maxGz, gz);
      this.stats.sumGz += gz;

      this.stats.minTemp = Math.min(this.stats.minTemp, temp);
      this.stats.maxTemp = Math.max(this.stats.maxTemp, temp);
      this.stats.sumTemp += temp;

      this.stats.peakShockG = Math.max(this.stats.peakShockG, dynamicG);
      this.stats.maxTiltDeg = Math.max(this.stats.maxTiltDeg, Math.abs(pitch), Math.abs(roll));

      return entry;
    }

    // Computes 2-second dynamic acceleration RMS: sqrt(sum(dG^2) / N) (Item A6)
    get2sRmsVibration() {
      if (this.rmsWindow2s.length === 0) return 0.0;
      let sumSq = 0;
      for (const pt of this.rmsWindow2s) {
        sumSq += pt.dynamicG * pt.dynamicG;
      }
      return Math.sqrt(sumSq / this.rmsWindow2s.length);
    }

    // Get filtered readings by time range (1m, 5m, 30m) (Item C2)
    getReadingsWindow(durationMs = 60 * 1000) {
      const now = Date.now();
      const cutoff = now - durationMs;
      return this.readings.filter(r => r.t >= cutoff);
    }

    // Summary statistics object (Item C3)
    getSummaryStats() {
      const n = this.totalSamplesReceived;
      const uptimeSec = Math.floor((Date.now() - this.sessionStartTime) / 1000);
      const mins = Math.floor(uptimeSec / 60);
      const secs = uptimeSec % 60;
      const uptimeStr = `${mins}m ${secs}s`;

      if (n === 0) {
        return {
          uptimeStr: uptimeStr,
          totalSamples: 0,
          peakShock: '0.00 G',
          maxTilt: '0.0°',
          avgAx: '0.00', minAx: '0.00', maxAx: '0.00',
          avgAy: '0.00', minAy: '0.00', maxAy: '0.00',
          avgAz: '1.00', minAz: '1.00', maxAz: '1.00',
          avgGx: '0.0', maxGx: '0.0',
          avgTemp: '25.0 °C', maxTemp: '25.0 °C'
        };
      }

      return {
        uptimeStr: uptimeStr,
        totalSamples: n,
        peakShock: `${this.stats.peakShockG.toFixed(2)} G`,
        maxTilt: `${this.stats.maxTiltDeg.toFixed(1)}°`,
        avgAx: (this.stats.sumAx / n).toFixed(2),
        minAx: this.stats.minAx.toFixed(2),
        maxAx: this.stats.maxAx.toFixed(2),
        avgAy: (this.stats.sumAy / n).toFixed(2),
        minAy: this.stats.minAy.toFixed(2),
        maxAy: this.stats.maxAy.toFixed(2),
        avgAz: (this.stats.sumAz / n).toFixed(2),
        minAz: this.stats.minAz.toFixed(2),
        maxAz: this.stats.maxAz.toFixed(2),
        avgGx: (this.stats.sumGx / n).toFixed(1),
        maxGx: this.stats.maxGx.toFixed(1),
        avgTemp: `${(this.stats.sumTemp / n).toFixed(1)} °C`,
        maxTemp: `${this.stats.maxTemp.toFixed(1)} °C`
      };
    }

    // CSV Exporter for Sensor Readings (Item C5)
    exportReadingsCSV() {
      if (this.readings.length === 0) {
        alert('No readings recorded in buffer yet.');
        return;
      }
      const headers = ['Timestamp_ISO', 'Timestamp_MS', 'Ax_g', 'Ay_g', 'Az_g', 'Gx_dps', 'Gy_dps', 'Gz_dps', 'Pitch_deg', 'Roll_deg', 'Total_G', 'Dynamic_G', 'Temp_C', 'RSSI_dBm'];
      const rows = this.readings.map(r => [
        new Date(r.t).toISOString(),
        r.t,
        r.ax.toFixed(3),
        r.ay.toFixed(3),
        r.az.toFixed(3),
        r.gx.toFixed(2),
        r.gy.toFixed(2),
        r.gz.toFixed(2),
        r.pitch.toFixed(2),
        r.roll.toFixed(2),
        r.totalG.toFixed(3),
        r.dynamicG.toFixed(3),
        r.temp.toFixed(1),
        r.rssi
      ]);

      const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
      this._triggerDownload(csvContent, `mpu6050_telemetry_${Date.now()}.csv`, 'text/csv');
    }

    // CSV Exporter for Alert Log (Item C5)
    exportAlertsCSV(alertLog) {
      if (!alertLog || alertLog.length === 0) {
        alert('Alert log is currently empty.');
        return;
      }
      const headers = ['ID', 'Timestamp_ISO', 'Severity', 'Rule_Key', 'Rule_Name', 'Measured_Value', 'Threshold', 'Unit', 'Status', 'Message'];
      const rows = alertLog.map(a => [
        a.id,
        new Date(a.timestamp).toISOString(),
        a.severity,
        a.key,
        `"${a.name}"`,
        a.measuredValue,
        a.threshold,
        `"${a.unit}"`,
        a.status,
        `"${a.message.replace(/"/g, '""')}"`
      ]);

      const csvContent = [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
      this._triggerDownload(csvContent, `mpu6050_alerts_${Date.now()}.csv`, 'text/csv');
    }

    _triggerDownload(content, filename, mimeType) {
      const blob = new Blob([content], { type: mimeType + ';charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  }

  window.MpuHistoryStore = MpuHistoryStore;
})(window);
