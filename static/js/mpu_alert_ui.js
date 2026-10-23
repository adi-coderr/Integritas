/**
 * MPU-6050 Master Alert & Analytics UI Controller
 * ------------------------------------------------
 * Coordinates Master Status Banner, Tilt Gauges, 2s RMS Vibration,
 * Alert Log Table, Summary Cards, Thresholds Modal, CSV Export,
 * Print-to-PDF Report, and Presentation Mode.
 */

(function (window) {
  'use strict';

  class MpuAlertUI {
    constructor() {
      this.engine = new window.MpuAlertEngine();
      this.history = new window.MpuHistoryStore();
      this.simulator = new window.MpuSimulator((simData) => this.processReading(simData, true));

      this.currentFilter = 'all'; // 'all' | 'critical' | 'warning' | 'active' | 'ack'
      this.timeRangeMs = 60 * 1000; // default 1 min
      this.isSimMode = false;
      this.lastPacketTime = 0;
      this.packetIntervals = []; // for measured Hz computation (Item A3)
      this.measuredHz = 0.0;

      // Watchdog timer (runs every 500ms to detect Stale / Offline)
      this.watchdogTimer = setInterval(() => this.checkConnectionWatchdog(), 500);

      // DOM references initialized when mounted
      this.dom = {};
    }

    mount(containerId = 'mpuAlertSystemMount') {
      const mount = document.getElementById(containerId);
      if (!mount) {
        console.warn(`[AlertUI] Mount container #${containerId} not found.`);
        return;
      }

      mount.innerHTML = this.renderHTML();
      this.bindDOM();
      this.bindEvents();
      this.updateThresholdsInputs();

      // Hook into existing global update if available, or poll telemetry read-only
      this.hookExistingPipeline();
    }

    renderHTML() {
      return `
        <!-- B1 (P0): Top Master Overall Status Banner -->
        <div id="masterStatusBanner" class="mpu-master-banner status-normal">
          <div class="master-banner-left">
            <span class="master-status-indicator"></span>
            <div>
              <div id="masterStatusTitle" class="master-status-title">SYSTEM STATUS: NORMAL</div>
              <div id="masterStatusSub" class="master-status-sub">All structural telemetry parameters within nominal safe bounds</div>
            </div>
            <div id="simModeWatermark" class="sim-mode-watermark" style="display: none;">
              ⚠️ SIMULATED DATA (DEMO MODE)
            </div>
          </div>
          <div class="master-banner-actions">
            <!-- Audio chime toggle (Item B5) -->
            <button class="source-tab-btn" id="btnAlertAudio" onclick="window.mpuAlertUI.toggleAudio()" title="Toggle audible alert chime">
              🔔 Enable Sound
            </button>
            <!-- Browser Notification permission button (Item B8) -->
            <button class="source-tab-btn" id="btnBrowserNotif" onclick="window.mpuAlertUI.requestNotifications()" title="Enable OS / Browser push notifications">
              📢 Push Alerts
            </button>
            <!-- Test alert button (Item D2) -->
            <button class="source-tab-btn" onclick="window.mpuAlertUI.fireTestAlert()" title="Fire simulated sample alert to demonstrate toast, sound, banner, and log">
              ⚡ Test Alert
            </button>
            <!-- Thresholds settings button (Item B3) -->
            <button class="source-tab-btn" onclick="window.mpuAlertUI.openThresholdsModal()" title="Customize alert thresholds">
              ⚙ Thresholds
            </button>
            <!-- Fullscreen Presentation Mode (Item D5) -->
            <button class="source-tab-btn" onclick="window.mpuAlertUI.togglePresentationMode()" title="Toggle clean projector presentation view">
              📺 Present Mode
            </button>
            <!-- One-Click Printable PDF Report (Item C6) -->
            <button class="source-tab-btn" onclick="window.mpuAlertUI.openReportModal()" title="Generate printable PDF summary report">
              📄 PDF Report
            </button>
          </div>
        </div>

        <!-- D3 (P1): Project Header & How It Works Architecture Card -->
        <div class="mpu-system-overview">
          <div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" onclick="window.mpuAlertUI.toggleArchDiagram()">
            <div>
              <span style="font-weight: 700; color: #001D39; font-size: 0.85rem;">
                🏛 Structural Health Monitoring System • How It Works
              </span>
              <span style="display: block; font-size: 0.73rem; color: var(--text-muted); margin-top: 2px;">
                Autonomous multi-axis tilt, dynamic shock, vibration RMS, and thermal safety inspection pipeline.
              </span>
            </div>
            <span id="archToggleIcon" style="font-size: 0.85rem; color: #0A4174; font-weight: 700;">▼ Details</span>
          </div>

          <div id="archDiagramBox" class="architecture-diagram-box" style="display: none;">
            <div class="arch-node">
              ESP32 + MPU-6050
              <span>Hardware I2C Sensor (200ms)</span>
            </div>
            <div class="arch-arrow">➔</div>
            <div class="arch-node">
              Wi-Fi Direct Push
              <span>Port 3000 /api/sensor</span>
            </div>
            <div class="arch-arrow">➔</div>
            <div class="arch-node">
              Flask Engine
              <span>Ingestion & Sync</span>
            </div>
            <div class="arch-arrow">➔</div>
            <div class="arch-node">
              Live Dashboard
              <span>Attitude, Waves & RMS</span>
            </div>
            <div class="arch-arrow">➔</div>
            <div class="arch-node" style="border-color: #86efac; background: #f0fdf4;">
              Alert Engine
              <span>Debounce, Chimes & Log</span>
            </div>
          </div>
        </div>

        <!-- B7 (P1): Alert Summary Metrics Row -->
        <div class="alert-summary-grid">
          <div class="alert-summary-card">
            <div class="alert-summary-icon summary-icon-active">⚠️</div>
            <div>
              <div id="sumActiveAlerts" class="alert-summary-val">0</div>
              <div class="alert-summary-lbl">Active Alerts</div>
            </div>
          </div>
          <div class="alert-summary-card">
            <div class="alert-summary-icon summary-icon-warn">⚡</div>
            <div>
              <div id="sumWarningsToday" class="alert-summary-val">0</div>
              <div class="alert-summary-lbl">Warnings Logged</div>
            </div>
          </div>
          <div class="alert-summary-card">
            <div class="alert-summary-icon summary-icon-crit">🚨</div>
            <div>
              <div id="sumCriticalToday" class="alert-summary-val">0</div>
              <div class="alert-summary-lbl">Critical Alarms</div>
            </div>
          </div>
          <div class="alert-summary-card">
            <div class="alert-summary-icon summary-icon-clock">⏱</div>
            <div>
              <div id="sumLastAlertTime" class="alert-summary-val" style="font-size: 0.95rem;">None</div>
              <div class="alert-summary-lbl">Last Alert Time</div>
            </div>
          </div>
        </div>

        <!-- Real-Time Enhanced Gauges & Vibration RMS Row -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 16px;">
          <!-- A5 (P1): Dual Circular Angle Tilt Gauges (Safe, Warning, Critical zones) -->
          <div class="mpu-subcard">
            <div class="subcard-title">
              <span>Dual Pitch & Roll Tilt Gauges</span>
              <span id="gaugeStatusBadge" class="tag-level" style="font-size: 0.65rem;">SAFE (&lt;15°)</span>
            </div>
            <div class="tilt-gauges-row">
              <!-- Pitch Gauge -->
              <div class="tilt-gauge-box">
                <svg class="tilt-gauge-svg" viewBox="0 0 100 100">
                  <circle class="tilt-gauge-bg" cx="50" cy="50" r="42"/>
                  <!-- Safe zone (0-15 deg = 0 - 25% of 60 deg max) -->
                  <circle class="tilt-gauge-zone-safe" cx="50" cy="50" r="42" stroke-dasharray="66 198" stroke-dashoffset="0"/>
                  <!-- Warning zone (15-30 deg) -->
                  <circle class="tilt-gauge-zone-warn" cx="50" cy="50" r="42" stroke-dasharray="66 198" stroke-dashoffset="-66"/>
                  <!-- Critical zone (>30 deg) -->
                  <circle class="tilt-gauge-zone-crit" cx="50" cy="50" r="42" stroke-dasharray="132 132" stroke-dashoffset="-132"/>
                  <!-- Dynamic needle -->
                  <circle id="pitchGaugeNeedle" class="tilt-gauge-needle" cx="50" cy="50" r="42" stroke-dasharray="264" stroke-dashoffset="264"/>
                </svg>
                <div class="tilt-gauge-center">
                  <span id="gaugePitchVal" class="angle-deg">+0.0°</span>
                  <span class="angle-name">Pitch (θ)</span>
                </div>
              </div>

              <!-- Roll Gauge -->
              <div class="tilt-gauge-box">
                <svg class="tilt-gauge-svg" viewBox="0 0 100 100">
                  <circle class="tilt-gauge-bg" cx="50" cy="50" r="42"/>
                  <circle class="tilt-gauge-zone-safe" cx="50" cy="50" r="42" stroke-dasharray="66 198" stroke-dashoffset="0"/>
                  <circle class="tilt-gauge-zone-warn" cx="50" cy="50" r="42" stroke-dasharray="66 198" stroke-dashoffset="-66"/>
                  <circle class="tilt-gauge-zone-crit" cx="50" cy="50" r="42" stroke-dasharray="132 132" stroke-dashoffset="-132"/>
                  <circle id="rollGaugeNeedle" class="tilt-gauge-needle" cx="50" cy="50" r="42" stroke-dasharray="264" stroke-dashoffset="264"/>
                </svg>
                <div class="tilt-gauge-center">
                  <span id="gaugeRollVal" class="angle-deg">+0.0°</span>
                  <span class="angle-name">Roll (φ)</span>
                </div>
              </div>
            </div>
            <div style="font-size: 0.68rem; color: var(--text-muted); display: flex; justify-content: space-between; border-top: 1px solid #BDD8E9; padding-top: 6px;">
              <span>Safe: &lt;15°</span>
              <span style="color: #d97706;">Warn: 15°–30°</span>
              <span style="color: #dc2626;">Crit: &gt;30°</span>
            </div>
          </div>

          <!-- A6 (P1): Vibration Intensity & 2-Second Rolling RMS Meter -->
          <div class="mpu-subcard">
            <div class="subcard-title">
              <span>Dynamic Shock & 2s Rolling RMS Vibration</span>
              <span id="rmsVibrationPill" class="tag-level" style="font-size: 0.65rem;">NOMINAL</span>
            </div>
            <div class="vibration-rms-container">
              <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.72rem; font-weight: 600; color: #001D39;">2-Second Rolling RMS Vibration:</span>
                <span id="valRmsVibration" style="font-family: var(--font-mono); font-size: 1.15rem; font-weight: 700; color: #0A4174;">0.008 G</span>
              </div>
              <div class="vibration-rms-bar-track">
                <div id="barRmsVibration" class="vibration-rms-bar-fill"></div>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 0.66rem; color: var(--text-muted);">
                <span>0.00 G</span>
                <span>Warn: 0.05 G</span>
                <span>Crit: 0.15 G</span>
                <span>0.30 G+</span>
              </div>
            </div>

            <!-- Instantaneous Dynamic Acceleration abs(|a| - 1g) -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 0.74rem;">
              <span style="color: var(--text-muted);">Instantaneous Dynamic Accel (|a| - 1g):</span>
              <span id="valInstantDynamicG" style="font-family: var(--font-mono); font-weight: 700; color: #001D39;">0.002 G</span>
            </div>
            <!-- Measured Hz and Last Seen (Item A3) -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; font-size: 0.74rem; border-top: 1px solid #BDD8E9; padding-top: 6px;">
              <span style="color: var(--text-muted);">Measured Packet Frequency:</span>
              <span id="valMeasuredHz" style="font-family: var(--font-mono); font-weight: 700; color: #0A4174;">0.0 Hz</span>
            </div>
          </div>
        </div>

        <!-- C3 (P1): Comprehensive Rolling Statistics Grid (30-min buffer) -->
        <div class="mpu-analytics-card">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
              <span style="font-weight: 700; color: #001D39; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">
                📊 Rolling Telemetry Statistics (Last 30 Minutes)
              </span>
              <!-- C2 (P1): Time-Range Selector -->
              <div style="display: flex; gap: 4px;">
                <button class="alert-filter-btn active" onclick="window.mpuAlertUI.setTimeRange(60000, this)">1 Min</button>
                <button class="alert-filter-btn" onclick="window.mpuAlertUI.setTimeRange(300000, this)">5 Min</button>
                <button class="alert-filter-btn" onclick="window.mpuAlertUI.setTimeRange(1800000, this)">30 Min</button>
              </div>
            </div>
            <!-- C5 (P1): CSV Export Buttons -->
            <div style="display: flex; gap: 6px;">
              <button class="source-tab-btn" onclick="window.mpuAlertUI.history.exportReadingsCSV()" title="Download complete readings buffer as CSV">
                📥 Export Telemetry CSV
              </button>
              <button class="source-tab-btn" onclick="window.mpuAlertUI.history.exportAlertsCSV(window.mpuAlertUI.engine.alertLog)" title="Download alert log as CSV">
                📥 Export Alerts CSV
              </button>
            </div>
          </div>

          <div class="analytics-grid">
            <div class="stat-item">
              <span class="stat-lbl">Session Uptime</span>
              <div id="statUptime" class="stat-val">0m 0s</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Total Samples</span>
              <div id="statTotalSamples" class="stat-val">0</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Peak Dynamic Shock</span>
              <div id="statPeakShock" class="stat-val">0.00 G</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Max Recorded Tilt</span>
              <div id="statMaxTilt" class="stat-val">0.0°</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Avg / Max Ax</span>
              <div id="statAx" class="stat-val">+0.00 / 0.00</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Avg / Max Ay</span>
              <div id="statAy" class="stat-val">+0.00 / 0.00</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Avg / Max Az</span>
              <div id="statAz" class="stat-val">+1.00 / 1.00</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Avg / Max Gyro</span>
              <div id="statGyro" class="stat-val">0.0 / 0.0 °/s</div>
            </div>
            <div class="stat-item">
              <span class="stat-lbl">Avg / Max Temp</span>
              <div id="statTemp" class="stat-val">25.0 / 25.0 °C</div>
            </div>
          </div>
        </div>

        <!-- B6 (P0): Alert Log & Event History Table -->
        <div class="alert-log-card">
          <div class="alert-log-header">
            <div style="display: flex; align-items: center; gap: 10px;">
              <span style="font-weight: 700; color: #001D39; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em;">
                🚨 Real-Time Structural Alert Event Log
              </span>
              <span id="logBadgeCount" class="tag-level" style="font-size: 0.65rem;">0 EVENTS</span>
            </div>
            <div class="alert-filter-bar">
              <button class="alert-filter-btn active" onclick="window.mpuAlertUI.filterLog('all', this)">All</button>
              <button class="alert-filter-btn" onclick="window.mpuAlertUI.filterLog('critical', this)">Critical</button>
              <button class="alert-filter-btn" onclick="window.mpuAlertUI.filterLog('warning', this)">Warning</button>
              <button class="alert-filter-btn" onclick="window.mpuAlertUI.filterLog('active', this)">Active Only</button>
              <button class="alert-filter-btn" onclick="window.mpuAlertUI.filterLog('ack', this)">Acknowledged</button>
              <button class="alert-filter-btn" onclick="window.mpuAlertUI.clearAlertLog()" title="Clear resolved events from log">🗑 Clear Resolved</button>
            </div>
          </div>

          <div class="alert-table-wrapper">
            <table class="alert-table">
              <thead>
                <tr>
                  <th style="width: 80px;">Time</th>
                  <th style="width: 85px;">Severity</th>
                  <th style="width: 150px;">Alert Type</th>
                  <th style="width: 90px;">Measured</th>
                  <th style="width: 90px;">Threshold</th>
                  <th style="width: 100px;">Status</th>
                  <th>Action / Details</th>
                </tr>
              </thead>
              <tbody id="alertTableBody">
                <tr>
                  <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">
                    No alert events recorded. System operating in nominal safe state.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- D1 (P0): Demo Simulator Toolbar Bar -->
        <div style="background: #ffffff; border: 1.5px solid #BDD8E9; border-radius: 8px; padding: 12px 16px; margin-top: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
          <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <span style="font-weight: 700; color: #001D39; font-size: 0.8rem; text-transform: uppercase;">
              🎮 Judge Demo Scenario Simulator:
            </span>
            <select id="simScenarioSelect" onchange="window.mpuAlertUI.onScenarioChange(this.value)" style="padding: 5px 10px; font-size: 0.78rem; border: 1px solid #BDD8E9; border-radius: 6px; background: #f4f9fd; color: #001D39; font-weight: 600;">
              <option value="normal">Scenario 1: Normal Structural State (Baseline)</option>
              <option value="tilt">Scenario 2: Excessive Tilt Angle (18° Warning -> 34° Critical)</option>
              <option value="shock">Scenario 3: Severe Physical Impact Shock (>1.85G Pulse)</option>
              <option value="vibration">Scenario 4: Sustained High Vibration (2s RMS > 0.20G)</option>
              <option value="freefall">Scenario 5: Structural Drop / Free-Fall (&lt;0.20G)</option>
              <option value="rotation">Scenario 6: High Centrifugal Rotation (&gt;160°/s)</option>
              <option value="overheat">Scenario 7: Thermal Runaway Overheat (&gt;70°C)</option>
              <option value="fault">Scenario 8: Hardware Fault / Frozen I2C Bus</option>
              <option value="offline">Scenario 9: Sensor Offline Timeout (3s Stale -> 10s Offline)</option>
            </select>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="btn-capture" id="btnToggleSimMode" onclick="window.mpuAlertUI.toggleSimulator()" style="padding: 6px 14px; font-size: 0.76rem;">
              ▶ Start Demo Simulator
            </button>
          </div>
        </div>

        <!-- B3 (P0): Threshold Customization Modal -->
        <div id="thresholdModalBackdrop" class="threshold-modal-backdrop">
          <div class="threshold-modal-box">
            <div class="threshold-modal-header">
              <span style="font-weight: 700; color: #001D39; font-size: 0.95rem;">⚙ Alert Rules & Threshold Settings</span>
              <button onclick="window.mpuAlertUI.closeThresholdsModal()" style="background: none; border: none; font-size: 1.2rem; cursor: pointer; color: var(--text-muted);">&times;</button>
            </div>
            <div class="threshold-modal-body">
              <span style="font-size: 0.74rem; color: var(--text-muted);">
                Configure warning and critical alarm thresholds. Values are automatically persisted to localStorage and applied in real time.
              </span>

              <!-- Tilt Thresholds -->
              <div class="threshold-group">
                <div class="threshold-group-title">
                  <span>📐 Tilt Angle Thresholds (degrees)</span>
                  <span style="font-weight: normal; font-size: 0.7rem; color: var(--text-muted);">Pitch / Roll from baseline</span>
                </div>
                <div class="threshold-row">
                  <span>Tilt Warning (°):</span>
                  <input type="number" id="th_tilt_warning" class="threshold-input" step="1" min="1" max="90">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 15°</span>
                </div>
                <div class="threshold-row">
                  <span>Tilt Critical (°):</span>
                  <input type="number" id="th_tilt_critical" class="threshold-input" step="1" min="1" max="90">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 30°</span>
                </div>
              </div>

              <!-- Shock / Dynamic Accel -->
              <div class="threshold-group">
                <div class="threshold-group-title">
                  <span>💥 Shock & Dynamic G Thresholds (g)</span>
                  <span style="font-weight: normal; font-size: 0.7rem; color: var(--text-muted);">abs(|a| - 1g)</span>
                </div>
                <div class="threshold-row">
                  <span>Shock Warning (g):</span>
                  <input type="number" id="th_shock_warning" class="threshold-input" step="0.05" min="0.1" max="5.0">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 0.50 g</span>
                </div>
                <div class="threshold-row">
                  <span>Shock Critical (g):</span>
                  <input type="number" id="th_shock_critical" class="threshold-input" step="0.05" min="0.1" max="5.0">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 1.00 g</span>
                </div>
              </div>

              <!-- Sustained Vibration RMS -->
              <div class="threshold-group">
                <div class="threshold-group-title">
                  <span>〰 Sustained Vibration 2s RMS (g)</span>
                  <span style="font-weight: normal; font-size: 0.7rem; color: var(--text-muted);">2-second rolling envelope</span>
                </div>
                <div class="threshold-row">
                  <span>Vibration Warning (g RMS):</span>
                  <input type="number" id="th_vibration_warning" class="threshold-input" step="0.01" min="0.01" max="1.0">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 0.05 g</span>
                </div>
                <div class="threshold-row">
                  <span>Vibration Critical (g RMS):</span>
                  <input type="number" id="th_vibration_critical" class="threshold-input" step="0.01" min="0.01" max="1.0">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 0.15 g</span>
                </div>
              </div>

              <!-- Rotation Gyro -->
              <div class="threshold-group">
                <div class="threshold-group-title">
                  <span>🔄 Sudden Rotation Thresholds (°/s)</span>
                  <span style="font-weight: normal; font-size: 0.7rem; color: var(--text-muted);">Any Gyro axis</span>
                </div>
                <div class="threshold-row">
                  <span>Rotation Warning (°/s):</span>
                  <input type="number" id="th_rotation_warning" class="threshold-input" step="5" min="10" max="500">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 50 °/s</span>
                </div>
                <div class="threshold-row">
                  <span>Rotation Critical (°/s):</span>
                  <input type="number" id="th_rotation_critical" class="threshold-input" step="5" min="10" max="500">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 150 °/s</span>
                </div>
              </div>

              <!-- Free Fall & Temperature -->
              <div class="threshold-group">
                <div class="threshold-group-title">
                  <span>🌡 Free-Fall, Temperature & Timeout</span>
                </div>
                <div class="threshold-row">
                  <span>Free-Fall Cutoff (Total G):</span>
                  <input type="number" id="th_freefall_critical" class="threshold-input" step="0.05" min="0.05" max="0.8">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 0.30 G</span>
                </div>
                <div class="threshold-row">
                  <span>Temp Warning (°C):</span>
                  <input type="number" id="th_temp_warning" class="threshold-input" step="1" min="20" max="100">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 55 °C</span>
                </div>
                <div class="threshold-row">
                  <span>Temp Critical (°C):</span>
                  <input type="number" id="th_temp_critical" class="threshold-input" step="1" min="20" max="120">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 70 °C</span>
                </div>
                <div class="threshold-row">
                  <span>Sensor Stale Warning (s):</span>
                  <input type="number" id="th_offline_warning" class="threshold-input" step="0.5" min="1" max="20">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 3.0 s</span>
                </div>
                <div class="threshold-row">
                  <span>Sensor Offline Critical (s):</span>
                  <input type="number" id="th_offline_critical" class="threshold-input" step="0.5" min="2" max="60">
                  <span style="color: var(--text-muted); font-size: 0.7rem;">Default: 10.0 s</span>
                </div>
              </div>

              <!-- Webhook URL (Item B9) -->
              <div class="threshold-group">
                <div class="threshold-group-title">
                  <span>🌐 Optional External Webhook URL (B9)</span>
                </div>
                <div style="margin-top: 4px;">
                  <input type="text" id="th_webhook_url" class="threshold-input" placeholder="https://your-server.com/webhook/alerts">
                  <span style="font-size: 0.68rem; color: var(--text-muted); display: block; margin-top: 4px;">
                    POSTs JSON payload when critical alarms occur. Leave blank to disable.
                  </span>
                </div>
              </div>
            </div>

            <div class="threshold-modal-footer">
              <button class="source-tab-btn" onclick="window.mpuAlertUI.restoreDefaultThresholds()">↺ Restore Defaults</button>
              <div style="display: flex; gap: 8px;">
                <button class="source-tab-btn" onclick="window.mpuAlertUI.closeThresholdsModal()">Cancel</button>
                <button class="btn-capture" onclick="window.mpuAlertUI.saveThresholdsFromInputs()" style="padding: 6px 16px; font-size: 0.78rem;">Save & Apply</button>
              </div>
            </div>
          </div>
        </div>

        <!-- C6 (P2): Printable Summary Report Modal -->
        <div id="reportModalBackdrop" class="threshold-modal-backdrop">
          <div class="threshold-modal-box" style="max-width: 780px;">
            <div class="threshold-modal-header">
              <span style="font-weight: 700; color: #001D39; font-size: 0.95rem;">📄 MPU-6050 Structural Inspection Audit Report</span>
              <button onclick="window.mpuAlertUI.closeReportModal()" style="background: none; border: none; font-size: 1.2rem; cursor: pointer; color: var(--text-muted);">&times;</button>
            </div>
            <div id="printableReportContent" class="threshold-modal-body">
              <!-- Dynamically populated report content -->
            </div>
            <div class="threshold-modal-footer">
              <button class="source-tab-btn" onclick="window.mpuAlertUI.closeReportModal()">Close</button>
              <button class="btn-capture" onclick="window.print()" style="padding: 6px 16px; font-size: 0.78rem;">🖨 Print / Save to PDF</button>
            </div>
          </div>
        </div>
      `;
    }

    bindDOM() {
      this.dom = {
        banner: document.getElementById('masterStatusBanner'),
        bannerTitle: document.getElementById('masterStatusTitle'),
        bannerSub: document.getElementById('masterStatusSub'),
        simBadge: document.getElementById('simModeWatermark'),
        btnAudio: document.getElementById('btnAlertAudio'),
        btnSim: document.getElementById('btnToggleSimMode'),
        scenarioSelect: document.getElementById('simScenarioSelect'),
        sumActive: document.getElementById('sumActiveAlerts'),
        sumWarnings: document.getElementById('sumWarningsToday'),
        sumCritical: document.getElementById('sumCriticalToday'),
        sumLastTime: document.getElementById('sumLastAlertTime'),
        gaugeStatusBadge: document.getElementById('gaugeStatusBadge'),
        gaugePitchNeedle: document.getElementById('pitchGaugeNeedle'),
        gaugeRollNeedle: document.getElementById('rollGaugeNeedle'),
        gaugePitchVal: document.getElementById('gaugePitchVal'),
        gaugeRollVal: document.getElementById('gaugeRollVal'),
        rmsVibrationPill: document.getElementById('rmsVibrationPill'),
        valRmsVibration: document.getElementById('valRmsVibration'),
        barRmsVibration: document.getElementById('barRmsVibration'),
        valInstantDynamicG: document.getElementById('valInstantDynamicG'),
        valMeasuredHz: document.getElementById('valMeasuredHz'),
        statUptime: document.getElementById('statUptime'),
        statTotalSamples: document.getElementById('statTotalSamples'),
        statPeakShock: document.getElementById('statPeakShock'),
        statMaxTilt: document.getElementById('statMaxTilt'),
        statAx: document.getElementById('statAx'),
        statAy: document.getElementById('statAy'),
        statAz: document.getElementById('statAz'),
        statGyro: document.getElementById('statGyro'),
        statTemp: document.getElementById('statTemp'),
        tableBody: document.getElementById('alertTableBody'),
        logBadgeCount: document.getElementById('logBadgeCount'),
        modalBackdrop: document.getElementById('thresholdModalBackdrop'),
        reportBackdrop: document.getElementById('reportModalBackdrop'),
        reportContent: document.getElementById('printableReportContent'),
        archBox: document.getElementById('archDiagramBox'),
        archIcon: document.getElementById('archToggleIcon')
      };
    }

    bindEvents() {
      // Listen to keydown to exit presentation mode
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.body.classList.contains('presentation-mode')) {
          this.togglePresentationMode();
        }
      });
    }

    hookExistingPipeline() {
      // Periodically read data from existing backend polling loop or directly
      setInterval(async () => {
        if (this.isSimMode) return; // simulator drives UI in demo mode
        try {
          const resp = await fetch('/api/mpu/telemetry');
          if (resp.ok) {
            const data = await resp.json();
            if (data && data.latest) {
              const cur = data.latest;
              // Pass through if we have fresh packet
              if (cur.timestamp) {
                this.processReading(cur, false);
              }
            }
          }
        } catch (e) {}
      }, 100);
    }

    processReading(data, isSimulated = false) {
      const now = Date.now();

      // Compute packet rate in Hz (Item A3)
      if (this.lastPacketTime > 0) {
        const deltaSec = (now - this.lastPacketTime) / 1000;
        if (deltaSec > 0.01 && deltaSec < 5.0) {
          const instHz = 1.0 / deltaSec;
          this.packetIntervals.push(instHz);
          if (this.packetIntervals.length > 10) this.packetIntervals.shift();
          let sum = 0;
          for (const h of this.packetIntervals) sum += h;
          this.measuredHz = sum / this.packetIntervals.length;
        }
      }
      this.lastPacketTime = now;

      // Add to rolling history store (Item C1)
      this.history.addReading(data);
      const rms2s = this.history.get2sRmsVibration();

      // Evaluate alerts
      const elapsedSinceData = (now - this.lastPacketTime) / 1000;
      const evalRes = this.engine.evaluate(data, rms2s, elapsedSinceData);

      // Render UI components
      this.renderMasterBanner(evalRes.masterStatus, evalRes.highestAlert);
      this.renderSummaryCards(evalRes.activeAlerts.length, evalRes.countsToday, evalRes.lastAlertTime);
      this.renderTiltGauges(data.pitch || 0, data.roll || 0);
      this.renderVibrationRMS(rms2s, data.ax || 0, data.ay || 0, data.az || 1);
      this.renderStatistics();
      this.renderAlertTable(evalRes.alertLog);

      // Trigger toasts for new events
      if (evalRes.newEvents && evalRes.newEvents.length > 0) {
        for (const ev of evalRes.newEvents) {
          this.showToast(ev);
        }
      }
    }

    // Watchdog checking Stale (>3s) / Offline (>10s) (Item A3)
    checkConnectionWatchdog() {
      if (this.isSimMode && this.simulator.scenario !== 'offline') return;

      const now = Date.now();
      const elapsedSec = this.lastPacketTime > 0 ? (now - this.lastPacketTime) / 1000 : 999;

      if (elapsedSec >= 3.0) {
        this.measuredHz = 0.0;
        const evalRes = this.engine.evaluate(null, 0.0, elapsedSec);
        this.renderMasterBanner(evalRes.masterStatus, evalRes.highestAlert);
        this.renderSummaryCards(evalRes.activeAlerts.length, evalRes.countsToday, evalRes.lastAlertTime);
        this.renderAlertTable(evalRes.alertLog);

        if (this.dom.valMeasuredHz) {
          this.dom.valMeasuredHz.innerText = '0.0 Hz (Offline)';
          this.dom.valMeasuredHz.style.color = '#dc2626';
        }
      }
    }

    renderMasterBanner(status, highestAlert) {
      if (!this.dom.banner) return;
      this.dom.banner.className = `mpu-master-banner status-${status}`;

      if (status === 'critical') {
        this.dom.bannerTitle.innerText = 'CRITICAL ALARM: ' + (highestAlert ? highestAlert.name.toUpperCase() : 'ACTION REQUIRED');
        this.dom.bannerSub.innerText = highestAlert ? highestAlert.message : 'Immediate structural inspection required.';
      } else if (status === 'warning') {
        this.dom.bannerTitle.innerText = 'WARNING: ' + (highestAlert ? highestAlert.name.toUpperCase() : 'THRESHOLD EXCEEDED');
        this.dom.bannerSub.innerText = highestAlert ? highestAlert.message : 'Parameter deviation detected.';
      } else {
        this.dom.bannerTitle.innerText = 'SYSTEM STATUS: NORMAL';
        this.dom.bannerSub.innerText = 'All structural telemetry parameters within nominal safe bounds';
      }
    }

    renderSummaryCards(activeCount, countsToday, lastTime) {
      if (this.dom.sumActive) this.dom.sumActive.innerText = activeCount;
      if (this.dom.sumWarnings) this.dom.sumWarnings.innerText = countsToday.warning;
      if (this.dom.sumCritical) this.dom.sumCritical.innerText = countsToday.critical;
      if (this.dom.sumLastTime) {
        this.dom.sumLastTime.innerText = lastTime ? new Date(lastTime).toLocaleTimeString() : 'None';
      }
    }

    renderTiltGauges(pitch, roll) {
      if (!this.dom.gaugePitchNeedle || !this.dom.gaugeRollNeedle) return;

      const pClamp = Math.min(60, Math.max(0, Math.abs(pitch)));
      const rClamp = Math.min(60, Math.max(0, Math.abs(roll)));

      // Circumference of r=42 is ~264
      const pOffset = 264 - (pClamp / 60) * 264;
      const rOffset = 264 - (rClamp / 60) * 264;

      this.dom.gaugePitchNeedle.style.strokeDashoffset = pOffset;
      this.dom.gaugeRollNeedle.style.strokeDashoffset = rOffset;

      const maxAngle = Math.max(Math.abs(pitch), Math.abs(roll));
      if (maxAngle > this.engine.thresholds.tilt_critical) {
        this.dom.gaugePitchNeedle.style.stroke = '#dc2626';
        this.dom.gaugeRollNeedle.style.stroke = '#dc2626';
        if (this.dom.gaugeStatusBadge) {
          this.dom.gaugeStatusBadge.className = 'tag-shock-alarm';
          this.dom.gaugeStatusBadge.innerText = 'CRITICAL TILT (>30°)';
        }
      } else if (maxAngle > this.engine.thresholds.tilt_warning) {
        this.dom.gaugePitchNeedle.style.stroke = '#d97706';
        this.dom.gaugeRollNeedle.style.stroke = '#d97706';
        if (this.dom.gaugeStatusBadge) {
          this.dom.gaugeStatusBadge.className = 'tag-tilt';
          this.dom.gaugeStatusBadge.innerText = 'WARNING TILT (15°-30°)';
        }
      } else {
        this.dom.gaugePitchNeedle.style.stroke = '#0A4174';
        this.dom.gaugeRollNeedle.style.stroke = '#4E8EA2';
        if (this.dom.gaugeStatusBadge) {
          this.dom.gaugeStatusBadge.className = 'tag-level';
          this.dom.gaugeStatusBadge.innerText = 'SAFE (<15°)';
        }
      }

      const pSign = pitch >= 0 ? '+' : '';
      const rSign = roll >= 0 ? '+' : '';
      if (this.dom.gaugePitchVal) this.dom.gaugePitchVal.innerText = `${pSign}${pitch.toFixed(1)}°`;
      if (this.dom.gaugeRollVal) this.dom.gaugeRollVal.innerText = `${rSign}${roll.toFixed(1)}°`;
    }

    renderVibrationRMS(rmsVal, ax, ay, az) {
      if (!this.dom.valRmsVibration || !this.dom.barRmsVibration) return;

      this.dom.valRmsVibration.innerText = `${rmsVal.toFixed(3)} G`;
      const pct = Math.min(100, Math.max(3, (rmsVal / 0.30) * 100));
      this.dom.barRmsVibration.style.width = `${pct}%`;

      if (rmsVal >= this.engine.thresholds.vibration_critical) {
        if (this.dom.rmsVibrationPill) {
          this.dom.rmsVibrationPill.className = 'tag-shock-alarm';
          this.dom.rmsVibrationPill.innerText = 'CRITICAL VIBRATION';
        }
      } else if (rmsVal >= this.engine.thresholds.vibration_warning) {
        if (this.dom.rmsVibrationPill) {
          this.dom.rmsVibrationPill.className = 'tag-tilt';
          this.dom.rmsVibrationPill.innerText = 'ELEVATED VIBRATION';
        }
      } else {
        if (this.dom.rmsVibrationPill) {
          this.dom.rmsVibrationPill.className = 'tag-level';
          this.dom.rmsVibrationPill.innerText = 'NOMINAL';
        }
      }

      const tot = Math.sqrt(ax * ax + ay * ay + az * az);
      const dynamicG = Math.abs(tot - 1.0);
      if (this.dom.valInstantDynamicG) {
        this.dom.valInstantDynamicG.innerText = `${dynamicG.toFixed(3)} G`;
      }
      if (this.dom.valMeasuredHz) {
        this.dom.valMeasuredHz.innerText = `${this.measuredHz.toFixed(1)} Hz`;
        this.dom.valMeasuredHz.style.color = '#0A4174';
      }
    }

    renderStatistics() {
      const stats = this.history.getSummaryStats();
      if (this.dom.statUptime) this.dom.statUptime.innerText = stats.uptimeStr;
      if (this.dom.statTotalSamples) this.dom.statTotalSamples.innerText = stats.totalSamples.toLocaleString();
      if (this.dom.statPeakShock) this.dom.statPeakShock.innerText = stats.peakShock;
      if (this.dom.statMaxTilt) this.dom.statMaxTilt.innerText = stats.maxTilt;
      if (this.dom.statAx) this.dom.statAx.innerText = `${stats.avgAx} / ${stats.maxAx}`;
      if (this.dom.statAy) this.dom.statAy.innerText = `${stats.avgAy} / ${stats.maxAy}`;
      if (this.dom.statAz) this.dom.statAz.innerText = `${stats.avgAz} / ${stats.maxAz}`;
      if (this.dom.statGyro) this.dom.statGyro.innerText = `${stats.avgGx} / ${stats.maxGx} °/s`;
      if (this.dom.statTemp) this.dom.statTemp.innerText = `${stats.avgTemp} / ${stats.maxTemp}`;
    }

    renderAlertTable(alertLog) {
      if (!this.dom.tableBody) return;
      if (this.dom.logBadgeCount) this.dom.logBadgeCount.innerText = `${alertLog.length} EVENTS`;

      const filtered = alertLog.filter(a => {
        if (this.currentFilter === 'critical') return a.severity === 'critical';
        if (this.currentFilter === 'warning') return a.severity === 'warning';
        if (this.currentFilter === 'active') return a.status === 'active';
        if (this.currentFilter === 'ack') return a.status === 'acknowledged';
        return true;
      });

      if (filtered.length === 0) {
        this.dom.tableBody.innerHTML = `
          <tr>
            <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 20px;">
              No alerts matching filter "${this.currentFilter.toUpperCase()}".
            </td>
          </tr>
        `;
        return;
      }

      this.dom.tableBody.innerHTML = filtered.slice(0, 50).map(a => {
        const isCrit = a.severity === 'critical';
        const badgeClass = isCrit ? 'tag-shock-alarm' : 'tag-tilt';
        const rowClass = isCrit ? 'row-critical' : 'row-warning';
        const ackButton = a.status === 'active'
          ? `<button class="alert-filter-btn" style="padding: 2px 8px; font-size: 0.68rem;" onclick="window.mpuAlertUI.engine.acknowledgeAlert('${a.id}')">Ack</button>`
          : `<span style="font-size: 0.68rem; color: var(--text-muted);">${a.status}</span>`;

        return `
          <tr class="${rowClass}">
            <td style="font-family: var(--font-mono); font-weight: 600;">${a.timeStr}</td>
            <td><span class="${badgeClass}" style="font-size: 0.62rem; padding: 2px 6px;">${a.severity.toUpperCase()}</span></td>
            <td style="font-weight: 600; color: #001D39;">${a.name}</td>
            <td style="font-family: var(--font-mono);">${a.measuredValue} ${a.unit}</td>
            <td style="font-family: var(--font-mono); color: var(--text-muted);">${a.threshold} ${a.unit}</td>
            <td><span style="font-size: 0.68rem; font-weight: 600; color: ${a.status === 'active' ? '#dc2626' : '#0A4174'};">${a.status.toUpperCase()}</span></td>
            <td>${ackButton} <span style="font-size: 0.7rem; color: var(--text-muted); margin-left: 6px;">${a.message}</span></td>
          </tr>
        `;
      }).join('');
    }

    showToast(alertEvent) {
      const container = document.getElementById('toastContainer');
      if (!container) return;

      const toast = document.createElement('div');
      toast.className = `toast-item toast-${alertEvent.severity}`;
      toast.style.cssText = `
        background: ${alertEvent.severity === 'critical' ? '#fef2f2' : '#fffbeb'};
        border: 1.5px solid ${alertEvent.severity === 'critical' ? '#f87171' : '#fde68a'};
        color: ${alertEvent.severity === 'critical' ? '#7f1d1d' : '#78350f'};
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        animation: fadeIn 0.2s ease-out;
      `;
      toast.innerHTML = `
        <div>
          <strong style="display: block; font-size: 0.8rem;">${alertEvent.severity.toUpperCase()}: ${alertEvent.name}</strong>
          <span style="font-size: 0.74rem;">${alertEvent.message}</span>
        </div>
        <button onclick="this.parentElement.remove()" style="background: none; border: none; font-size: 1.1rem; cursor: pointer; color: inherit;">&times;</button>
      `;

      container.appendChild(toast);
      setTimeout(() => {
        if (toast.parentElement) toast.remove();
      }, 5000);
    }

    // Audio Mute toggle
    toggleAudio() {
      this.engine.initAudio();
      const muted = this.engine.toggleMute();
      if (this.dom.btnAudio) {
        this.dom.btnAudio.innerText = muted ? '🔇 Sound: Muted' : '🔔 Sound: ON';
      }
    }

    async requestNotifications() {
      const granted = await this.engine.requestNotificationPermission();
      if (this.dom.btnBrowserNotif) {
        this.dom.btnBrowserNotif.innerText = granted ? '📢 Notifications: ON' : '📢 Notifications: Blocked';
      }
    }

    // Fire simulated test alert (Item D2)
    fireTestAlert() {
      this.engine.initAudio();
      const testEvent = {
        id: 'test_' + Date.now(),
        key: 'shock',
        name: 'Manual Test Impact',
        severity: 'critical',
        measuredValue: 1.85,
        threshold: 1.0,
        unit: 'G',
        timestamp: Date.now(),
        timeStr: new Date().toLocaleTimeString(),
        status: 'active',
        message: 'Manual test alarm injected: 1.85 G dynamic impact (threshold: 1.00 G)'
      };
      this.engine.alertLog.unshift(testEvent);
      this.engine.playTone('critical');
      this.showToast(testEvent);
      this.renderMasterBanner('critical', testEvent);
      this.renderAlertTable(this.engine.alertLog);
    }

    toggleSimulator() {
      this.isSimMode = !this.isSimMode;
      if (this.isSimMode) {
        const scenario = this.dom.scenarioSelect ? this.dom.scenarioSelect.value : 'normal';
        this.simulator.start(scenario);
        if (this.dom.btnSim) {
          this.dom.btnSim.innerText = '⏹ Stop Simulator';
          this.dom.btnSim.className = 'btn-secondary';
        }
        if (this.dom.simBadge) this.dom.simBadge.style.display = 'inline-flex';
      } else {
        this.simulator.stop();
        if (this.dom.btnSim) {
          this.dom.btnSim.innerText = '▶ Start Demo Simulator';
          this.dom.btnSim.className = 'btn-capture';
        }
        if (this.dom.simBadge) this.dom.simBadge.style.display = 'none';
      }
    }

    onScenarioChange(scenario) {
      if (this.simulator) {
        this.simulator.setScenario(scenario);
      }
    }

    toggleArchDiagram() {
      if (!this.dom.archBox) return;
      const isHidden = this.dom.archBox.style.display === 'none';
      this.dom.archBox.style.display = isHidden ? 'flex' : 'none';
      if (this.dom.archIcon) this.dom.archIcon.innerText = isHidden ? '▲ Hide' : '▼ Details';
    }

    togglePresentationMode() {
      document.body.classList.toggle('presentation-mode');
    }

    filterLog(filterType, btn) {
      this.currentFilter = filterType;
      const btns = document.querySelectorAll('.alert-filter-btn');
      btns.forEach(b => b.classList.remove('active'));
      if (btn) btn.classList.add('active');
      this.renderAlertTable(this.engine.alertLog);
    }

    clearAlertLog() {
      this.engine.clearResolvedAlerts();
      this.renderAlertTable(this.engine.alertLog);
    }

    setTimeRange(durationMs, btn) {
      this.timeRangeMs = durationMs;
      const btns = btn.parentElement.querySelectorAll('button');
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }

    openThresholdsModal() {
      this.updateThresholdsInputs();
      if (this.dom.modalBackdrop) this.dom.modalBackdrop.style.display = 'flex';
    }

    closeThresholdsModal() {
      if (this.dom.modalBackdrop) this.dom.modalBackdrop.style.display = 'none';
    }

    updateThresholdsInputs() {
      const th = this.engine.thresholds;
      const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
      };
      setVal('th_tilt_warning', th.tilt_warning);
      setVal('th_tilt_critical', th.tilt_critical);
      setVal('th_shock_warning', th.shock_warning);
      setVal('th_shock_critical', th.shock_critical);
      setVal('th_vibration_warning', th.vibration_warning);
      setVal('th_vibration_critical', th.vibration_critical);
      setVal('th_rotation_warning', th.rotation_warning);
      setVal('th_rotation_critical', th.rotation_critical);
      setVal('th_freefall_critical', th.freefall_critical);
      setVal('th_temp_warning', th.temp_warning);
      setVal('th_temp_critical', th.temp_critical);
      setVal('th_offline_warning', th.offline_warning);
      setVal('th_offline_critical', th.offline_critical);
      setVal('th_webhook_url', th.webhook_url || '');
    }

    saveThresholdsFromInputs() {
      const getVal = (id, fallback) => {
        const el = document.getElementById(id);
        return el ? parseFloat(el.value) || fallback : fallback;
      };
      const newTh = {
        tilt_warning: getVal('th_tilt_warning', 15),
        tilt_critical: getVal('th_tilt_critical', 30),
        shock_warning: getVal('th_shock_warning', 0.50),
        shock_critical: getVal('th_shock_critical', 1.00),
        vibration_warning: getVal('th_vibration_warning', 0.05),
        vibration_critical: getVal('th_vibration_critical', 0.15),
        rotation_warning: getVal('th_rotation_warning', 50),
        rotation_critical: getVal('th_rotation_critical', 150),
        freefall_critical: getVal('th_freefall_critical', 0.30),
        temp_warning: getVal('th_temp_warning', 55),
        temp_critical: getVal('th_temp_critical', 70),
        offline_warning: getVal('th_offline_warning', 3.0),
        offline_critical: getVal('th_offline_critical', 10.0),
        webhook_url: (document.getElementById('th_webhook_url')?.value || '').trim()
      };
      this.engine.saveThresholds(newTh);
      this.closeThresholdsModal();
    }

    restoreDefaultThresholds() {
      this.engine.resetThresholds();
      this.updateThresholdsInputs();
    }

    openReportModal() {
      const stats = this.history.getSummaryStats();
      const alertCount = this.engine.alertLog.length;
      const html = `
        <div style="border-bottom: 2px solid #001D39; padding-bottom: 12px; margin-bottom: 14px;">
          <h2 style="color: #001D39; margin: 0 0 4px 0; font-size: 1.3rem;">UNet-VGG16 & MPU-6050 Structural Telemetry Report</h2>
          <span style="font-size: 0.76rem; color: var(--text-muted);">
            Generated: ${new Date().toLocaleString()} • Inspection Session Duration: ${stats.uptimeStr}
          </span>
        </div>

        <div style="margin-bottom: 16px;">
          <h4 style="color: #0A4174; margin: 0 0 8px 0; font-size: 0.9rem;">1. Executive Summary & Telemetry Metrics</h4>
          <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
            <tr><td style="padding: 4px; font-weight: 600;">Total Ingested Samples:</td><td>${stats.totalSamples}</td><td style="padding: 4px; font-weight: 600;">Peak Shock Event:</td><td>${stats.peakShock}</td></tr>
            <tr><td style="padding: 4px; font-weight: 600;">Max Inclination Tilt:</td><td>${stats.maxTilt}</td><td style="padding: 4px; font-weight: 600;">Max Die Temperature:</td><td>${stats.maxTemp}</td></tr>
            <tr><td style="padding: 4px; font-weight: 600;">Average / Max Ax:</td><td>${stats.avgAx} / ${stats.maxAx} g</td><td style="padding: 4px; font-weight: 600;">Average / Max Gyro:</td><td>${stats.avgGx} / ${stats.maxGx} °/s</td></tr>
          </table>
        </div>

        <div>
          <h4 style="color: #0A4174; margin: 0 0 8px 0; font-size: 0.9rem;">2. Incident & Alert Audit Log (${alertCount} total events)</h4>
          <table style="width: 100%; border-collapse: collapse; font-size: 0.75rem; text-align: left;">
            <thead>
              <tr style="background: #f4f9fd; border-bottom: 1px solid #BDD8E9;">
                <th style="padding: 6px;">Time</th>
                <th style="padding: 6px;">Severity</th>
                <th style="padding: 6px;">Rule Name</th>
                <th style="padding: 6px;">Measured Value</th>
                <th style="padding: 6px;">Threshold</th>
                <th style="padding: 6px;">Status</th>
              </tr>
            </thead>
            <tbody>
              ${this.engine.alertLog.slice(0, 15).map(a => `
                <tr style="border-bottom: 1px solid #eee;">
                  <td style="padding: 5px;">${a.timeStr}</td>
                  <td style="padding: 5px; font-weight: 700; color: ${a.severity === 'critical' ? '#dc2626' : '#d97706'};">${a.severity.toUpperCase()}</td>
                  <td style="padding: 5px;">${a.name}</td>
                  <td style="padding: 5px;">${a.measuredValue} ${a.unit}</td>
                  <td style="padding: 5px;">${a.threshold} ${a.unit}</td>
                  <td style="padding: 5px;">${a.status}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      `;
      if (this.dom.reportContent) this.dom.reportContent.innerHTML = html;
      if (this.dom.reportBackdrop) this.dom.reportBackdrop.style.display = 'flex';
    }

    closeReportModal() {
      if (this.dom.reportBackdrop) this.dom.reportBackdrop.style.display = 'none';
    }
  }

  // Export and instantiate globally
  window.MpuAlertUI = MpuAlertUI;
  window.addEventListener('DOMContentLoaded', () => {
    window.mpuAlertUI = new MpuAlertUI();
    window.mpuAlertUI.mount('mpuAlertSystemMount');
  });
})(window);
