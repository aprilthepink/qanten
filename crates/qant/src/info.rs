//! Device information, sensor readings and performance counters.

use std::time::Duration;

/// Board sensor readings. Temperatures in °C, voltages in V, current in A.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SensorReadings {
    pub temp_pd1: f64,
    pub temp_pd2: f64,
    pub temp_pd3: f64,
    pub temp_dac_1: f64,
    pub temp_dac_2: f64,
    pub voltage_1v8: f64,
    pub voltage_3v3: f64,
    pub voltage_6v0_1: f64,
    pub voltage_6v0_2: f64,
    pub voltage_6v2: f64,
    pub voltage_6v5: f64,
    pub voltage_8v0_1: f64,
    pub voltage_8v0_2: f64,
    pub voltage_12v: f64,
    pub voltage_12v5: f64,
    pub voltage_n12v5: f64,
    pub current_12v: f64,
}

impl SensorReadings {
    /// All readings as `(name, value, unit)` rows, handy for printing.
    pub fn rows(&self) -> [(&'static str, f64, &'static str); 17] {
        [
            ("temp_pd1", self.temp_pd1, "°C"),
            ("temp_pd2", self.temp_pd2, "°C"),
            ("temp_pd3", self.temp_pd3, "°C"),
            ("temp_dac_1", self.temp_dac_1, "°C"),
            ("temp_dac_2", self.temp_dac_2, "°C"),
            ("voltage_1v8", self.voltage_1v8, "V"),
            ("voltage_3v3", self.voltage_3v3, "V"),
            ("voltage_6v0_1", self.voltage_6v0_1, "V"),
            ("voltage_6v0_2", self.voltage_6v0_2, "V"),
            ("voltage_6v2", self.voltage_6v2, "V"),
            ("voltage_6v5", self.voltage_6v5, "V"),
            ("voltage_8v0_1", self.voltage_8v0_1, "V"),
            ("voltage_8v0_2", self.voltage_8v0_2, "V"),
            ("voltage_12v", self.voltage_12v, "V"),
            ("voltage_12v5", self.voltage_12v5, "V"),
            ("voltage_n12v5", self.voltage_n12v5, "V"),
            ("current_12v", self.current_12v, "A"),
        ]
    }

    /// Board power draw on the 12 V rail, in watts.
    pub fn power_12v_watts(&self) -> f64 {
        self.voltage_12v * self.current_12v
    }
}

/// Firmware and board identification.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct VersionInfo {
    pub firmware: String,
    pub zephyr: String,
    pub board_serial: String,
}

/// Sensors plus version info of one NPU.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DeviceInfo {
    pub sensors: SensorReadings,
    pub version: VersionInfo,
}

/// NPU performance counters since the last reset.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PerfCounters {
    /// Wall time since the last reset.
    pub elapsed: Duration,
    /// Time the NPU spent idle (stalled) in that window.
    pub stalled: Duration,
}

impl PerfCounters {
    /// Fraction of time the NPU was busy, in `[0, 1]`.
    pub fn utilization(&self) -> f64 {
        let total = self.elapsed.as_secs_f64();
        if total <= 0.0 {
            return 0.0;
        }
        (1.0 - self.stalled.as_secs_f64() / total).clamp(0.0, 1.0)
    }
}
