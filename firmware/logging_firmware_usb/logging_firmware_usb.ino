#include <WiFi.h>
#include <WebServer.h>
#include <stdlib.h>

// ================= LOGGING MODE =================
// false = WiFi OFF, data sent over the USB cable (use logger_usb.py).
//         No radio near the ADC, so no WiFi noise and no WiFi dropouts.
// true  = WiFi access point "BMS-Monitor" + /data page (use logger.py).
const bool USE_WIFI = false;
// ================================================

const char* AP_SSID = "BMS-Monitor";
const char* AP_PASSWORD = "bms12345";   // at least 8 characters

// ESP32-S3 pins.
// ADC1 = GPIO 1-10: safe to use while WiFi is on.
// ADC2 = GPIO 11-20: shared with WiFi, so readings can fail or glitch.
#define TAP1_PIN 7
#define TAP2_PIN 6
#define TAP3_PIN 5
#define TAP4_PIN 4
#define CURRENT_PIN 8     // ADC1, with capacitor to GND
#define TEMP1_PIN 3
#define TEMP2_PIN 9

// ===== YOUR calibration values =====
const float TAP1_RATIO = 0.1596;
const float TAP2_RATIO = 0.1583;
const float TAP3_RATIO = 0.1611;
const float TAP4_RATIO = 0.1584;
const float CURRENT_DIVIDER_RATIO = 0.4870;
const float ACS712_ZERO_OFFSET = 2.4279;    // the EKF also re-learns the zero at every rest
const float ACS712_SENSITIVITY = 0.1929;    // calibrated at the 2.6 A discharge point
// ===================================

// Sign convention: discharge = positive, charge = negative (Plett's convention).
const float CURRENT_SIGN = -1.0;

// Thermistor constants (10K NTC, B=3950)
const float NTC_NOMINAL_RES = 10000.0;
const float NTC_NOMINAL_TEMP = 25.0;
const float NTC_B_COEFF = 3950.0;
const float SERIES_RESISTOR = 10000.0;

// ----- Sampling plan (one reading published per second) -----
// Current: plain average of ~2000 samples per second (handles pulsed current).
// Cell taps: ~500 samples per tap per second, combined with a TRIMMED MEAN
//   (sort, drop the highest and lowest 10%, average the rest).
// Temperatures: median of 31 quick samples.
// DUMMY READS: one throwaway reading after switching ADC pins, so the
// previous pin's voltage doesn't leak into the next reading.
const unsigned long REPORT_INTERVAL_MS = 1000;
const unsigned long CURRENT_SAMPLE_INTERVAL_US = 500;
const unsigned long TAP_SAMPLE_INTERVAL_US = 2000;
const int MAX_TAP_SAMPLES = 600;
const float TRIM_FRACTION = 0.10;
const int TEMP_SAMPLES = 31;

const int tapPins[4] = {TAP1_PIN, TAP2_PIN, TAP3_PIN, TAP4_PIN};
const float tapRatios[4] = {TAP1_RATIO, TAP2_RATIO, TAP3_RATIO, TAP4_RATIO};

WebServer server(80);

struct Reading {
  unsigned long timestamp_ms;
  float cell[4];
  float current;
  float currentMin;
  float currentMax;
  uint32_t samples;
  int vSamples;
  float temp1;
  float temp2;
};

Reading latest;

double currentSumMv = 0;
uint32_t currentCount = 0;
int currentMinMv = 100000;
int currentMaxMv = -1;
unsigned long lastCurrentSampleUs = 0;

int16_t tapBuf[4][MAX_TAP_SAMPLES];
int tapCount = 0;
unsigned long lastTapSampleUs = 0;

unsigned long windowStartMs = 0;

// ---------- Helpers ----------

int cleanReadMv(int pin) {
  analogReadMilliVolts(pin);           // throwaway read to settle the ADC on this pin
  return analogReadMilliVolts(pin);
}

float readMedianVolts(int pin) {
  int buf[TEMP_SAMPLES];
  analogReadMilliVolts(pin);           // throwaway read after switching to this pin
  for (int i = 0; i < TEMP_SAMPLES; i++) {
    buf[i] = analogReadMilliVolts(pin);
  }
  for (int i = 1; i < TEMP_SAMPLES; i++) {
    int key = buf[i];
    int j = i - 1;
    while (j >= 0 && buf[j] > key) {
      buf[j + 1] = buf[j];
      j--;
    }
    buf[j + 1] = key;
  }
  return buf[TEMP_SAMPLES / 2] / 1000.0;
}

int compareInt16(const void* a, const void* b) {
  int16_t x = *(const int16_t*)a;
  int16_t y = *(const int16_t*)b;
  return (x > y) - (x < y);
}

float trimmedMeanMv(int16_t* buf, int n) {
  if (n <= 0) return 0;
  qsort(buf, n, sizeof(int16_t), compareInt16);
  int cut = (int)(n * TRIM_FRACTION);
  long sum = 0;
  int count = 0;
  for (int i = cut; i < n - cut; i++) {
    sum += buf[i];
    count++;
  }
  return (float)sum / count;
}

float mvToAmps(float pinMv) {
  float acsVoltage = (pinMv / 1000.0) / CURRENT_DIVIDER_RATIO;
  return CURRENT_SIGN * (acsVoltage - ACS712_ZERO_OFFSET) / ACS712_SENSITIVITY;
}

float ntcToCelsius(int pin) {
  float vOut = readMedianVolts(pin);
  if (vOut <= 0.01) return -999.0;
  float resistance = SERIES_RESISTOR * (3.3 / vOut - 1.0);
  float steinhart = resistance / NTC_NOMINAL_RES;
  steinhart = log(steinhart);
  steinhart /= NTC_B_COEFF;
  steinhart += 1.0 / (NTC_NOMINAL_TEMP + 273.15);
  steinhart = 1.0 / steinhart;
  steinhart -= 273.15;
  return steinhart;
}

// ---------- Sampling (called continuously from loop) ----------

void sampleCurrent() {
  unsigned long nowUs = micros();
  if (nowUs - lastCurrentSampleUs < CURRENT_SAMPLE_INTERVAL_US) return;
  lastCurrentSampleUs = nowUs;

  int mv = cleanReadMv(CURRENT_PIN);
  currentSumMv += mv;
  currentCount++;
  if (mv < currentMinMv) currentMinMv = mv;
  if (mv > currentMaxMv) currentMaxMv = mv;
}

void sampleTaps() {
  unsigned long nowUs = micros();
  if (nowUs - lastTapSampleUs < TAP_SAMPLE_INTERVAL_US) return;
  lastTapSampleUs = nowUs;
  if (tapCount >= MAX_TAP_SAMPLES) return;

  for (int t = 0; t < 4; t++) {
    tapBuf[t][tapCount] = cleanReadMv(tapPins[t]);
  }
  tapCount++;
}

// ---------- Publishing (once per second) ----------

void publishReading() {
  latest.timestamp_ms = millis();

  float tapV[4];
  for (int t = 0; t < 4; t++) {
    tapV[t] = (trimmedMeanMv(tapBuf[t], tapCount) / 1000.0) / tapRatios[t];
  }
  latest.cell[0] = tapV[0];
  latest.cell[1] = tapV[1] - tapV[0];
  latest.cell[2] = tapV[2] - tapV[1];
  latest.cell[3] = tapV[3] - tapV[2];
  latest.vSamples = tapCount;
  tapCount = 0;

  if (currentCount > 0) {
    latest.current = mvToAmps(currentSumMv / currentCount);
    float a = mvToAmps(currentMinMv);
    float b = mvToAmps(currentMaxMv);
    latest.currentMin = min(a, b);
    latest.currentMax = max(a, b);
  } else {
    latest.current = 0;
    latest.currentMin = 0;
    latest.currentMax = 0;
  }
  latest.samples = currentCount;

  currentSumMv = 0;
  currentCount = 0;
  currentMinMv = 100000;
  currentMaxMv = -1;

  latest.temp1 = ntcToCelsius(TEMP1_PIN);
  latest.temp2 = ntcToCelsius(TEMP2_PIN);

  // One machine-readable line per second for logger_usb.py:
  // DATA,ms,c1,c2,c3,c4,current,current_min,current_max,samples,v_samples,temp1,temp2
  Serial.printf("DATA,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%u,%d,%.2f,%.2f\n",
                latest.timestamp_ms,
                latest.cell[0], latest.cell[1], latest.cell[2], latest.cell[3],
                latest.current, latest.currentMin, latest.currentMax,
                latest.samples, latest.vSamples, latest.temp1, latest.temp2);
}

void handleData() {
  String json = "{";
  json += "\"timestamp_ms\":" + String(latest.timestamp_ms) + ",";
  json += "\"cells\":[";
  for (int i = 0; i < 4; i++) {
    json += String(latest.cell[i], 4);
    if (i < 3) json += ",";
  }
  json += "],";
  json += "\"current\":" + String(latest.current, 4) + ",";
  json += "\"current_min\":" + String(latest.currentMin, 3) + ",";
  json += "\"current_max\":" + String(latest.currentMax, 3) + ",";
  json += "\"samples\":" + String(latest.samples) + ",";
  json += "\"v_samples\":" + String(latest.vSamples) + ",";
  json += "\"temp1\":" + String(latest.temp1, 2) + ",";
  json += "\"temp2\":" + String(latest.temp2, 2);
  json += "}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  if (USE_WIFI) {
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    WiFi.setTxPower(WIFI_POWER_2dBm);
    Serial.print("Access point started. Join WiFi '");
    Serial.print(AP_SSID);
    Serial.print("' and open http://");
    Serial.print(WiFi.softAPIP());
    Serial.println("/data");
    server.on("/data", handleData);
    server.begin();
  } else {
    WiFi.mode(WIFI_OFF);               // radio fully off: nothing to disturb the ADC
    Serial.println("WiFi OFF - logging over USB (run logger_usb.py)");
  }

  windowStartMs = millis();
  lastCurrentSampleUs = micros();
  lastTapSampleUs = micros();
}

void loop() {
  if (USE_WIFI) server.handleClient();
  sampleCurrent();
  sampleTaps();

  if (millis() - windowStartMs >= REPORT_INTERVAL_MS) {
    windowStartMs = millis();
    publishReading();
  }
}
