/*
  Plant electrical signal acquisition with an AD8232.

  Supported boards:
  - Arduino UNO and classic ATmega328P Nano: 10-bit ADC, nominal 5 V DEFAULT
  - Arduino Nano ESP32: 12-bit ADC, calibrated millivolt conversion

  This sketch is for plant measurements only. It emits one CSV row at 100 Hz:
  millis,raw_adc,voltage,lo_plus,lo_minus

  See README.md before enabling the AVR-only external AREF option below.
*/

#include <Arduino.h>

const uint8_t SIGNAL_PIN = A0;

#if defined(ARDUINO_NANO_ESP32)
// Labels keep the physical Nano pins stable in either Nano ESP32 pin mode.
const uint8_t LO_PLUS_PIN = D10;
const uint8_t LO_MINUS_PIN = D11;
#else
const uint8_t LO_PLUS_PIN = 10;
const uint8_t LO_MINUS_PIN = 11;
#endif

const unsigned long SERIAL_BAUD = 115200;
const unsigned long SAMPLE_RATE_HZ = 100;
const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

#if !defined(ARDUINO_ARCH_ESP32)
/*
  AVR-only option. Leave this false for the five-wire setup in README.md.

  To use a 3.3 V ADC reference:
  1. Disconnect power.
  2. Add a wire from Arduino 3.3V to Arduino AREF.
  3. Set USE_EXTERNAL_3V3_AREF to true.

  Never connect 3.3 V to AREF while this remains false.
*/
const bool USE_EXTERNAL_3V3_AREF = false;
const float ADC_REFERENCE_VOLTAGE =
    USE_EXTERNAL_3V3_AREF ? 3.3F : 5.0F;
const float ADC_COUNTS = 1024.0F;
#endif

unsigned long nextSampleMicros = 0;

void configureAdc() {
#if defined(ARDUINO_ARCH_ESP32)
  // Nano ESP32 uses a 12-bit ADC and does not provide analogReference().
  analogReadResolution(12);
#else
  if (USE_EXTERNAL_3V3_AREF) {
    analogReference(EXTERNAL);
  } else {
    analogReference(DEFAULT);
  }
#endif
}

float readVoltage(const int rawAdc) {
#if defined(ARDUINO_ARCH_ESP32)
  /*
    Arduino-ESP32 provides a calibrated millivolt conversion. It performs a
    second ADC conversion immediately after rawAdc, which is adequate at 100 Hz.
  */
  (void)rawAdc;
  return analogReadMilliVolts(SIGNAL_PIN) / 1000.0F;
#else
  return rawAdc * ADC_REFERENCE_VOLTAGE / ADC_COUNTS;
#endif
}

void setup() {
  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);

  Serial.begin(SERIAL_BAUD);
  configureAdc();

  // Discard the first reading after selecting the analog reference.
  delay(10);
  (void)analogRead(SIGNAL_PIN);

  Serial.println(F("millis,raw_adc,voltage,lo_plus,lo_minus"));
  nextSampleMicros = micros();
}

void loop() {
  const unsigned long now = micros();

  // Signed subtraction keeps the scheduler correct across micros() rollover.
  if ((long)(now - nextSampleMicros) < 0) {
    return;
  }

  nextSampleMicros += SAMPLE_PERIOD_US;

  // Avoid a burst of stale catch-up samples if Serial was blocked briefly.
  if ((long)(now - nextSampleMicros) >= (long)SAMPLE_PERIOD_US) {
    nextSampleMicros = now + SAMPLE_PERIOD_US;
  }

  const int rawAdc = analogRead(SIGNAL_PIN);
  const float voltage = readVoltage(rawAdc);
  const int loPlus = digitalRead(LO_PLUS_PIN);
  const int loMinus = digitalRead(LO_MINUS_PIN);

  Serial.print(millis());
  Serial.print(',');
  Serial.print(rawAdc);
  Serial.print(',');
  Serial.print(voltage, 6);
  Serial.print(',');
  Serial.print(loPlus);
  Serial.print(',');
  Serial.println(loMinus);
}
