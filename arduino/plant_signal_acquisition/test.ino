void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Setup for leads-off detection
  pinMode(10, INPUT); // LO+
  pinMode(11, INPUT); // LO-
}

void loop() {
  // If LO+ or LO- is HIGH, electrode contact is bad
  if ((digitalRead(10) == 1) || (digitalRead(11) == 1)) {
    Serial.println("!");
  } 
  else {
    // Read signal from AD8232 OUTPUT connected to A0
    Serial.println(analogRead(A0));
  }

  // Small delay to avoid serial saturation
  delay(1);
}