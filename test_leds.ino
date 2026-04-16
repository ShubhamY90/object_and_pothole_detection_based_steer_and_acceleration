// test_leds.ino
// Pure Hardware Test - No Wi-Fi, No Python.
// Flashes each LED one by one in a loop to verify wiring.

const int PIN_UP    = 4;   // Forward
const int PIN_DOWN  = 2;   // Reverse 
const int PIN_LEFT  = 14;  // Left
const int PIN_RIGHT = 15;  // Right

void setup() {
    Serial.begin(115200);
    
    pinMode(PIN_UP,    OUTPUT);
    pinMode(PIN_DOWN,  OUTPUT);
    pinMode(PIN_LEFT,  OUTPUT);
    pinMode(PIN_RIGHT, OUTPUT);
}

void loop() {
    Serial.println("Testing PIN 4 (Forward)");
    digitalWrite(PIN_UP, HIGH);
    delay(1000);
    digitalWrite(PIN_UP, LOW);
    
    Serial.println("Testing PIN 2 (Reverse)");
    digitalWrite(PIN_DOWN, HIGH);
    delay(1000);
    digitalWrite(PIN_DOWN, LOW);
    
    Serial.println("Testing PIN 14 (Left)");
    digitalWrite(PIN_LEFT, HIGH);
    delay(1000);
    digitalWrite(PIN_LEFT, LOW);
    
    Serial.println("Testing PIN 15 (Right)");
    digitalWrite(PIN_RIGHT, HIGH);
    delay(1000);
    digitalWrite(PIN_RIGHT, LOW);
    
    Serial.println("====================");
    delay(500);
}
