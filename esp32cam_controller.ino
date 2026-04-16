/*
 * esp32cam_controller.ino
 *
 * Handles HTTP GET /led with query params:
 *   g, r       — PWM 0-255 for green (GPIO18) and red (GPIO19) LEDs
 *   up, down,
 *   left, right — digital 0/1 for GPIO 12, 13, 14, 15
 *
 * Flash to your ESP32-CAM with Arduino IDE or PlatformIO.
 * Board: "AI Thinker ESP32-CAM"
 */

#include <WiFi.h>
#include <WebServer.h>

// ── WiFi credentials ─────────────────────────────────────────────────────────
const char* SSID     = "YOUR_SSID";
const char* PASSWORD = "YOUR_PASSWORD";

// ── Pin definitions ───────────────────────────────────────────────────────────
const int PIN_GREEN = 18;   // PWM — forward speed (green LED)
const int PIN_RED   = 19;   // PWM — turning / reversing (red LED)
const int PIN_UP    = 12;   // digital — forward command
const int PIN_DOWN  = 13;   // digital — reverse command
const int PIN_LEFT  = 14;   // digital — left-turn command
const int PIN_RIGHT = 15;   // digital — right-turn command

// ── PWM channels (ESP32 ledc API) ────────────────────────────────────────────
const int CH_GREEN  = 0;
const int CH_RED    = 1;
const int PWM_FREQ  = 5000;   // Hz
const int PWM_BITS  = 8;      // 0-255

WebServer server(80);

// ── Handler ───────────────────────────────────────────────────────────────────
void handleLed() {
    // PWM channels
    int g = server.hasArg("g") ? server.arg("g").toInt() : 0;
    int r = server.hasArg("r") ? server.arg("r").toInt() : 0;
    g = constrain(g, 0, 255);
    r = constrain(r, 0, 255);

    ledcWrite(CH_GREEN, g);
    ledcWrite(CH_RED,   r);

    // Direction pins
    int up    = server.hasArg("up")    ? server.arg("up").toInt()    : 0;
    int down  = server.hasArg("down")  ? server.arg("down").toInt()  : 0;
    int left  = server.hasArg("left")  ? server.arg("left").toInt()  : 0;
    int right = server.hasArg("right") ? server.arg("right").toInt() : 0;

    digitalWrite(PIN_UP,    up    ? HIGH : LOW);
    digitalWrite(PIN_DOWN,  down  ? HIGH : LOW);
    digitalWrite(PIN_LEFT,  left  ? HIGH : LOW);
    digitalWrite(PIN_RIGHT, right ? HIGH : LOW);

    server.send(200, "text/plain", "OK");
}

void handleNotFound() {
    server.send(404, "text/plain", "Not found");
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    // Configure PWM
    ledcSetup(CH_GREEN, PWM_FREQ, PWM_BITS);
    ledcSetup(CH_RED,   PWM_FREQ, PWM_BITS);
    ledcAttachPin(PIN_GREEN, CH_GREEN);
    ledcAttachPin(PIN_RED,   CH_RED);

    // Configure direction pins
    pinMode(PIN_UP,    OUTPUT);
    pinMode(PIN_DOWN,  OUTPUT);
    pinMode(PIN_LEFT,  OUTPUT);
    pinMode(PIN_RIGHT, OUTPUT);

    // All off at start
    ledcWrite(CH_GREEN, 0);
    ledcWrite(CH_RED,   0);
    digitalWrite(PIN_UP,    LOW);
    digitalWrite(PIN_DOWN,  LOW);
    digitalWrite(PIN_LEFT,  LOW);
    digitalWrite(PIN_RIGHT, LOW);

    // WiFi
    WiFi.begin(SSID, PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());

    // Routes
    server.on("/led", HTTP_GET, handleLed);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("HTTP server started");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    server.handleClient();
}