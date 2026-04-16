#include <WiFi.h>
#include <WebServer.h>

const char* SSID     = "mayank";
const char* PASSWORD = "hello_world3";

// Your exact physical wiring
const int PIN_UP    = 4;    // Forward
const int PIN_DOWN  = 2;    // Reverse
const int PIN_LEFT  = 14;   // Left
const int PIN_RIGHT = 15;   // Right

WebServer server(80);

void handleLed() {
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

void setup() {
    Serial.begin(115200);

    // Setup LEDs
    pinMode(PIN_UP,    OUTPUT);
    pinMode(PIN_DOWN,  OUTPUT);
    pinMode(PIN_LEFT,  OUTPUT);
    pinMode(PIN_RIGHT, OUTPUT);

    // Turn all LEDs off at start
    digitalWrite(PIN_UP,    LOW);
    digitalWrite(PIN_DOWN,  LOW);
    digitalWrite(PIN_LEFT,  LOW);
    digitalWrite(PIN_RIGHT, LOW);

    // Connect WiFi
    WiFi.begin(SSID, PASSWORD);
    Serial.print("\nConnecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("\nWiFi connected!");
    Serial.print("Camera Ready! Use 'http://");
    Serial.print(WiFi.localIP());
    Serial.println("' to connect");

    server.on("/led", HTTP_GET, handleLed);
    server.begin();
}

void loop() {
    server.handleClient();
}
