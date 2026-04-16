#include "esp_camera.h"
#include <WiFi.h>

// ===========================
// Select camera model in board_config.h
// ===========================
#include "board_config.h"
#include <WebServer.h>

WebServer server(80);

const char *ssid = "mayank";
const char *password = "hello_world3";

// ✅ Safer pins for ESP32-CAM
#define PIN_UP    4    // OK
#define PIN_DOWN  2    // OK (but careful)
#define PIN_LEFT  14   // OK
#define PIN_RIGHT 15   // OK

void startCameraServer();
void setupLedFlash();

// ================= LED HANDLER =================
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

// ================= FREERTOS THREAD (CORE 0) =================
TaskHandle_t LedTaskHandle;

void ledTaskCode(void * pvParameters) {
  Serial.print("LED Wi-Fi Handler running natively on Core: ");
  Serial.println(xPortGetCoreID());

  for(;;) {
    server.handleClient();
    delay(10); // Very important! Yields to the FreeRTOS Wi-Fi watchdog
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // ================= CAMERA CONFIG (UNCHANGED) =================
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // ================= LED PINS =================
  pinMode(PIN_UP, OUTPUT);
  pinMode(PIN_DOWN, OUTPUT);
  pinMode(PIN_LEFT, OUTPUT);
  pinMode(PIN_RIGHT, OUTPUT);

  digitalWrite(PIN_UP, LOW);
  digitalWrite(PIN_DOWN, LOW);
  digitalWrite(PIN_LEFT, LOW);
  digitalWrite(PIN_RIGHT, LOW);

  // ================= CAMERA INIT =================
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();

  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }

  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif

  // ================= WIFI =================
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");

  // ================= START CAMERA SERVER =================
  startCameraServer();

  // ================= ADD LED ROUTE =================
  server.on("/led", HTTP_GET, handleLed);
  server.begin();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");

  Serial.println("LED control ready: /led?up=1");

  // ================= LAUNCH THREAD ON CORE 0 =================
  xTaskCreatePinnedToCore(
      ledTaskCode,   // Task function
      "LEDTask",     // Task name
      8192,          // Stack size in words
      NULL,          // Task input parameters
      1,             // Priority
      &LedTaskHandle,// Task handle
      0);            // Core 0 (Protocol Core)
}

// ================= MAIN LOOP (CORE 1) =================
void loop() {
  // Core 1 is now 100% completely empty and free! 
  // FreeRTOS handles the camera and the LED task completely in the background.
  delay(1000);
}