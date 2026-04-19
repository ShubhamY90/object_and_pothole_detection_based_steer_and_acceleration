#include "esp_camera.h"
#include <WiFi.h>
#include "board_config.h"
#include "esp_http_server.h"

// ================= WIFI =================
const char* ssid = "mayank";
const char* password = "hello_world3";

// ================= LED PINS =================
#define PIN_UP    4
#define PIN_DOWN  2
#define PIN_LEFT  14
#define PIN_RIGHT 15

// ================= GLOBAL LED STATE =================
volatile int g_up    = 0;
volatile int g_down  = 0;
volatile int g_left  = 0;
volatile int g_right = 0;

// ================= HTTP SERVER =================
httpd_handle_t server = NULL;

// ================= LED HANDLER =================
static esp_err_t led_handler(httpd_req_t *req) {
    char buf[100];

    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        char param[10];
        if (httpd_query_key_value(buf, "up",    param, sizeof(param)) == ESP_OK) g_up    = atoi(param);
        if (httpd_query_key_value(buf, "down",  param, sizeof(param)) == ESP_OK) g_down  = atoi(param);
        if (httpd_query_key_value(buf, "left",  param, sizeof(param)) == ESP_OK) g_left  = atoi(param);
        if (httpd_query_key_value(buf, "right", param, sizeof(param)) == ESP_OK) g_right = atoi(param);
    }

    Serial.printf("CMD: U=%d D=%d L=%d R=%d\n", g_up, g_down, g_left, g_right);
    httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

// ================= STREAM HANDLER =================
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res   = ESP_OK;

    // Tell client this is a forever-running MJPEG stream
    res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
    if (res != ESP_OK) return res;

    // Disable send timeout so long streams don't get killed by the server
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    while (true) {

        // ── 1. Apply LED state ──────────────────────────────────────────
        digitalWrite(PIN_UP,    g_up    ? HIGH : LOW);
        digitalWrite(PIN_DOWN,  g_down  ? HIGH : LOW);
        digitalWrite(PIN_LEFT,  g_left  ? HIGH : LOW);
        digitalWrite(PIN_RIGHT, g_right ? HIGH : LOW);

        // ── 2. Grab frame ───────────────────────────────────────────────
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Frame grab failed — skipping");
            vTaskDelay(10 / portTICK_PERIOD_MS);
            continue;   // ← skip this frame, DON'T break the stream
        }

        // ── 3. Send MJPEG frame header ──────────────────────────────────
        char part_buf[128];
        size_t hlen = snprintf(part_buf, sizeof(part_buf),
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %u\r\n\r\n",
            (unsigned)fb->len);

        res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res != ESP_OK) {
            esp_camera_fb_return(fb);   // ← always free before exit
            Serial.println("Client disconnected (header send failed)");
            return res;
        }

        // ── 4. Send JPEG data ───────────────────────────────────────────
        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        if (res != ESP_OK) {
            esp_camera_fb_return(fb);   // ← always free before exit
            Serial.println("Client disconnected (data send failed)");
            return res;
        }

        // ── 5. Send frame boundary ──────────────────────────────────────
        res = httpd_resp_send_chunk(req, "\r\n", 2);

        // ── 6. Free frame buffer EVERY time ────────────────────────────
        esp_camera_fb_return(fb);
        fb = NULL;  // safety — prevent double-free if loop restarts

        if (res != ESP_OK) {
            Serial.println("Client disconnected (boundary send failed)");
            return res;
        }

        // ── 7. Yield — lets LED handler get CPU time ────────────────────
        vTaskDelay(30 / portTICK_PERIOD_MS); // ~30fps cap, tune as needed
    }

    return ESP_OK;
}

// ================= START SERVER =================
void startServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port      = 80;
    config.max_uri_handlers = 8;

    // ✅ Increase these to prevent the 30-request timeout kill
    config.recv_wait_timeout  = 10;
    config.send_wait_timeout  = 10;

    if (httpd_start(&server, &config) != ESP_OK) {
        Serial.println("HTTP server start failed!");
        return;
    }

    httpd_uri_t stream_uri = {
        .uri      = "/stream",
        .method   = HTTP_GET,
        .handler  = stream_handler,
        .user_ctx = NULL
    };
    httpd_register_uri_handler(server, &stream_uri);

    httpd_uri_t led_uri = {
        .uri      = "/led",
        .method   = HTTP_GET,
        .handler  = led_handler,
        .user_ctx = NULL
    };
    httpd_register_uri_handler(server, &led_uri);

    Serial.println("Server started");
}

// ================= SETUP =================
void setup() {
    Serial.begin(115200);

    // LED pins
    pinMode(PIN_UP,    OUTPUT);
    pinMode(PIN_DOWN,  OUTPUT);
    pinMode(PIN_LEFT,  OUTPUT);
    pinMode(PIN_RIGHT, OUTPUT);
    digitalWrite(PIN_UP,    LOW);
    digitalWrite(PIN_DOWN,  LOW);
    digitalWrite(PIN_LEFT,  LOW);
    digitalWrite(PIN_RIGHT, LOW);

    // Camera config
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode    = CAMERA_GRAB_LATEST;   // ✅ always get freshest frame

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 10;
        config.fb_count     = 2;
        config.fb_location  = CAMERA_FB_IN_PSRAM; // ✅ was missing — caused buffer exhaustion
    } else {
        config.frame_size   = FRAMESIZE_QQVGA;
        config.jpeg_quality = 12;
        config.fb_count     = 1;
        config.fb_location  = CAMERA_FB_IN_DRAM;
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed");
        return;
    }

    WiFi.begin(ssid, password);
    Serial.print("Connecting");
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    Serial.println("\nConnected!");
    Serial.print("Stream: http://"); Serial.print(WiFi.localIP()); Serial.println("/stream");
    Serial.print("LED:    http://"); Serial.print(WiFi.localIP()); Serial.println("/led?up=1");

    startServer();
}

// ================= LOOP =================
void loop() {
    delay(1);
}
