#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h> // Use the built-in ESP32 WebServer
#include <TinyGPS++.h>
#include <HardwareSerial.h>
#include <ArduinoJson.h> // For creating JSON responses

// --- WiFi Credentials ---
const char* ssid = "Big chungus";         // <<<--- REPLACE with your WiFi network name
const char* password = "paru1234"; // <<<--- REPLACE with your WiFi password

// --- GPS Setup ---
TinyGPSPlus gps;
HardwareSerial& gpsSerial = Serial2; // Use Serial2 (RX=16, TX=17)
#define RX_PIN 16
#define TX_PIN 17
static const uint32_t GPS_BAUD = 9600;

// --- Web Server Setup ---
WebServer server(80); // Create a web server object on port 80

// --- Global variables to store latest GPS data ---
// Initialize with invalid values or flags
double currentLat = 0.0;
double currentLng = 0.0;
bool locationIsValid = false;
uint32_t satellites = 0;
double altitude = 0.0;
double speedKmph = 0.0;
double hdop = 99.0; // High value indicates poor precision initially
String lastUpdateTime = "No Fix Yet";
uint32_t lastUpdateMillis = 0;

// --- Function Prototypes ---
void connectToWiFi();
void handleRoot();
void handleGpsData();
void handleNotFound();
void updateGpsData(); // Renamed from displayInfo

// --- Setup ---
void setup() {
  Serial.begin(115200);
  Serial.println("\nESP32 GPS WiFi Sender Starting...");

  // Start GPS Serial
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RX_PIN, TX_PIN);
  Serial.println("GPS Serial Initialized.");

  // Connect to WiFi
  connectToWiFi();

  // --- Web Server Handlers ---
  server.on("/", HTTP_GET, handleRoot);             // Handler for the root path
  server.on("/gps", HTTP_GET, handleGpsData);       // Handler for GPS data request
  server.onNotFound(handleNotFound);               // Handler for 404 Not Found

  // Start the server
  server.begin();
  Serial.println("HTTP server started");
  Serial.print("Connect to http://");
  Serial.println(WiFi.localIP());
  Serial.println("Use /gps endpoint to get data (e.g., http://" + WiFi.localIP().toString() + "/gps)");
  Serial.println("----------------------------------------");
}

// --- Main Loop ---
void loop() {
  // 1. Process GPS data
  while (gpsSerial.available() > 0) {
    if (gps.encode(gpsSerial.read())) {
      updateGpsData(); // Update global variables when a sentence is parsed
    }
  }

  // 2. Handle incoming web server requests
  server.handleClient();

  // Optional: Add a small delay to prevent watchdog timeout if loop is too tight
  // delay(1);
}

// --- Function Implementations ---

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts > 20) { // Timeout after ~10 seconds
        Serial.println("\nFailed to connect to WiFi. Restarting...");
        ESP.restart();
    }
  }

  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// Update global GPS variables if data is valid
void updateGpsData() {
  if (gps.location.isValid()) {
    currentLat = gps.location.lat();
    currentLng = gps.location.lng();
    locationIsValid = true;
    lastUpdateMillis = millis(); // Record time of last valid update
  } else {
    // Optional: Decide if you want to invalidate old data after some time
    // if (millis() - lastUpdateMillis > 30000) { // e.g., invalidate after 30s
    //   locationIsValid = false;
    // }
  }

  if (gps.satellites.isValid()) {
    satellites = gps.satellites.value();
  }
  if (gps.altitude.isValid()) {
    altitude = gps.altitude.meters();
  }
  if (gps.speed.isValid()) {
    speedKmph = gps.speed.kmph();
  }
   if (gps.hdop.isValid()) {
     hdop = gps.hdop.value() / 100.0; // HDOP value
  }

  if (gps.date.isValid() && gps.time.isValid()) {
      char dtBuffer[25];
      snprintf(dtBuffer, sizeof(dtBuffer), "%04d-%02d-%02dT%02d:%02d:%02dZ",
               gps.date.year(), gps.date.month(), gps.date.day(),
               gps.time.hour(), gps.time.minute(), gps.time.second());
      lastUpdateTime = String(dtBuffer);
  }

   // Minimal Serial print to show GPS is being read (optional)
   // Serial.print(".");
}


// Handler for GET requests on / (root)
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>ESP32 GPS Server</title></head><body>";
  html += "<h1>ESP32 GPS Data Server</h1>";
  html += "<p>Use the <a href=\"/gps\">/gps</a> endpoint to fetch the latest data in JSON format.</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

// Handler for GET requests on /gps
void handleGpsData() {
  // Use ArduinoJson V6 syntax
  StaticJsonDocument<300> doc; // Adjust size if needed

  doc["valid"] = locationIsValid;
  if (locationIsValid) {
    doc["latitude"] = currentLat;
    doc["longitude"] = currentLng;
    doc["timestamp_utc"] = lastUpdateTime;
    doc["satellites"] = satellites;
    doc["altitude_m"] = altitude;
    doc["speed_kmph"] = speedKmph;
    doc["hdop"] = hdop;
  } else {
     doc["message"] = "Waiting for valid GPS fix...";
  }

  String jsonResponse;
  serializeJson(doc, jsonResponse); // Serialize the JSON document to a String

  server.send(200, "application/json", jsonResponse); // Send JSON response
}

// Handler for 404 Not Found
void handleNotFound() {
  String message = "File Not Found\n\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET) ? "GET" : "POST";
  message += "\nArguments: ";
  message += server.args();
  message += "\n";
  for (uint8_t i = 0; i < server.args(); i++) {
    message += " " + server.argName(i) + ": " + server.arg(i) + "\n";
  }
  server.send(404, "text/plain", message);
}