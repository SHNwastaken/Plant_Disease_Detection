#include <Servo.h> // Include the Servo library

// --- Pin Definitions ---
// L298N Motor Driver Pins
#define ENA 3  // Left Motor Speed (PWM) - MUST be a PWM pin
#define IN1 4  // Left Motor Direction 1
#define IN2 5  // Left Motor Direction 2
#define IN3 7  // Right Motor Direction 1
#define IN4 8 // Right Motor Direction 2
#define ENB 11  // Right Motor Speed (PWM) - MUST be a PWM pin

// HC-SR04 Ultrasonic Sensor Pins
#define TRIG_PIN 9
#define ECHO_PIN 10

// Servo Motor Pin
#define SERVO_PIN 6 // Choose an available digital pin (PWM capable preferred but not strictly necessary for basic write)

// --- Constants ---
#define OBSTACLE_THRESHOLD 20 // Distance in cm to stop and turn
#define MOTOR_SPEED 180       // Speed value (0-255) for forward/backward movement
#define TURN_SPEED 150        // Speed value (0-255) for turning
#define TURN_DURATION 700     // Milliseconds to turn (adjust as needed)
#define REVERSE_DURATION 800  // Milliseconds to reverse (adjust as needed)
#define SERVO_DETECT_ANGLE 45 // Angle (degrees) when obstacle detected
#define SERVO_DEFAULT_ANGLE 90 // Default angle (degrees) - usually straight ahead

// --- Global Objects ---
Servo myservo; // Create a servo object to control a servo

void setup() {
  Serial.begin(9600); // Initialize serial communication for debugging
  Serial.println("Rover Initializing...");

  // Set L298N control pins as OUTPUTs
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Set HC-SR04 pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Attach the servo on SERVO_PIN
  myservo.attach(SERVO_PIN);
  Serial.println("Servo Attached.");

  // Set servo to default position initially
  myservo.write(SERVO_DEFAULT_ANGLE);
  delay(500); // Give the servo time to reach the position

  // Ensure motors are stopped initially
  stopMotors();
  Serial.println("Ready.");
}

void loop() {
  long distance = getDistance(); // Get distance in cm

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Check if an obstacle is detected within the threshold
  if (distance <= OBSTACLE_THRESHOLD && distance > 0) { // distance > 0 check helps ignore sensor errors
    Serial.println("Obstacle Detected!");

    // 0. Turn Servo to Detect Angle
    Serial.println("Turning Servo...");
    myservo.write(SERVO_DETECT_ANGLE);
    delay(300); // Allow servo time to move

    // 1. Stop Motors
    stopMotors();
    Serial.println("Stopping...");
    delay(500); // Pause briefly

    // 2. Reverse Motors
    Serial.println("Reversing...");
    moveBackward(MOTOR_SPEED);
    delay(REVERSE_DURATION); // Reverse for a set duration

    // 3. Stop before turning
    stopMotors();
    delay(300);

    // 4. Turn Robot (e.g., turn Right)
    Serial.println("Turning Right...");
    turnRight(TURN_SPEED);
    delay(TURN_DURATION); // Turn for a set duration

    // 5. Stop after turning
    stopMotors();
    Serial.println("Turn Complete.");
    delay(300);

    // 6. Return Servo to Default Position
    Serial.println("Returning Servo to Default...");
    myservo.write(SERVO_DEFAULT_ANGLE);
    delay(500); // Allow servo time to return & pause before moving

    Serial.println("Resuming forward...");

  } else {
    // No obstacle, move forward
    moveForward(MOTOR_SPEED);
    // Ensure servo is in default position if it wasn't already
    // (Optional - uncomment if you notice the servo drifts or gets stuck)
    // if (myservo.read() != SERVO_DEFAULT_ANGLE) {
    //   myservo.write(SERVO_DEFAULT_ANGLE);
    //   delay(20); // Small delay for servo adjustment
    // }
  }

  delay(100); // Short delay between sensor readings/actions
}

// --- Motor Control Functions ---

void moveForward(int speed) {
  // Left Motor Forward
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, speed);

  // Right Motor Forward
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, speed);
}

void moveBackward(int speed) {
  // Left Motor Backward
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, speed);

  // Right Motor Backward
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENB, speed);
}

void turnRight(int speed) {
  // Left Motor Forward
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, speed);

  // Right Motor Backward
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  analogWrite(ENB, speed);
}

void turnLeft(int speed) {
  // Left Motor Backward
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, speed);

  // Right Motor Forward
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, speed);
}

void stopMotors() {
  // Stop Left Motor
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0); // Set speed to 0

  // Stop Right Motor
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0); // Set speed to 0
}

// --- HC-SR04 Sensor Function ---

long getDistance() {
  // Clears the trigPin condition
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  // pulseIn() will wait for the pin to go HIGH, start timing, then wait for the pin to go LOW and stop timing.
  // It returns the length of the pulse in microseconds, or 0 if no pulse started within the timeout (1 second).
  long duration = pulseIn(ECHO_PIN, HIGH);

  // Calculating the distance
  // Speed of sound wave divided by 2 (go and back)
  // Speed of sound = 343 m/s = 0.0343 cm/us
  long distance = duration * 0.0343 / 2;

  return distance;
}