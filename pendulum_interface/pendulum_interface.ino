#include <Encoder.h>
#include <Motor.h>
#include <Wire.h>

#define SLAVE_ADDRESS 0x04

// CONFIGURATION
const int MOTOR_ENCODER_A = 2;
const int MOTOR_ENCODER_B = 4;
const int MOTOR_ENCODER_CPR = 600;
const int MOTOR_PWM_PIN = 3;
const int MOTOR_BRAKE_PIN = 9;
const int MOTOR_DIRECTION_PIN = 12;

// Initialize the attached device
Motor motor(MOTOR_BRAKE_PIN, MOTOR_DIRECTION_PIN, MOTOR_PWM_PIN);
Encoder motor_encoder(MOTOR_ENCODER_A, MOTOR_ENCODER_B, MOTOR_ENCODER_CPR);

char * system_state_buffer = malloc(32 * sizeof(char));
int state_data = 0;

void setup() {
  Serial.begin(9600);
    
  motor.set_direction(COUNTER_CLOCKWISE);
  motor.set_speed(0);
  
  // Initialize i2c as slave
  Wire.begin(SLAVE_ADDRESS);
  Wire.setClock(100000);
  
  // Define callbacks for i2c communication
  Wire.onReceive(receiveData);
  Wire.onRequest(sendData);
  
  // Attach encoder interrupts
  attachInterrupt(digitalPinToInterrupt(MOTOR_ENCODER_A), motor_encoder_channel_A_change, CHANGE);
  attachInterrupt(digitalPinToInterrupt(MOTOR_ENCODER_B), motor_encoder_channel_B_change, CHANGE);
}

// Loop Function
void loop() {
  refresh_system_state_buffer();
}

// ISR static functions
void motor_encoder_channel_A_change() { motor_encoder.channel_A_change(); }
void motor_encoder_channel_B_change() { motor_encoder.channel_B_change(); }

// Raspberry pi is sending data to the arduino.
// Parse the data to get motor speed and direction
void receiveData(int byte_count) {
  if (Wire.available()) {
    int received = Wire.read();
    // Did I receive a motor speed command? (0 -> 200)
    float motor_speed = 0;
    if (received <= 100) {
      motor_speed = received / 100.0;
      motor.set_direction(CLOCKWISE);
      motor.set_speed(motor_speed);
    } else if (received <= 200) {
      motor_speed = -(received-100) / 100.0;
      motor.set_direction(COUNTER_CLOCKWISE);
      motor.set_speed(-motor_speed);
    }
    
    // Did I receive an encoder query command? (201, 202, 203)
    if (received == 201) { state_data = motor_encoder.speed * 100;}
    else if (received == 202) {state_data = motor_encoder.direction;}
    else if (received == 203) {state_data = motor_encoder.angle / (2 * 3.14159265) * 100;}
  }
}

// Raspberry pi is requesting device state
void sendData() {
  Wire.write(state_data);
  // Wire.write(system_state_buffer);
}

void refresh_system_state_buffer() {
  String motor_speed = String(motor_encoder.speed);
  String motor_direction = String(motor_encoder.direction);
  String motor_angle = String(motor_encoder.angle);
  // String pendulum_speed = String(pendulum_encoder.speed);
  // String pendulum_direction = String(pendulum_encoder.direction);
  // String pendulum_angle = String(pendulum_encoder.angle);
  String system_state = "["+motor_speed+","+motor_direction+","+motor_angle+/*","+pendulum_speed+","+pendulum_direction+","+pendulum_angle+*/"]";
  system_state.substring(0,10).toCharArray(system_state_buffer, 32);
}