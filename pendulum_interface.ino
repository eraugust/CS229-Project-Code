#include <Encoder.h>
#include <Motor.h>


// CONFIGURATION
const int MOTOR_ENCODER_A = 53;
const int MOTOR_ENCODER_B = 52;
const int MOTOR_ENCODER_CPR = 600;
const int PENDULUM_ENCODER_A = 51;
const int PENDULUM_ENCODER_B = 50;
const int PENDULUM_ENCODER_CPR = 400;
const int MOTOR_PWM_PIN = 3;
const int MOTOR_BRAKE_PIN = 9;
const int MOTOR_DIRECTION_PIN = 12;

//FUNCTIONS REQUESTEDOVER SERIAL LINK
const int UPDATE_MOTOR_FUNCTION = 0;
const int UPDATE_SMOOTHING_FACTOR_FUNCTION = 1;
const int BRAKE_FUNCTION = 2;
const int REQUEST_PARAMETERS_FUNCTION = 3;

// Hysteresis arrays for encoders
// Running Average
const int HYSTERESIS_LENGTH = 30;
float motor_speed_hysteresis[HYSTERESIS_LENGTH];
float pendulum_speed_hysteresis[HYSTERESIS_LENGTH];
// Exponential Smoothing
float SMOOTHING_FACTOR = 0.2;
float motor_last_speed = 0;
float pendulum_last_speed = 0;

// Initialize the attached devices
Motor motor(MOTOR_BRAKE_PIN, MOTOR_DIRECTION_PIN, MOTOR_PWM_PIN);
Encoder motor_encoder(MOTOR_ENCODER_A, MOTOR_ENCODER_B, MOTOR_ENCODER_CPR);
Encoder pendulum_encoder(PENDULUM_ENCODER_A, PENDULUM_ENCODER_B, PENDULUM_ENCODER_CPR);

String system_state = "";

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
    
  motor.set_direction(COUNTER_CLOCKWISE);
  motor.set_speed(0);
  
  // Attach encoder interrupts
  // Motor
  attachInterrupt(digitalPinToInterrupt(MOTOR_ENCODER_A), motor_encoder_channel_A_change, CHANGE);
  attachInterrupt(digitalPinToInterrupt(MOTOR_ENCODER_B), motor_encoder_channel_B_change, CHANGE);
  // Pendulum
  attachInterrupt(digitalPinToInterrupt(PENDULUM_ENCODER_A), pendulum_encoder_channel_A_change, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PENDULUM_ENCODER_B), pendulum_encoder_channel_B_change, CHANGE);
  
  // Reset encoder speed hystersis
  for(int i=0; i<HYSTERESIS_LENGTH; i++) {
    motor_speed_hysteresis[i] = 0.;
    pendulum_speed_hysteresis[i] = 0.;
  }
}

// Loop Function
void loop() {
  refresh_system_state_buffer();
}

// ISR static functions
// Motor
void motor_encoder_channel_A_change() { motor_encoder.channel_A_change(); }
void motor_encoder_channel_B_change() { motor_encoder.channel_B_change(); }
// Pendulum
void pendulum_encoder_channel_A_change() { pendulum_encoder.channel_A_change(); }
void pendulum_encoder_channel_B_change() { pendulum_encoder.channel_B_change(); }

void serialEvent() {
  // There are bytes in the Serial link buffer
  // Parse the serial data and change motor speed accordingly
  String commands = Serial.readStringUntil('X'); // One line of CSV, termination character is a 'X'
  Serial.readString(); // Remove the newline character
  
  commands.remove(commands.length()); // Remove the termination character
  // Last value identifies the requested function
  int requested_function = commands.substring(commands.lastIndexOf(',')+1).toInt();
  commands.remove(commands.lastIndexOf(','));
  
  // Other values depend on the type of function requested:
  if (requested_function == UPDATE_MOTOR_FUNCTION) {
    // One value: The new motor speed
    float motor_speed = commands.toFloat();
    // Set the motor speed
    if(motor_speed >= 0) {
      motor.set_direction(CLOCKWISE);
      motor.set_speed(motor_speed);
    } else {
      motor.set_direction(COUNTER_CLOCKWISE);
      motor.set_speed(-motor_speed);
    }
  } else if (requested_function == UPDATE_SMOOTHING_FACTOR_FUNCTION) {
    // One value: The new smoothing factor (for the exponential smoothing)
    SMOOTHING_FACTOR = commands.toFloat();
  } else if (requested_function == BRAKE_FUNCTION) {
    // Brake the motor (automatically sets the speed to zero)
    motor.brake();
  } else if (requested_function == REQUEST_PARAMETERS_FUNCTION) {
    // Write data back to the raspberry pi
    Serial.println(system_state);
    Serial.flush();
  }
}

void refresh_system_state_buffer() {
  // Calculate hysteresis for encoder speeds (0 is latest speed, HYSTERESIS_LENGTH-1 is oldest speed)
  for(int i=HYSTERESIS_LENGTH-1; i>0; i--) {
    motor_speed_hysteresis[i] = motor_speed_hysteresis[i-1];
    pendulum_speed_hysteresis[i] = pendulum_speed_hysteresis[i-1];
  }
  // Get raw speed
  motor_speed_hysteresis[0] = motor_encoder.get_speed();
  pendulum_speed_hysteresis[0] = pendulum_encoder.get_speed();
  float motor_speed = 0;
  float pendulum_speed = 0;
  for(int i=0; i<HYSTERESIS_LENGTH; i++) {
    motor_speed += motor_speed_hysteresis[i] / HYSTERESIS_LENGTH;
    pendulum_speed += pendulum_speed_hysteresis[i] / HYSTERESIS_LENGTH;
  }
  
  // Calculate exponential smoothing
  motor_speed = SMOOTHING_FACTOR * motor_encoder.get_speed() + (1 - SMOOTHING_FACTOR) * motor_last_speed;
  pendulum_speed = SMOOTHING_FACTOR * pendulum_encoder.get_speed() + (1 - SMOOTHING_FACTOR) * pendulum_last_speed;
  motor_last_speed = motor_speed;
  pendulum_last_speed = pendulum_speed;
  
  // Get the state of the encoders
  String motor_speed_string = String(motor_speed);
  String motor_direction = motor_encoder.get_direction() == CLOCKWISE ? "true" : "false";
  String motor_angle = String(motor_encoder.get_angle());
  String pendulum_speed_string = String(pendulum_speed);
  String pendulum_direction = pendulum_encoder.get_direction() == CLOCKWISE ? "true" : "false";
  String pendulum_angle = String(pendulum_encoder.get_angle());
  
  system_state = "{"
    "\"motor_speed\":"+motor_speed_string+","
    "\"motor_direction\":"+motor_direction+","
    "\"motor_angle\":"+motor_angle+","
    "\"pendulum_speed\":"+pendulum_speed_string+","
    "\"pendulum_direction\":"+pendulum_direction+","
    "\"pendulum_angle\":"+pendulum_angle+"}";
}