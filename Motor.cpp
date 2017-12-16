#include "Motor.h"

Motor::Motor(const int brake_pin, const int direction_pin, const int pwm_pin) {
  pinMode(brake_pin,OUTPUT);
  pinMode(direction_pin,OUTPUT);

  this->PIN_BRAKE = brake_pin;
  this->PIN_DIRECTION = direction_pin;
  this->PIN_PWM = pwm_pin;
}

void Motor::brake() {
  set_speed(0);
  digitalWrite(this->PIN_BRAKE, HIGH);
}

void Motor::release_brake() {
  digitalWrite(this->PIN_BRAKE, LOW);
}

void Motor::set_speed(float speed) {
  // clip input
  if (speed > 1) {
    speed = 1;
  } else if (speed < 0) {
    speed = 0;
  }
  if (speed > 0) {
    release_brake();
  }
  analogWrite(this->PIN_PWM, speed * 255.0);
}

void Motor::set_direction(boolean direction) {
  if (direction == CLOCKWISE) {
    digitalWrite(this->PIN_DIRECTION, HIGH);
  } else {
    digitalWrite(this->PIN_DIRECTION, LOW);
  }
}