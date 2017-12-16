#include "Encoder.h"

Encoder::Encoder(const int channel_A_pin, const int channel_B_pin, const int CPR) {
  pinMode(channel_A_pin,INPUT);
  digitalWrite(channel_A_pin, HIGH);
  pinMode(channel_B_pin,INPUT);
  digitalWrite(channel_B_pin, HIGH);

  this->counter = 0;
  this->CPR = CPR;
  this->PIN_ENCODER_A = channel_A_pin;
  this->PIN_ENCODER_B = channel_B_pin;

  // Initialize Variables
  this->direction = CLOCKWISE;
  this->speed = 0;
  this->angle = 0;
}

void Encoder::channel_B_change() {
  this->encoder_A = digitalRead(this->PIN_ENCODER_A);
  this->encoder_B = digitalRead(this->PIN_ENCODER_B);

  // Determine direction
  if (this->encoder_B == HIGH && this->encoder_A == LOW) {
    this->direction = CLOCKWISE;
  } else if (this->encoder_B == LOW && this->encoder_A == HIGH) {
    this->direction = CLOCKWISE;
  } else {
    this->direction = COUNTER_CLOCKWISE;
  }
  
  // Update speed and angle (independent of A/B change)
  this->update_speed();
  this->update_angle();
}

void Encoder::channel_A_change() {

  this->encoder_A = digitalRead(this->PIN_ENCODER_A);
  this->encoder_B = digitalRead(this->PIN_ENCODER_B);

  // Determine direction
  if (this->encoder_A == HIGH && this->encoder_B == HIGH) {
    this->direction = CLOCKWISE;
  } else if (this->encoder_A == LOW && this->encoder_B == LOW) {
    this->direction = CLOCKWISE;
  } else {
    this->direction = COUNTER_CLOCKWISE;
  }
  
  // Update speed and angle (independent of A/B change)
  this->update_speed();
  this->update_angle();
}

void Encoder::update_speed() {
  float delta = (float)(micros() - this->last_channel_change_time);
  this->speed = 1.0 / (float)(this->CPR*2) * 1000000.0 / (delta);
  this->last_channel_change_time = micros();
}

void Encoder::update_angle() {
  this->counter += (this->direction == CLOCKWISE ? 1 : -1);
  this->counter += (this->counter < 0 ? this->CPR * 2 : 0);
  this->counter = this->counter % (this->CPR * 2);
  this->angle = this->counter / (float)(this->CPR*2) * 2 * M_PI;
}

// Getters
float Encoder::get_speed() {
  // Check for stopped encoder (i.e. lack of input about angle)
  // Calculate what speed would be if an ISR fired.
  float delta = (float)(micros() - this->last_channel_change_time);
  float check_speed = 1.0 / (float)(this->CPR*2) * 1000000.0 / (delta);
  if (check_speed < 0.01) {
    // Motor must be stopped
    this->speed = 0;
  }

  return this->speed;
}

DIR Encoder::get_direction() {
  return this->direction;
}

float Encoder::get_angle() {
  return this->angle;
}