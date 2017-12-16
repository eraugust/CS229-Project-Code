
#ifndef Motor_h
#define Motor_h

#include <Arduino.h>

typedef boolean DIR;

#ifndef CLOCKWISE
#define CLOCKWISE true
#endif
#define COUNTER_CLOCKWISE (!CLOCKWISE)


class Motor {

  private:
    // Setup variables
    int PIN_BRAKE;
    int PIN_DIRECTION;
    int PIN_PWM;
  
  public:
	void brake();
	void release_brake();
	void set_speed(float speed);
	void set_direction(DIR clockwise);
	Motor(const int brake_pin, const int direction_pin, const int pwm_pin);

};

#endif
