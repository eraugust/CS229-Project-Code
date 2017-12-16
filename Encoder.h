#ifndef Encoder_h
#define Encoder_h

#include <Arduino.h>

typedef boolean DIR;

#ifndef CLOCKWISE
#define CLOCKWISE true
#endif
#define COUNTER_CLOCKWISE (!CLOCKWISE)

class Encoder {

  private:
    // Setup variables
    int CPR;
    int PIN_ENCODER_A;
    int PIN_ENCODER_B;
    // State variables
    volatile unsigned long last_channel_change_time = micros();
    volatile int encoder_A = HIGH;
    volatile int encoder_B = HIGH;
    // Second order state variables
    volatile int counter;
    volatile float speed;
    volatile DIR direction;
    volatile float angle;
    // Private functions
    void update_angle();
    void update_speed();
  
  public:
    // Constructor
    Encoder(const int channel_A_pin, const int channel_B_pin, const int CPR);
    // Call theses on ISRs
    void channel_A_change();
    void channel_B_change();
    // Getters
    float get_speed();
    DIR get_direction();
    float get_angle();
};

#endif
