import serial
import time
import json
import picamera
import picamera.array
import numpy as np
from scipy.misc import imshow
from scipy.special import expit
import sys

#  CONSTANTS
# Motor/Pendulum Hardware
CLOCKWISE = True
COUNTER_CLOCKWISE = not CLOCKWISE
MOTOR_ANGLE_SPREAD_LIMIT = 3*np.pi/2#  Must be between 0 and 2PI
MINIMUM_ABS_MOTOR_SPEED = 0.20      #  Unloaded is 0.15.  Might be more with pendulum attached
SMOOTHING_FACTOR = 0.2              #  For the exponential smoothing of the encoder speeds.
ARDUINO_EXECUTE_TIME = 0.002        #  Time delay so the arduino can execute the code.
MAXIMUM_ABS_MOTOR_SPEED = 0.2       #  Motor speed clipped before sending to arduino
RESET_SLEEP_TIME = 2                #  Sleep time so the pendulum can settle
CLOCKWISE = True
COUNTER_CLOCKWISE = not CLOCKWISE

#  Camera
MAXIMUM_MOL_FREQUENCY = 20          #  (Same as framerate for camera) MOLs are forced to wait if they finish too quickly.
IMAGE_WIDTH = 32                    #  Captured frame will be downsized to this (Must be divisible by 32)
IMAGE_HEIGHT = 32                   #  Captured frame will be downsized to this (Must be divisible by 16)
CAMERA_SENSOR_MODE = 4              #  Recommend 4 (full field of view, max 40fps) or 6 (smaller fieldof view, 41-90fps)
EXPOSURE_MODE = 'sports'            #  Recommend 'sports' for the faster shutter speed
RECORDING_TIME = 30                 #  Amount of time that camera is recording

# Machine Learning
DISCOUNT_FACTOR = 0.99              #  This is gamma, the discount learning factor
TOLERANCE = 0.01                    #  This is the threshold where we consider value iteration to have converged
NUMBER_MOTOR_SPEEDS = 3             #  Fidelity of motor speeds to be sent to motor
NUMBER_MOTOR_ANGLES = 6             #  Fidelity of motor angles
NUMBER_PENDULUM_SPEEDS = 6          #  Fidelity of pendulum speeds
PENDULUM_SPEED_LOW_HIGH_BISTABLE = 0.1  #  Breakpoint value between high and low speed pendulum
NUMBER_PENDULUM_ANGLES = 16         #  Fidelity of pendulum angles
NUMBER_STATES = NUMBER_MOTOR_SPEEDS * NUMBER_MOTOR_ANGLES * NUMBER_PENDULUM_SPEEDS * NUMBER_PENDULUM_ANGLES
ANNEALING_DURATION = RECORDING_TIME*MAXIMUM_MOL_FREQUENCY/50  #  Amount of time epsilon is annealed from 1.0 to 0.1 (a fraction of the frames)
NUMBER_OF_CONVERGENCES_TO_SAVE = 2  #  Save the parameters to file if the algorithm seems to be converging

def main():
    #  Ensure the power supply is disconnected
    input("Check the power supply is off.  Press enter to continue.")
    #  Connect to the arduino.  It will automatically reset.
    with serial.Serial('/dev/ttyACM0', baudrate=115200) as due_serial:
        time.sleep(2)
        print("Arduino initialized, turn on power supply.")
        #  Training loop.  Each loop is an episode where the balancer can swing up the pendulum.
        #parameters = Pendulum_Balancer.parameters_from_file()
        parameters = Pendulum_Balancer.empty_parameters()
        #  Setup the results file for later analysis
        Pendulum_Balancer.setup_iteration_save_file()
        while True:
            #  Initialize pendulum balancer
            balancer = Pendulum_Balancer(due_serial)
            balancer.copy_parameters(parameters)
            #  Connect to the camera
            with picamera.PiCamera(sensor_mode=CAMERA_SENSOR_MODE, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT), framerate=MAXIMUM_MOL_FREQUENCY) as camera:
                #  Warm up the camera (set gain/white balance, etc.
                time.sleep(2)
                camera.exposure_mode = EXPOSURE_MODE
                #  Start the preview for debugging
                #camera.start_preview()
                #  Start the timer on the balancer, for later analysis of FPS
                balancer.loop_time_start = time.time()
                #  Record
                camera.start_recording(balancer, 'yuv')
                camera.wait_recording(RECORDING_TIME)
                camera.stop_recording()
            #  Update the V and perform value iteration
            balancer.update_MDP_model()
            balancer.execute_value_iteration()
            #  Reset the motor position
            balancer.reset_motor()
            parameters = balancer.parameters
            #  Save the results to file
            balancer.save_iteration_results()
            #  Save the parameters if the algirthm is close to converging
            if balancer.convergence_number <= NUMBER_OF_CONVERGENCES_TO_SAVE:
                balancer.save_parameters()

#  Class required to efficiently capture camera frames 
class Pendulum_Balancer:

    def __init__(self, arduino_serial_connection):
        #  Save the arduino connection
        self.due_serial = arduino_serial_connection
        #  Variables used for analysis:
        self.loop_counter = 0.
        self.total_rewards = 0.
        self.convergence_number = np.inf

        #  Variables used for controlling pendulum:
        #  Set Motor speed.  This is sent to the motor every MOL.
        self.new_motor_speed = 0.
        self.motor_speeds = np.arange(start=-MAXIMUM_ABS_MOTOR_SPEED,stop=MAXIMUM_ABS_MOTOR_SPEED + (2.*MAXIMUM_ABS_MOTOR_SPEED/(NUMBER_MOTOR_SPEEDS-1)),step=(2.*MAXIMUM_ABS_MOTOR_SPEED/(NUMBER_MOTOR_SPEEDS-1)))
        self.should_reset_motor = False

        #  RL variables
        self.copy_parameters(Pendulum_Balancer.empty_parameters())
        motor_angle, motor_speed, motor_direction, pendulum_angle, pendulum_speed, pendulum_direction = self.get_pendulum_state()
        self.old_state = self.get_state(motor_angle, motor_speed, motor_direction, pendulum_angle, pendulum_speed, pendulum_direction)
        self.action = int(np.ceil(NUMBER_MOTOR_SPEEDS / 2.))

    def save_parameters(self):
        np.savez('RL parameters.npz', V=self.V, P_sa = self.P_sa, rewards = self.rewards, model_P = self.model_P, model_R = self.model_R)

    @staticmethod
    def setup_iteration_save_file():
        #  Access mode "w" to clear the file
        with open("RL Iteration.csv", "w") as save_file:
            #  Add CSV headers
            save_file.write("Total Reward,Iterations Until Convergence\n")
        
    def save_iteration_results(self):
        #  Access mode "a" to append to the file
        with open("RL Iteration.csv", "a") as save_file:
            save_file.write(str(self.total_rewards) + "," + str(self.convergence_number) + "\n")

    @property
    def parameters(self):
        return {"V":self.V,
                "P_sa":self.P_sa,
                "rewards":self.rewards,
                "model_P":self.model_P,
                "model_R":self.model_R}

    @staticmethod
    def parameters_from_file():
        parameters = {}
        with np.load('RL parameters.npz') as saved_parameters:
            parameters = {"V":saved_parameters["V"],
                          "P_sa":saved_parameters["P_sa"],
                          "rewards":saved_parameters["rewards"],
                          "model_P":saved_parameters["model_P"],
                          "model_R":saved_parameters["model_R"]}
        return parameters

    @property
    def epsilon(self):
        annealed_epsilon = 1 - (float(self.loop_counter) / ANNEALING_DURATION)
        return annealed_epsilon if annealed_epsilon > 0.05 else 0.1

    @staticmethod
    def empty_parameters():
        return {"V" : np.random.uniform(0,0.1,(NUMBER_STATES,1)),
                "P_sa": np.full((NUMBER_STATES, NUMBER_MOTOR_SPEEDS, NUMBER_STATES), 1 / float(NUMBER_STATES)),
                "rewards" : np.zeros((NUMBER_STATES, 1)),
                "model_P" : np.zeros((NUMBER_STATES, NUMBER_MOTOR_SPEEDS, NUMBER_STATES)),
                "model_R" : np.zeros((NUMBER_STATES,2))}

    def copy_parameters(self, parameters):
        self.V = np.copy(parameters["V"])
        self.P_sa = np.copy(parameters["P_sa"])
        self.rewards = np.copy(parameters["rewards"])
        self.model_P = np.copy(parameters["model_P"])
        self.model_R = np.copy(parameters["model_R"])

    def observe_reward(self, pendulum_angle, motor_angle):
        pendulum_reward = 0.
        motor_reward = 0.
        #  Pendulum angle scales up from zero to one at the top
        if pendulum_angle < np.pi:
            pendulum_reward += (pendulum_angle - np.pi / 4) / np.pi
        else:
            pendulum_reward += (2*np.pi - (pendulum_angle + np.pi / 4)) / np.pi

        if motor_angle > np.pi / 2 and motor_angle < 3 * np.pi / 2:
            motor_reward += -5
            
        return motor_reward + pendulum_reward

    def update_MDP_model(self):
        total_actions_taken = np.sum(self.model_P, keepdims = True, axis = 2)
        np.divide(self.model_P, total_actions_taken, out = self.P_sa, where = total_actions_taken != 0)
        total_rewardable_states = self.model_R[:,1].reshape((NUMBER_STATES,1))
        np.divide(self.model_R[:,0].reshape((NUMBER_STATES,1)), total_rewardable_states, out = self.rewards, where = total_rewardable_states!=0)

    def execute_value_iteration(self):
        #  Compare this with the old V(s)
        diffs = TOLERANCE + 1
        num_iterations = 0
        while diffs > TOLERANCE:
            #  Start another iteration
            V_new = self.rewards + np.amax(DISCOUNT_FACTOR * (self.P_sa @ self.V), axis=1)
            diffs = np.linalg.norm(V_new - self.V, np.inf)
            self.V = V_new
            #  Increment the iteration counter
            num_iterations += 1
        self.V = V_new
        print("Number of iterations until convergence: " + str(num_iterations))

    def get_state(self, motor_angle, motor_speed, motor_direction, pendulum_angle, pendulum_speed, pendulum_direction):
        #  Find the motor_angle sub-state
        motor_angle_sub_state = int(motor_angle * NUMBER_MOTOR_SPEEDS / (2*np.pi))
        #  Find the motor_speed sub-state
        motor_speed_sub_state = 0 if motor_speed == 0 else 1 if motor_direction == CLOCKWISE else 2
        #  Find the pendulum_angle sub-state
        pendulum_angle_sub_state = int(pendulum_angle * NUMBER_PENDULUM_SPEEDS / (2*np.pi))
        #  Find the pendulum_speed sub-state
        if pendulum_direction == CLOCKWISE:
            if pendulum_speed < 0.1:
                pendulum_speed_sub_state = 0
            elif pendulum_speed < 0.2:
                pendulum_speed_sub_state = 1
            else:
                pendulum_speed_sub_state = 2
        else:
            if pendulum_speed < 0.1:
                pendulum_speed_sub_state = 3
            elif pendulum_speed < 0.2:
                pendulum_speed_sub_state = 4
            else:
                pendulum_speed_sub_state = 5

        return motor_angle_sub_state + \
               NUMBER_MOTOR_ANGLES * motor_speed_sub_state + \
               NUMBER_MOTOR_ANGLES * NUMBER_MOTOR_SPEEDS * pendulum_angle_sub_state + \
               NUMBER_MOTOR_ANGLES * NUMBER_MOTOR_SPEEDS * NUMBER_PENDULUM_ANGLES * pendulum_speed_sub_state
        
    #  Main Operating Loop (MOL)
    def write(self, buf):

        #  Get physical parameters for learning
        motor_angle, motor_speed, motor_direction, pendulum_angle, pendulum_speed, pendulum_direction = self.get_pendulum_state()

        #  Capture new frame from camera (just the first plane in YUV, which is grayscale (luma)
        camera_frame = np.frombuffer(buf, dtype=np.uint8, count=IMAGE_WIDTH*IMAGE_HEIGHT).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

        #######################################################################
        #  \/ MACHINE LEARNING GOES HERE \/
        #  VARIABLES:
        #   self.new_motor_speed:       (float) The speed of the motor for the next MOL.
        #   self.action:                (int) The speed that was chosen last loop.
        #   camera_frame:               (3D numpy array) camera frame captured at the beginning of this MOL


        #  Discretize which state this is
        new_state = self.get_state(motor_angle, motor_speed, motor_direction, pendulum_angle, pendulum_speed, pendulum_direction)
        reward = self.observe_reward(pendulum_angle, motor_angle)
        self.total_rewards += reward
        #  Record the number of times `state, action, new_state` occurs
        self.model_P[self.old_state][self.action][new_state] += 1
        #  Record the rewards for every `new_state`
        self.model_R[new_state][0] += reward
        #  Record the number of times `new_state` was reached
        self.model_R[new_state][1] += 1
        self.old_state = new_state


        #  /\ MACHINE LEARNING GOES HERE /\
        #######################################################################

        #######################################################################
        #  \/ EXECUTION GOES HERE \/
        #  VARIABLES:
        #   self.new_motor_speed:       (float) The speed of the motor for the next MOL.
        #   self.action:                (int) The speed that was chosen last loop.
        #   camera_frame:               (3D numpy array) camera frame captured at the beginning of this MOL


        min_action = np.argmin(self.P_sa[new_state] @ self.V)
        greedy_action = np.argmax(self.P_sa[new_state] @ self.V)
        random_action = np.random.choice(np.arange(NUMBER_MOTOR_SPEEDS))
        if min_action == self.action:
            greedy_action = np.random.choice(np.arange(NUMBER_MOTOR_SPEEDS))
        self.action = np.random.choice([random_action, greedy_action], p=[self.epsilon, 1-self.epsilon])


        #  /\ EXECUTION GOES HERE /\
        #######################################################################
        
        #  Set new motor speed
        self.set_motor_speed(self.motor_speeds[self.action])

        #  Keep track of how efficient the algorithm is
        self.loop_counter += 1

    def flush(self):
        #######################################################################
        #  REPORT STATISTICS HERE
        #  Baseline MOL frequency is ~60Hz (just read/write)
        self.set_motor_speed(0.)
        #print("Finished one episode. MOL frequency (Hz): " + "{:.2f}".format(self.loop_counter / (time.time()-self.loop_time_start)))
        print("Total rewards earned: " + "{:.2f}".format(self.total_rewards))
        #print("Total number of loops: " + str(self.loop_counter))

        #######################################################################
        
    #  Reset motor position here.  Blocks until pendulum is back in the middle.
    def reset_motor(self):
        #  1: Stop the motor
        self.brake_motor()
        #  2: Determine which direction is fastest back to the center position.
        motor_angle, _, _, _, _, _ = self.get_pendulum_state()
        moving_clockwise = motor_angle < np.pi
        slow_return_speed = MINIMUM_ABS_MOTOR_SPEED if not moving_clockwise else -MINIMUM_ABS_MOTOR_SPEED
        #  3: Move slowly back to that position.  If overshoot, just stop.  (Close enough)
        while (moving_clockwise and motor_angle < np.pi) or (not moving_clockwise and  motor_angle >= np.pi):
            self.set_motor_speed(slow_return_speed)
            motor_angle, _, _, _, _, _ = self.get_pendulum_state()
        #  1: Stop the motor
        self.brake_motor()
        #  2: Determine which direction is fastest back to the center position.
        motor_angle, _, _, _, _, _ = self.get_pendulum_state()
        moving_clockwise = motor_angle < np.pi
        slow_return_speed = MINIMUM_ABS_MOTOR_SPEED if not moving_clockwise else -MINIMUM_ABS_MOTOR_SPEED
        #  3: Move slowly back to that position.  If overshoot, just stop.  (Close enough)
        while (moving_clockwise and motor_angle < np.pi) or (not moving_clockwise and  motor_angle >= np.pi):
            self.set_motor_speed(slow_return_speed)
            motor_angle, _, _, _, _, _ = self.get_pendulum_state()
        self.brake_motor()

    #  Recommended: Do not call the following functions explicitly.  Stay within MOL.
    def set_motor_speed(self, motor_speed):
        if motor_speed > MAXIMUM_ABS_MOTOR_SPEED:
            motor_speed = MAXIMUM_ABS_MOTOR_SPEED
        if motor_speed < -MAXIMUM_ABS_MOTOR_SPEED:
            motor_speed = -MAXIMUM_ABS_MOTOR_SPEED
        #  Send the new motor speed to the arduino.
        self.due_serial.write(("{:.2f}".format(motor_speed) + ",0X").encode('ASCII'))
        time.sleep(ARDUINO_EXECUTE_TIME)
    def brake_motor(self):
        #  Send the new motor speed to the arduino.
        self.due_serial.write(("2X").encode('ASCII'))
        time.sleep(ARDUINO_EXECUTE_TIME)
    def set_smoothing_factor(self, smoothing_factor):
        #  Send the new motor speed to the arduino.
        self.due_serial.write(("{:.2f}".format(smoothing_factor) + ",1X").encode('ASCII'))
        time.sleep(ARDUINO_EXECUTE_TIME)
    def get_pendulum_state(self):
        #  Send the new motor speed to the arduino.
        self.due_serial.write(("3X").encode('ASCII'))
        #  Wait for the full arduino response, convert it to a Python string, then a Python object (from JSON)
        params = json.loads(self.due_serial.readline().decode('ASCII'))
        return (params['motor_angle'], params['motor_speed'], params['motor_direction'], params['pendulum_angle'], params['pendulum_speed'], params['pendulum_direction'])

if __name__ == "__main__":
    main()

