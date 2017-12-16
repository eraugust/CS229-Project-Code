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
MOTOR_ANGLE_SPREAD_LIMIT = np.pi    #  Must be between 0 and 2PI
MINIMUM_ABS_MOTOR_SPEED = 0.17       #  Unloaded is 0.15.  Might be more with pendulum attached
SMOOTHING_FACTOR = 0.2              #  For the exponential smoothing of the encoder speeds.
ARDUINO_EXECUTE_TIME = 0.002        #  Time delay so the arduino can execute the code.
MAXIMUM_ABS_MOTOR_SPEED = 0.3       #  Motor speed clipped before sending to arduino
RESET_SLEEP_TIME = 2                #  Sleep time so the pendulum can settle
#  Camera
MAXIMUM_MOL_FREQUENCY = 40          #  (Same as framerate for camera) MOLs are forced to wait if they finish too quickly.
IMAGE_WIDTH = 32                    #  Captured frame will be downsized to this (Must be divisible by 32)
IMAGE_HEIGHT = 32                   #  Captured frame will be downsized to this (Must be divisible by 16)
CAMERA_SENSOR_MODE = 4              #  Recommend 4 (full field of view, max 40fps) or 6 (smaller fieldof view, 41-90fps)
EXPOSURE_MODE = 'sports'            #  Recommend 'sports' for the faster shutter speed
RECORDING_TIME = 30                 #  Amount of time that camera is recording

# Machine Learning
DISCOUNT_FACTOR = 0.99              #  This is gamma, the discount learning factor
ANNEALING_DURATION = RECORDING_TIME*10/5        #  Amount of time epsilon is annealed from 1.0 to 0.1 (a fiftieth of the frames)
HIDDEN_LAYER_SIZE = 100              #  Amount of nodes in the NN hidden layer
ACTION_REWARD_DELAY = 0.000001      #  Delay time between executing action and observing reward
REPLAY_MEMORY_SIZE = 1000           #  Replay memory size
MINI_BATCH_SIZE = 10                #  Mini-batch size
Q_HAT_UPDATE_DURATION = 100         #  Every so often, set Q hat to Q
NUMBER_NEURAL_NETWORK_OUTPUTS = 21  #  K of the softmax output function
LEARNING_RATE = 1                   #  Learning rate used for neuralnetwork gradient descent
FORGETTING_FACTOR = 0.9             #  The forgetting factor for the RMSProp algorithm

def main():
    #  Connect to the arduino.  It will automatically reset.
    with serial.Serial('/dev/ttyACM0', baudrate=115200) as due_serial:
        time.sleep(2)
        print("Arduino initialized, turn on power supply.")
        #  Training loop.  Each loop is an episode where the balancer can swing up the pendulum.
        #parameters = Pendulum_Balancer.parameters_from_file()
        parameters = Pendulum_Balancer.empty_parameters()
        while True:
            #  Initialize pendulum balancer
            balancer = Pendulum_Balancer(due_serial)
            balancer.copy_parameters(parameters)
            #  Connect to the camera
            with picamera.PiCamera(sensor_mode=CAMERA_SENSOR_MODE, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT), framerate=MAXIMUM_MOL_FREQUENCY) as camera:
                #  Warm up the camera (set gain/white balance, etc.)
                time.sleep(2)
                camera.exposure_mode = EXPOSURE_MODE
                #  Start the preview for debugging
                camera.start_preview()
                #  Start the timer on the balancer, for later analysis of FPS
                balancer.loop_time_start = time.time()
                #  Record
                camera.start_recording(balancer, 'yuv')
                camera.wait_recording(RECORDING_TIME)
                camera.stop_recording()
            #  Reset the motor position
            #balancer.reset_motor()
            parameters = balancer.save_parameters()

#  Class required to efficiently capture camera frames 
class Pendulum_Balancer:

    def __init__(self, arduino_serial_connection):
        #  Save the arduino connection
        self.due_serial = arduino_serial_connection
        #  Variables used for analysis:
        self.loop_counter = 0.
        self.total_rewards = 0.

        #  Variables used for controlling pendulum:
        #  Set Motor speed.  This is sent to the motor every MOL.
        self.new_motor_speed = 0.
        #  Command a motor reset.  This is checked every MOL.
        self.should_reset_motor = False

        #  Variables for machine learning
        self.last_four_frames = np.zeros((4, IMAGE_WIDTH * IMAGE_HEIGHT + 2))
        self.motor_speeds = np.arange(start=-1,stop=1.1,step=0.1)
        self.building_first_state = True
        self.truth_action_mask = np.ones((len(self.motor_speeds), MINI_BATCH_SIZE))
        #  Initialize replay memory
        self.replay_memory_pre_states = np.zeros((4*IMAGE_WIDTH*IMAGE_HEIGHT + 8, REPLAY_MEMORY_SIZE))
        self.replay_memory_post_states = np.zeros((4*IMAGE_WIDTH*IMAGE_HEIGHT + 8, REPLAY_MEMORY_SIZE))
        self.replay_memory_actions = np.zeros(REPLAY_MEMORY_SIZE).astype(int)
        self.replay_memory_rewards = np.zeros(REPLAY_MEMORY_SIZE)
        self.size_of_replay_memory = 0
        #  Initialize NN parameters
        #  Initialize W1 as a matrix with shape (num_hidden X input_size), sampled according to Xavier
        #  Initialize W2 as a matrix with shape (K X num_hidden), sampled according to Xavier
        #  Initialize b1 as a vector with shape (num_hidden X 1), all zeros
        #  Initialize b2 as a vector with shape (K X 1), all zeros
        self.W1_Q = np.random.normal(0,np.sqrt(2. / (NUMBER_NEURAL_NETWORK_OUTPUTS + 4*IMAGE_WIDTH*IMAGE_HEIGHT + 8)), (HIDDEN_LAYER_SIZE, 4*IMAGE_WIDTH*IMAGE_HEIGHT + 8))
        self.W2_Q = np.random.normal(0,np.sqrt(2. / (NUMBER_NEURAL_NETWORK_OUTPUTS + HIDDEN_LAYER_SIZE)), (NUMBER_NEURAL_NETWORK_OUTPUTS, HIDDEN_LAYER_SIZE))
        self.b1_Q = np.zeros((HIDDEN_LAYER_SIZE, 1))
        self.b2_Q = np.zeros((NUMBER_NEURAL_NETWORK_OUTPUTS,1))
        self.W1_Q_hat = np.random.normal(0,np.sqrt(2. / (NUMBER_NEURAL_NETWORK_OUTPUTS + 4*IMAGE_WIDTH*IMAGE_HEIGHT + 8)), (HIDDEN_LAYER_SIZE, 4*IMAGE_WIDTH*IMAGE_HEIGHT + 8))
        self.W2_Q_hat = np.random.normal(0,np.sqrt(2. / (NUMBER_NEURAL_NETWORK_OUTPUTS + HIDDEN_LAYER_SIZE)), (NUMBER_NEURAL_NETWORK_OUTPUTS, HIDDEN_LAYER_SIZE))
        self.b1_Q_hat = np.zeros((HIDDEN_LAYER_SIZE, 1))
        self.b2_Q_hat = np.zeros((NUMBER_NEURAL_NETWORK_OUTPUTS,1))
        self.W1_RMSProp = np.ones(self.W1_Q.shape)
        self.W2_RMSProp = np.ones(self.W2_Q.shape)
        self.b1_RMSProp = np.ones(self.b1_Q.shape)
        self.b2_RMSProp = np.ones(self.b2_Q.shape)

    @property
    def epsilon(self):
        annealed_epsilon = 1 - (float(self.loop_counter) / ANNEALING_DURATION)
        return annealed_epsilon if annealed_epsilon > 0.1 else 0.1

    def save_parameters(self):
        with open('NN parameters.npz', 'wb') as outfile:
            np.savez(outfile, W1=self.W1_Q, W2=self.W2_Q, b1=self.b1_Q, b2=self.b2_Q)
        return {"W1":self.W1_Q, "W2":self.W2_Q, "b1":self.b1_Q, "b2":self.b2_Q}

    @staticmethod
    def parameters_from_file():
        parameters = {}
        with np.load('NN parameters.npz') as saved_parameters:
            parameters = {"W1":np.copy(saved_parameters["W1"]),
                          "W2":np.copy(saved_parameters["W2"]),
                          "b1":np.copy(saved_parameters["b1"]),
                          "b2":np.copy(saved_parameters["b2"])}
        return parameters

    def copy_parameters(self, parameters):
        np.copyto(self.W1_Q, parameters["W1"])
        np.copyto(self.W2_Q, parameters["W2"])
        np.copyto(self.b1_Q, parameters["b1"])
        np.copyto(self.b2_Q, parameters["b2"])

    @staticmethod
    def empty_parameters():
        return {"W1":np.random.normal(0,np.sqrt(2. / (NUMBER_NEURAL_NETWORK_OUTPUTS + 4*IMAGE_WIDTH*IMAGE_HEIGHT + 8)), (HIDDEN_LAYER_SIZE, 4*IMAGE_WIDTH*IMAGE_HEIGHT + 8)),
                "W2":np.random.normal(0,np.sqrt(2. / (NUMBER_NEURAL_NETWORK_OUTPUTS + HIDDEN_LAYER_SIZE)), (NUMBER_NEURAL_NETWORK_OUTPUTS, HIDDEN_LAYER_SIZE)),
                "b1":np.zeros((HIDDEN_LAYER_SIZE, 1)),
                "b2":np.zeros((NUMBER_NEURAL_NETWORK_OUTPUTS,1))}

    def remember_new_frame(self, frame):
        #  Shift the frames by one time step
        self.last_four_frames = np.roll(self.last_four_frames, 1)
        #  Forget the last frame, remember the first
        #  Need a copy here?
        self.last_four_frames[0] = frame.reshape((-1))

    def condition_input_frames_for_NN(self, four_frames):
        #  The input should be a 1D array, flatten the frames together
        return four_frames.reshape((-1))

    def observe_reward(self, pendulum_angle, motor_angle):
        #  Assume that 0 is pointing down
        pendulum_reward = 0
        motor_reward = 0
        if (pendulum_angle < np.pi / 4 or pendulum_angle > 7 * np.pi / 4):
            #  Give no reward for a pendulum in the bottom facing quadrant
            pendulum_reward += -0.01
        elif (pendulum_angle < np.pi / 2 or pendulum_angle > 3 * np.pi / 2):
            #  Give partial reward for a pendulum in the bottom half
            pendulum_reward += 0.2
        else:
            #  Give full reward for a pendulum in the top half (unstable)
            pendulum_reward += 1
        if motor_angle > np.pi / 2 and motor_angle < 3 * np.pi / 2:
            motor_reward += -1

        return -(pendulum_reward + motor_reward)

    def save_to_replay_memory(self, pre_state, action, reward, post_state):
        if self.size_of_replay_memory == REPLAY_MEMORY_SIZE:
            #  Shift the replay memory by one (circular)
            self.replay_memory_pre_states = np.roll(self.replay_memory_pre_states, 1, axis=1)
            self.replay_memory_post_states = np.roll(self.replay_memory_post_states, 1, axis=1)
            self.replay_memory_actions = np.roll(self.replay_memory_actions, 1)
            self.replay_memory_rewards = np.roll(self.replay_memory_rewards, 1)
            #  Save the new memory
            #  Need a copy here?
            self.replay_memory_pre_states[:,0] = pre_state
            self.replay_memory_post_states[:,0] = post_state
            self.replay_memory_actions[0] = action
            self.replay_memory_rewards[0] = reward
        else:
            #  Need a copy here?
            self.replay_memory_pre_states[:,self.size_of_replay_memory] = pre_state
            self.replay_memory_post_states[:,self.size_of_replay_memory] = post_state
            self.replay_memory_actions[self.size_of_replay_memory] = action
            self.replay_memory_rewards[self.size_of_replay_memory] = reward
            self.size_of_replay_memory += 1

    def update_Q_hat(self):
        #  Copy the NN parameters from Q to Q hat
        np.copyto(self.W1_Q_hat, self.W1_Q)
        np.copyto(self.W2_Q_hat, self.W2_Q)
        np.copyto(self.b1_Q_hat, self.b1_Q)
        np.copyto(self.b2_Q_hat, self.b2_Q)
    
    def predict_Q(self, state):
        z1 = self.W1_Q @ state + self.b1_Q
        a1 = sigmoid(z1)
        z2 = self.W2_Q @ a1 + self.b2_Q
        a2 = softmax(z2)

        h = a1
        y = a2

        return y, h
    
    def predict_Q_hat(self, state):
        z1 = self.W1_Q_hat @ state + self.b1_Q_hat
        a1 = sigmoid(z1)
        z2 = self.W2_Q_hat @ a1 + self.b2_Q_hat
        a2 = softmax(z2)

        h = a1
        y = a2

        return y, h

    def execute_gradient_descent_on_Q(self, pre_states, actions, post_states, rewards):

        if (np.linalg.norm(rewards, np.inf)) == 0:
            return
        
        #  Perform a single forward pass on the pre-state
        y, h = self.predict_Q(pre_states)

        #  Perform a single forward pass on the post state, then discount and add a reward
        truth = DISCOUNT_FACTOR * np.max(self.predict_Q_hat(post_states)[0], axis=0).reshape(1,-1) + rewards
        #  Calculate the layer deltas
        y_hat = np.copy(y)
        y_hat[actions,np.arange(y.shape[1])]=truth
        delta_2 = y_hat - y
        delta_1 = self.W2_Q.T @ delta_2 * h * (1 - h)

        W1_grad = delta_1 @ pre_states.T / MINI_BATCH_SIZE
        W2_grad = delta_2 @ h.T / MINI_BATCH_SIZE
        b1_grad = np.sum(delta_1, axis=1).reshape(self.b1_Q.shape) / MINI_BATCH_SIZE
        b2_grad = np.sum(delta_2, axis=1).reshape(self.b2_Q.shape) / MINI_BATCH_SIZE

        #  Update the RMSProp weights
        self.W1_RMSProp = FORGETTING_FACTOR * self.W1_RMSProp + (1 - FORGETTING_FACTOR) * W1_grad ** 2
        self.W2_RMSProp = FORGETTING_FACTOR * self.W2_RMSProp + (1 - FORGETTING_FACTOR) * W2_grad ** 2
        self.b1_RMSProp = FORGETTING_FACTOR * self.b1_RMSProp + (1 - FORGETTING_FACTOR) * b1_grad ** 2
        self.b2_RMSProp = FORGETTING_FACTOR * self.b2_RMSProp + (1 - FORGETTING_FACTOR) * b2_grad ** 2
        
        self.W1_Q -= LEARNING_RATE / np.sqrt(self.W1_RMSProp) * W1_grad
        self.W2_Q -= LEARNING_RATE / np.sqrt(self.W2_RMSProp) * W2_grad
        self.b1_Q -= LEARNING_RATE / np.sqrt(self.b1_RMSProp) * b1_grad
        self.b2_Q -= LEARNING_RATE / np.sqrt(self.b2_RMSProp) * b2_grad

        #self.W1_Q -= LEARNING_RATE * W1_grad
        #self.W2_Q -= LEARNING_RATE * W2_grad
        #self.b1_Q -= LEARNING_RATE * b1_grad
        #self.b2_Q -= LEARNING_RATE * b2_grad

    def get_action_index_from_motor_speed(self,motor_speed):
        return int(motor_speed*10.+10)

    #  Main Operating Loop (MOL)
    def write(self, buf):

        #  Get physical parameters for learning
        motor_angle, motor_speed, motor_direction, pendulum_angle, pendulum_speed, pendulum_direction = self.get_pendulum_state()

        #  Capture new frame from camera (just the first plane in YUV, which is grayscale (luma)
        actual_image = np.frombuffer(buf, dtype=np.uint8, count=IMAGE_WIDTH*IMAGE_HEIGHT).reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        camera_frame = np.append(actual_image.reshape((-1)).astype(np.float64),[motor_angle, motor_speed])

        #######################################################################
        #  PENDULUM SAFETY CONTROLS
        #  Assume the motor starts in the center position
        #  Limit the motor to MOTOR_ANGLE_SPREAD_LIMIT turn in each direction
        #if  (motor_angle > MOTOR_ANGLE_SPREAD_LIMIT / 2) and (motor_angle < 2 * np.pi - MOTOR_ANGLE_SPREAD_LIMIT / 2):
        #    self.should_reset_motor = True

        #  Reset the motor if requested (either for safety or algorithm)
        #if self.should_reset_motor:
        #    self.should_reset_motor = False
        #    self.reset_motor()
        #    self.new_motor_speed = 0.

        #######################################################################

        #######################################################################
        #  \/ MACHINE LEARNING GOES HERE \/
        #  VARIABLES:
        #   self.new_motor_speed:       (float) The speed of the motor for the next MOL.
        #   self.should_reset_motor:    (bool) Resets the motor speed/position to 0 at the next MOL.
        #   camera_frame:               (3D numpy array) camera frame captured at the beginning of this MOL

        #  Save this frame to the buffer
        self.remember_new_frame(camera_frame)

        #  If we are still building the first state, this means we are filling the "last_four_frames" buffer
        if self.building_first_state and self.loop_counter == 3:
            #  Start the clock for calculating efficiency
            self.first_state = self.condition_input_frames_for_NN(self.last_four_frames)
            self.building_first_state = False

        #  If the pendulum is reset, this episode is finished.  Wait until the camera is finished recording.
        if not self.building_first_state:
            #  Observe the reward
            reward = self.observe_reward(pendulum_angle, motor_angle)
            self.total_rewards += reward
            #  Get the pre/post states
            pre_state = self.first_state if self.size_of_replay_memory == 0 else self.replay_memory_post_states[:,0]
            post_state = self.condition_input_frames_for_NN(self.last_four_frames)
            self.save_to_replay_memory(pre_state, self.get_action_index_from_motor_speed(self.new_motor_speed), reward, post_state)

            #  TRAINING
            #  Sample one mini-batch for training
            batch_pre_states = self.replay_memory_pre_states[:,np.random.choice(np.arange(self.size_of_replay_memory), size=MINI_BATCH_SIZE, replace = True)]
            batch_post_states = self.replay_memory_post_states[:,np.random.choice(np.arange(self.size_of_replay_memory), size=MINI_BATCH_SIZE, replace = True)]
            batch_rewards = self.replay_memory_rewards[np.random.choice(np.arange(self.size_of_replay_memory), size=MINI_BATCH_SIZE, replace = True)].reshape((1, MINI_BATCH_SIZE))
            batch_actions = self.replay_memory_actions[np.random.choice(np.arange(self.size_of_replay_memory), size=MINI_BATCH_SIZE, replace = True)].reshape((1, MINI_BATCH_SIZE))
                        
            #  Perform gradient descent step on Q
            self.execute_gradient_descent_on_Q(batch_pre_states, batch_actions, batch_post_states, batch_rewards)
            
            #  Update Q hat to Q periodically
            if self.loop_counter % Q_HAT_UPDATE_DURATION == 0:
                #  Update the Q hat parameters periodically
                self.update_Q_hat()

        #  /\ MACHINE LEARNING GOES HERE /\
        #######################################################################

        #######################################################################
        #  \/ EXECUTION GOES HERE \/
        #  VARIABLES:
        #   self.new_motor_speed:       (float) The speed of the motor for the next MOL.
        #   self.should_reset_motor:    (bool) Resets the motor speed/position to 0 at the next MOL.

        #  Epsilon-greedy algorithm:
        #  With probability epsilon, shoose random motor speed.
        #  Otherwise, run a forward pass and pick the speed with the highest predicted reward
        if not self.building_first_state:
            action_expected_rewards,_ = self.predict_Q(self.condition_input_frames_for_NN(self.last_four_frames).reshape(-1,1))
            greedy_action = self.motor_speeds[np.argmax(action_expected_rewards)]
            random_action = np.random.choice(self.motor_speeds)
            self.new_motor_speed = np.random.choice([random_action, greedy_action], p=[self.epsilon, 1-self.epsilon])

        #  /\ EXECUTION GOES HERE /\
        #######################################################################
        
        #  Set new motor speed
        self.set_motor_speed(self.new_motor_speed)

        #  Keep track of how efficient the algorithm is
        self.loop_counter += 1

    def flush(self):
        #######################################################################
        #  REPORT STATISTICS HERE
        #  Baseline MOL frequency is ~60Hz (just read/write)
        self.set_motor_speed(0.)
        print("Finished one episode. MOL frequency (Hz): " + "{:.2f}".format(self.loop_counter / (time.time()-self.loop_time_start)))
        print("Total rewards earned: " + str(self.total_rewards))
        print("Total number of loops: " + str(self.loop_counter))
        print("Rewards per loop: " + "{:.2f}".format(self.total_rewards / self.loop_counter))


        #######################################################################

    #  Recommended: Do not call this explicitly.  Stay within MOL.
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
        time.sleep(RESET_SLEEP_TIME)
        
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

def softmax(x):
    """
    Compute softmax function for input. 
    """
    shifted_input = x - np.max(x, axis=0)
    numerators = np.exp(shifted_input)
    return (numerators / np.sum(numerators, axis=0))

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    #x = 1 / (1 + np.exp(-x))
    return expit(x)

if __name__ == "__main__":
    main()

