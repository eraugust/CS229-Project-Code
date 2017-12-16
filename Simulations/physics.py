import numpy
import math
import cv2
import matplotlib.pyplot as plt
import neural


# https://en.wikipedia.org/wiki/Furuta_pendulum
class pendulum:

	def __init__(self):
		#############	System parameters	 ###################
		# arm 0 is motor arm, arm 1 is free arm (pendulum)
		self.l = numpy.array([1, 1]);
		self.m_pos = numpy.array([0.5, 0.5]);
		self.m = numpy.array([0.300, 0.1]);
		self.damp = -1 * numpy.array([1e-3, 3e-3]);
		# assumptions about arms :
		# long & slender => MOI around axis of arms = 0
		# rot symmetry  => MOI in two principal axes are equal
		J1 = 2.48e-2;
		J2 = 3.86e-3;
		J1_mat = [	[0, 0, 	0],	\
					[0, J1, 0], \
					[0, 0, J1]	];
		J2_mat = [	[0, 0,  0],	\
					[0, J2, 0],	\
					[0, 0, J2]	];
		# storing 	J0_hat = J1 + m1*l1^2 + m2*L1^2 
		# and 		J2_hat = J2 + m2*l2^2
		self.inertia = numpy.array(									[\
			J1+self.m[0]*self.m_pos[0]**2+self.m[1]*self.l[0]**2	,\
			J2+self.m[1]*self.m_pos[1]**2							]\
		);
		#############		State space		###################
		self.num_theta_states = 4
		self.num_dtheta_states = 4
		self.num_states = self.num_theta_states*self.num_dtheta_states+1
		# self.theta_lim = math.pi/180*numpy.array([150, 210])
		# self.dtheta_lim = math.pi/180*numpy.array([-15, 15])
		self.theta_lim = math.pi*150/180
		self.dtheta_lim = math.pi*15/180
		self.torque = numpy.array([0, 0]);
		# motor arm = 0rad corresponds to free hang
		self.theta = numpy.array([0, 0.03]);
		self.dtheta = numpy.array([0, 0]);
		self.ddtheta = numpy.array([0, 0]);
		self.dt = 0.1 # 100 ms update time
		self.trajectory = [	self.theta.tolist()		+\
							self.dtheta.tolist()	+\
							self.ddtheta.tolist()	+\
							self.torque.tolist()	];
		#############		Image params	###################
		self.h = 64
		self.w = 48
		self.d = 3
		self.theta_nn = None


	def update(self, torque_in):
		"""
		Simulation dynamics of the rotary pendulum system
		using input torque acting on the motor arm
		"""
		self.torque = numpy.array(torque_in)
		dthetadt = numpy.transpose(numpy.matrix(self.dtheta / self.dt));
		torque = numpy.transpose(numpy.matrix(self.torque));
		grav = numpy.matrix(											[\
		[	0															],\
		[	9.81*self.m[1]*self.m_pos[1]*math.sin(self.theta[1])		]]\
		);
		det = self.inertia[0]*self.inertia[1]+self.inertia[1]**2*math.sin(self.theta[1])**2 -\
			self.m[1]**2*self.l[0]**2*self.m_pos[1]**2*math.cos(self.theta[1])**2;
		w1_adj = numpy.matrix(											[\
		[	self.inertia[1]												,\
			-self.m[1]*self.l[0]*self.m_pos[1]*math.cos(self.theta[1])	],\
		[	-self.m[1]*self.l[0]*self.m_pos[1]*math.cos(self.theta[1])	,\
			self.inertia[0]+self.inertia[1]*math.sin(self.theta[1])**2	]]\
		);
		w2 = numpy.matrix(																		[\
		[	self.damp[0]+0.5*self.dtheta[1]/self.dt*self.inertia[1]*math.sin(2*self.theta[1])	,\
			0.5*self.dtheta[1]/self.dt*self.inertia[1]*math.sin(2*self.theta[1])- \
			self.m[1]*self.l[0]*self.m_pos[1]*math.sin(self.theta[1])*self.dtheta[1]/self.dt	],\
		[	-0.5*self.dtheta[0]/self.dt*self.inertia[1]*math.sin(2*self.theta[1])				,\
			self.damp[1]																		]]\
		);
		# update ddtheta
		ddtheta = w1_adj / det * (w2*dthetadt - grav + torque) * self.dt**2;
		# update dynamics
		self.ddtheta = numpy.squeeze(numpy.asarray(ddtheta))
		self.dtheta = self.dtheta + self.ddtheta;
		self.theta = (self.theta + self.dtheta) % (2*numpy.pi);
		assert(self.theta[1] >= 0)
		self.trajectory.append(	self.theta.tolist()		+\
								self.dtheta.tolist()	+\
								self.ddtheta.tolist()	+\
								self.torque.tolist()	);

	def label(self, x, lim, n):
		assert(lim[0] <= x)
		assert(x <= lim[1])
		step_size = (lim[1] - lim[0]) / n
		return (x - lim[0]) / step_size
	
	def get_state(self):
		"""
		Discretizes the continuous state vector
		Only theta from the arm contributes to state
		"""

		unwrap = False
		if self.theta[1] > math.pi:
			self.theta[1] -= 2*math.pi
			unwrap = True

		abs_theta = abs(self.theta[1])
		abs_dtheta = abs(self.dtheta[1])

		if abs_theta < self.theta_lim or self.dtheta_lim < abs_dtheta:
			state = self.num_states - 1
		else:
			# evenly spaced states
			theta_step = (math.pi-self.theta_lim) / (self.num_theta_states/2)
			dtheta_step = self.dtheta_lim / (self.num_dtheta_states/2)
			if self.theta[1] > 0:
				state = math.floor((math.pi-abs_theta) / theta_step)*2
			else:
				state = math.floor((math.pi-abs_theta) / theta_step)*2+1
			assert(state < self.num_theta_states)
			if self.dtheta[1] > 0:
				dstate = math.floor(abs_dtheta / dtheta_step)*2
			else:
				dstate = math.floor(abs_dtheta / dtheta_step)*2+1
			assert(dstate < self.num_dtheta_states)
			state += self.num_theta_states*dstate

		if unwrap:
			self.theta[1] += 2*math.pi

		return state

	def get_state_img(self, nn):

		img = self.generate_img()
		img_vec = img[:,:,0].reshape(self.h*self.w)
		theta_nn_step = nn.predict(img_vec)

		theta_nn1 = theta_nn_step / nn.output_size * 2 * math.pi
		theta_nn2 = (theta_nn_step+1) / nn.output_size * 2 * math.pi
		theta_nn = 0.5*(theta_nn1 + theta_nn2)

		if self.theta_nn is not None:
			dtheta = theta_nn - self.theta_nn
		else:
			dtheta = 0

		unwrap = False
		if theta_nn > math.pi:
			theta_nn -= 2*math.pi
			unwrap = True

		abs_theta = abs(theta_nn)
		abs_dtheta = abs(dtheta)

		if abs_theta < self.theta_lim or self.dtheta_lim < abs_dtheta:
			state = self.num_states - 1
		else:
			# evenly spaced states
			theta_step = (math.pi-self.theta_lim) / (self.num_theta_states/2)
			dtheta_step = self.dtheta_lim / (self.num_dtheta_states/2)
			if theta_nn > 0:
				state = math.floor((math.pi-abs_theta) / theta_step)*2
			else:
				state = math.floor((math.pi-abs_theta) / theta_step)*2+1
			assert(state < self.num_theta_states)
			if dtheta > 0:
				dstate = math.floor(abs_dtheta / dtheta_step)*2
			else:
				dstate = math.floor(abs_dtheta / dtheta_step)*2+1
			assert(dstate < self.num_dtheta_states)
			state += self.num_theta_states*dstate

		if unwrap:
			theta_nn += 2*math.pi

		self.theta_nn = theta_nn

		return state


	def generate_img(self):
		L = self.l[1] * 300
		c = (int(self.h/2), int(self.w/2))
		end_pt = (int(L*math.cos(self.theta[1]) + c[0]), int(L*math.sin(self.theta[1]) + c[1]))
		img = numpy.zeros((self.w,self.h,self.d), numpy.uint8)
		img = cv2.line(img, c, end_pt, (223, 120, 79), 2)
		return img

	def plot_arm(self, arm):
		assert(0 <= arm and arm <= 1)
		trajectory = [state[arm] for state in self.trajectory]
		plt.polar(trajectory, 1 / (0.1*numpy.array(range(len(trajectory))) + 1))
		plt.show()


	def reset(self,theta=[0,math.pi],dtheta=[0,0],ddtheta=[0,0],torque=[0,0]):
		self.torque = numpy.array(torque);
		self.theta = numpy.array(theta);
		self.dtheta = numpy.array(dtheta);
		self.ddtheta = numpy.array(ddtheta);
		self.trajectory = [	self.theta.tolist()		+\
							self.dtheta.tolist() 	+\
							self.ddtheta.tolist() 	+\
							self.torque.tolist() 	];







