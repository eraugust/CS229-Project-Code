import math
import numpy as np
import matplotlib.pyplot as plt
import neural
import physics
import time

p = physics.pendulum()
nn = neural.net()

noise_range = 1
step_size = 2 * math.pi / nn.output_size

print('training')

start = time.time()

######## Use pre-trained network ########
# nn.load_params('trained7.npz')

######## train network ########
train_iters = 100000
for i in range(int(train_iters / nn.mem_size)):
	for j in range(nn.mem_size):
		theta1 = np.random.uniform(0, 2*math.pi)
		p.reset(theta=[0,theta1])
		img = p.generate_img()
		img_vec = img[:,:,0].reshape(p.h*p.w)
		img_vec += np.random.randint(0, noise_range, size=len(img_vec), dtype=np.uint8)
		label = math.floor(theta1 / step_size)
		onehot = np.zeros(nn.output_size)
		onehot[label] = 1
		nn.memorize(img_vec, onehot)
	nn.train_memory()
	if i % 100 == 0:
		print(i)
################################

end = time.time()
print(end - start)

nn.save_params('trained7.npz')



print('testing')
test_iters = 1000
num_right = 0
err = 0
for i in range(test_iters):
	theta1 = np.random.uniform(0, 2*math.pi)
	p.reset(theta=[0,theta1])
	img = p.generate_img()
	img_vec = img[:,:,0].reshape(p.h*p.w)
	img_vec += np.random.randint(0, noise_range, size=len(img_vec), dtype=np.uint8)
	label = math.floor(theta1 / step_size)
	prediction = nn.predict(img_vec)
	print(label)
	print(prediction)
	print("")
	if label == prediction:
		num_right += 1 # spot-on accuracy
	err += abs(label - prediction)*step_size*180.0/math.pi/ test_iters # mean error

print("error " + str(err))
print("accuracy " + str(num_right * 100.0 / test_iters) )











