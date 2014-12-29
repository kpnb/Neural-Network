import random
import numpy as np

class Network(object):
	# creating the network class
	def __init__(self, nInputNode, nHiddenNode, nOutputNode):
		# learning rate
		self.alpha = 0.1
		# number of neurons per layer
		self.nInputNode = nInputNode
		self.nHiddenNode = nHiddenNode
		self.nOutputNode = nOutputNode
		# random weights for hidden and output nodes
		# the weights are between the input and hidden, and the hidden and output
		self.hiddenWeight = random.random() # +1 is for the bias node
		self.outputWeight = random.random()
		# node activation, sum of inputs in an array
		self.hiddenActivation = np.zeros((self.nHiddenNode+1, 1), dtype=float)
		self.outputActivation = np.zeros((self.nOutputNode, 1), dtype=float)
		# outputs
		self.inputOutput = np.zeros((self.nInputNode+1, 1), dtype=float)
		self.hiddenOutput = np.zeros((self.nHiddenNode+1, 1), dtype=float)
		self.outputOutput = np.zeros((self.nOutputNode, 1), dtype=float)
		# changes for hidden and output layers
		self.hiddenChange = np.zeros((self.nHiddenNode), dtype=float)
		self.outputChange = np.zeros((self.nOutputNode), dtype=float)

	def forward(self, input):
		# setting the input equal to the output of the first layer
		self.inputOutput[:-1, 0] = input
		self.inputOutput[-1:, 0] = 1.0
		# hidden layer
		self.hiddenActivation = np.dot(self.hiddenWeight, self.inputOutput)
		self.hiddenOutput[:, :] = self.hiddenActivation
		# hidden bias neuron to 1.0
		self.hiddenOutput[-1:, :] = 1.0
		# output layer
		self.outputActivation = np.dot(self.outputWeight, self.hiddenOutput)
		self.outputOutput = self.outputActivation

	def backward(self, teach):
		error = self.outputOutput - np.array(teach, dtype=float)

		# output neuron changes
		self.outputChange = (1 - np.tanh(self.outputActivation)) * np.tanh(self.outputActivation) * error

		# hidden neuron changes
		self.hiddenChange = (1 - np.tanh(self.hiddenActivation)) * np.tanh(self.hiddenActivation) * error

		# apply weight changes
		self.hiddenWeight = self.hiddenWeight - self.alpha * np.dot(self.hiddenChange, self.inputOutput.transpose())
		self.outputWeight = self.outputWeight - self.alpha * np.dot(self.outputChange, self.inputOutput.transpose())

	def getOutput(self):
		return self.outputOutput

if __name__ == '__main__':
	# training set
	addSet = [
			  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
			  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
			  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
			  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
			  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
			  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
			  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
			  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
			  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
			  ]
	addTeach = [[0], [1], [1], [0], [1], [0], [0], [1], [1], [1]]
	# create network
	network = Network(10, 10, 1)
	count = 0
	while(count < 10):
		# randomly choose a training sample
		rnd = random.randint(0, 9)
		# forward and backward
		network.forward(addSet[rnd])
		network.backward(addTeach[rnd])
		#output
		print "Count: ", count, "Set used for training: ", addSet[rnd], "Output: ", network.getOutput()[0], "True sum: ", np.sum(addSet[rnd])
		if network.getOutput()[0] == np.sum(addSet[rnd]):
			print "True"
		else:
			print "False"
			count += 1