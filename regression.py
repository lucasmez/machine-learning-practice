import numpy as np;

class Logistic():

	costHistory = [];
	alphaHistory = [];

	def __init__(self, Xs, Ys):
		self.X = Xs;
		self.Y = Ys;	
		self.m = Ys.size;
		self.thetas = np.ones(Xs.shape[1]);

	def resetTetha(self):
		self.thetas = np.ones(self.X.shape[1]);

	def getThetas(self):
		return self.thetas;

	def getCostHistory(self):
		return np.array(self.costHistory);

	def getAlphaHistory(self):
		return np.array(self.alphaHistory);

	def fit(self,
			numIters=1000,
			descentFactor=0.1, 
			regularizationFactor=0, 
			autoDescentAdjust=False, 
			descentAdjustRate=3):

		self.resetTetha();
		self.alphaHistory.append(descentFactor);

		while numIters > 0:
			self.thetas = self.thetas  - descentFactor * (self.computeDerivatives());
			self.costHistory.append(self.computeCost(regularizationFactor))

			if	len(self.costHistory) > 2 and autoDescentAdjust and (int(self.costHistory[-1]) > int(self.costHistory[-2])):
				descentFactor /= descentAdjustRate;
				self.alphaHistory.append(descentFactor);

			numIters -= 1;

		return self.thetas;
	
	def computeCost(self, alpha):
		sigmoid = self.computeHypothesis(self.X);
		firstTerm = (-self.Y.T).dot( np.log(sigmoid));
		secondTerm = ((1.0-self.Y).T).dot(np.log(1.0-sigmoid));
		result = firstTerm - secondTerm;

		if alpha > 0:
			regularization =  (alpha/self.m) * np.sum(self.thetas); 
			return (1.0/self.m) * (result + regularization);
		else:
			return (1.0/self.m) * result;

	def computeHypothesis(self, Xs):
		value = 1.0 / (1.0 + np.exp(-(Xs.dot(self.thetas)))); 
		# To avoid infinite values
		value[value > .99] = .99;
		value[value < .01] = .01;

		return value;

	def computeDerivatives(self):
		computed =  self.X.T.dot(self.computeHypothesis(self.X).T - self.Y.T);
		return computed;

	
	def estimate(self, Xs):
		#Append ones
		Xs = np.hstack([np.ones(Xs.shape[0])[:, np.newaxis], Xs]);
		return self.computeHypothesis(Xs);
