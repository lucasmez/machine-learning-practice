import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib import cm;
from mpl_toolkits.mplot3d import Axes3D

from regression import Logistic;
import data;



def plotRealTrainingData(Xs, Ys):
	# Plot real data
	zerosClass = Xs[:, 1:][Ys == 0];
	onesClass = Xs[:, 1:][Ys == 1];
	plt.title("Real Training Data");
	plt.plot(zerosClass[:, 0], zerosClass[:,1], 'bs');
	plt.plot(onesClass[:, 0], onesClass[:, 1], 'ro');
	plt.show();

def plotClassifiedTrainingData(Xs, classification):
	zerosClass = Xs[:, 1:][classification < .5]
	onesClass = Xs[:, 1:][classification >= .5]
	plt.title("Classified Training Data");
	plt.plot(zerosClass[:, 0], zerosClass[:,1], 'bs');
	plt.plot(onesClass[:, 0], onesClass[:, 1], 'ro');
	plt.show();

def plotRealTestData(Xs, Ys):
	zerosClass = Xs[Ys == 0];
	onesClass = Xs[Ys == 1];
	plt.title("Real Test Data");
	plt.plot(zerosClass[:, 0], zerosClass[:,1], 'bs');
	plt.plot(onesClass[:, 0], onesClass[:, 1], 'ro');
	plt.show();

def plotClassifiedTestData(Xs, classification):
	zerosClass = Xs[classification < .5]
	onesClass = Xs[classification >= .5]
	plt.title("Classified Test Data");
	plt.plot(zerosClass[:, 0], zerosClass[:,1], 'bs');
	plt.plot(onesClass[:, 0], onesClass[:, 1], 'ro');
	plt.show();

def plotCostHistory(Xs, Cost):
	plt.title("Cost History");
	plt.plot(Xs, Cost);
	plt.show();

Xs, Ys, XTest, YTest = data.genLogisticData();
Xs[:,1:] = data.featureNormalize(Xs[:, 1:]);
XTest = data.featureNormalize(XTest);

logistic = Logistic(Xs, Ys);
thetas = logistic.fit(numIters=10000 ,descentFactor=0.01 ,autoDescentAdjust=True, regularizationFactor=0);

print("Alpha History: ");print(logistic.getAlphaHistory());
print("thetas");print(thetas);

# Plot Training Data
plotRealTrainingData(Xs, Ys);
plotClassifiedTrainingData(Xs, logistic.computeHypothesis(Xs));
plotCostHistory(range(1,10001), logistic.getCostHistory());
# Plot Test Data
plotRealTestData(XTest, YTest);
plotClassifiedTestData(XTest, logistic.estimate(XTest));

