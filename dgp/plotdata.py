#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import axes3d

bL=2

def weekday(x):
	if(x%5==0 or x%6==0):
		return 0.5
	else:
		return 1
		

def main():
	x=np.linspace(0,4,1000)
	y=np.arange(0,30,1)
	X,Y=np.meshgrid(x,y)
	vweek=np.vectorize(weekday)
	Z=np.exp(vweek(Y)*X*(-bL))
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	ax.set_xlabel("Price")
	ax.set_ylabel("Weekday")
	ax.set_zlabel("Demand")
	ax.plot_surface(X,Y,Z,cmap="autumn_r")
	plt.show()

if __name__=="__main__":
	main()
