#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import axes3d

beta=0.5


def main():
	#leisure?
	x=np.linspace(0,100,1000)
	y=np.linspace(0,1,100)
	X,Y=np.meshgrid(x,y)
	Z=np.exp(Y*(-beta)*(X**beta))
	fig=plt.figure()
	ax=fig.add_subplot(211,projection='3d')
	ax.set_xlabel("Price")
	ax.set_ylabel("Weekday")
	ax.set_zlabel("Demand")
	ax.plot_surface(X,Y,Z,cmap="autumn_r")

	#business?
	ax2=fig.add_subplot(212,projection='3d')
	ax2.set_xlabel("Price")
	ax2.set_ylabel("Weekday")
	ax2.set_zlabel("Demand")
	Z=np.exp((1-Y)*(-beta)*(X**beta))
	ax2.plot_surface(X,Y,Z,cmap="autumn_r")

	plt.show()

if __name__=="__main__":
	main()
