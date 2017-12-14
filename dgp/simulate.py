#!/usr/bin/env python
#generate simulated data

import numpy as np
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import axes3d

bL=2
alpha=0.05
beta=0.05

def weekday(x):
	if(x%5==0 or x%6==0):
		return 0.5
	else:
		return 1

def h(p,w,b):
	return np.exp(-beta*pow(w,b)*pow(1-w,1-b)*pow(p,beta))

def g(f,w,b):
	return np.exp(-alpha*pow(w,b)*pow(1-w,1-b)*pow(f,alpha))

def rand_samp:
	w=np.sin(np.pi*np.random.uniform())
	b=np.random.randint(2)
	f=np.gamma(7.5,1)
	P=g(f,w,b)+np.gamma(5,1)
	D=h(P,w,b)+np.gamma(5,1)+np.true_divide(P,10)
	return(P,D,w,f,b)

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
