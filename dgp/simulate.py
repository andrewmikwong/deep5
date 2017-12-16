#!/usr/bin/env python
#generate simulated data

from __future__ import print_function

import warnings

from deepiv.models import Treatment, Response
import deepiv.architectures as architectures
import deepiv.densities as densities

import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.merge import Concatenate

import data_generator
import numpy as np
import matplotlib.pyplot as plt
import pdb

from mpl_toolkits.mplot3d import axes3d

n = 5000
dropout_rate = min(1000./(1000. + n), 0.5)
epochs = int(1500000./float(n)) # heuristic to select number of epochs
epochs = 30#300
batch_size = 100
images = False

bL=2
alpha=0.5
beta=0.5

def true_demand(x,z,p):
	exponent=-beta*np.power(x[:,0],x[:,1])*np.power(1-x[:,0],1-x[:,1])*(p.flatten()**beta)
	res= np.exp(exponent)+np.true_divide(p.flatten(),10)+np.random.gamma(5,1,(x.shape[0],1)).flatten()
	return res


def h(p,w,b):
	return np.exp(-beta*pow(w,b)*pow(1-w,1-b)*pow(p,beta))

def g(f,w,b):
	return np.exp(-alpha*pow(w,b)*pow(1-w,1-b)*pow(f,alpha))

def rand_samp():
	w=np.sin(np.pi*np.random.uniform())
	b=np.random.randint(2)
	f=np.random.gamma(7.5,1)
	P=g(f,w,b)+np.random.gamma(5,1)
	D=h(P,w,b)+np.random.gamma(5,1)+np.true_divide(P,10)
	return P,D,w,f,b

def datafunction(n,s,images=images,test=False):
	x=[]
	z=[]
	t=[]
	y=[]
	for i in range(n):
		P,D,w,f,b=rand_samp()
		x.append([w,b])
		z.append([f])
		t.append([P])
		y.append([D])
	x=np.array(x)
	z=np.array(z)
	t=np.array(t)
	y=np.array(y)
	return x,z,t,y, true_demand


#def datafunction(n, s, images=images, test=False):
#    return data_generator.demand(n=n, seed=s, ypcor=0.5, use_images=images, test=test)

x, z, t, y, g_true = datafunction(n, 1)

print("Data shapes:\n\
Features:{x},\n\
Instruments:{z},\n\
Treament:{t},\n\
Response:{y}".format(**{'x':x.shape, 'z':z.shape,
                        't':t.shape, 'y':y.shape}))
pdb.set_trace()

# Build and fit treatment model
instruments = Input(shape=(z.shape[1],), name="instruments")
features = Input(shape=(x.shape[1],), name="features")
treatment_input = Concatenate(axis=1)([instruments, features])

hidden = [128, 64, 32]

act = "relu"

est_treat = architectures.feed_forward_net(treatment_input, lambda x: densities.mixture_of_gaussian_output(x, 10),
                                           hidden_layers=hidden,
                                           dropout_rate=dropout_rate, l2=0.0001,
                                           activations=act)

treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
treatment_model.compile('adam',
                        loss="mixture_of_gaussians",
                        n_components=10)

treatment_model.fit([z, x], t, epochs=epochs, batch_size=batch_size)

# Build and fit response model

treatment = Input(shape=(t.shape[1],), name="treatment")
response_input = Concatenate(axis=1)([features, treatment])

est_response = architectures.feed_forward_net(response_input, Dense(1),
                                              activations=act,
                                              hidden_layers=hidden,
                                              l2=0.001,
                                              dropout_rate=dropout_rate)

response_model = Response(treatment=treatment_model,
                          inputs=[features, treatment],
                          outputs=est_response)
response_model.compile('adam', loss='mse')
response_model.fit([z, x], y, epochs=epochs, verbose=1,
                   batch_size=batch_size, samples_per_batch=2)

oos_perf = data_generator.monte_carlo_error(lambda x,z,t: response_model.predict([x,t]), datafunction, has_latent=images, debug=False)
print("Out of sample performance evaluated against the true function: %f" % oos_perf)
pdb.set_trace()
