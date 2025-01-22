"""
Imports
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy import ndimage
from keras.datasets import mnist
from hilbert import decode
import scipy.cluster.hierarchy as spc
import argparse
import os, sys


"""
Parse Args
"""

parser = argparse.ArgumentParser()

# Network Structure
parser.add_argument('--size', type=int,
                    help='Number of oscillators in network')



# Argument parsing
args = parser.parse_args()
size = args.size

"""
Directory setup
"""
sim_dir = '/home/ayeung_umass_edu/nv-nets/results/%d-ring-dataset' % size

# Create the directory if necessary
if not os.path.isdir(sim_dir): os.makedirs(sim_dir)


"""
Define Coupled Oscillator Class
"""

class CoupledOscillator():
    def __init__(self, n_rings = 200, ring_sizes = (3,10), eps = 0.25, p = 0.2, n_in = 8, dt = 1e-2, T_sim  = 100, seed = -1, v_thl = 0.2, v_thh = 0.49, dv = 1e-2, dt_init = 1e-3):
        self.T_init_max = 2 * ring_sizes[1]
        # Network parameters
        self.ring_sizes = ring_sizes # (min, max) Number of neurons per ring
        self.n_rings = n_rings # Number of rings in the network
        self.eps = eps # Coupling constant
        self.p = p #Frequency of random connection for small worlds graph
        self.n_in = n_in #Size of the sliding bit window

        # Simulation parameters
        self.dt = dt # Simulation timestep
        self.T_sim = T_sim  # Simulation duration
        self.seed = seed # Random seed for the simulation

        # Neuron parameters
        self.v_thl = v_thl # Lower threshold of the schmidt trigger
        self.v_thh = v_thh # Upper threshold of the schmidt trigger

        self.dv = dv # Controls the randomness in the initial state
        self.dt_init = dt_init

        if self.seed == -1: self.seed = np.random.randint(1e6)
        print('Random Seed:', self.seed)
        np.random.seed(self.seed)

        # Generating the ring sizes
        self.R = np.random.choice(np.arange(ring_sizes[0], ring_sizes[1]+1), size=n_rings)
        unique, counts = np.unique(self.R, return_counts=True)
        print("Ring Sizes: " + str(dict(zip(unique, counts))))

        self.n_h = self.R.sum() # Total number of neurons
        # Idx associated with the first neuron in each ring
        self.nrn1_idx = np.insert(self.R[:-1].cumsum(), 0, 0)
        # Shifted ring indices
        self.shifted_idx = np.concatenate(
            [
                np.roll(np.arange(r_i) + self.nrn1_idx[i], 1)
                for (i,r_i) in enumerate(self.R)
            ]
        )

        #Watts Strogatz Small World Graph
        G = nx.watts_strogatz_graph(n = n_rings, k = 2, p = p)
        self.W = nx.to_numpy_array(G) * self.eps

        # Setup arrays to store the simulation history
        self.n_ts = int(T_sim / dt) + 1 # Number of timesteps in the simulation
        self.v_cap = np.empty((self.n_ts,self.n_h), dtype=float)
        self.v_out = np.empty((self.n_ts,self.n_h), dtype=float)


        #Generate adjacency matrix describing input (W_in)
        W_in = np.random.binomial(1, p, size = (n_rings, n_in))
        np.fill_diagonal(W_in, 0)

        W_in_expanded = np.zeros((self.n_h,n_in), dtype=int)
        # Iterate through each outgoing ring (i) and target ring (j) for the input
        for i in range(n_rings):
            for j in range(n_in):
                # Get the indices for the first neuron in the outgoing ring (i) and the second neuron in the target ring (j)
                first_neuron_idx_i = self.nrn1_idx[i]

                # Get the connection strength from W
                connection_strength = W_in[i, j]

                # Set the connection in W_comb
                if connection_strength > 0:
                    W_in_expanded[first_neuron_idx_i, j] = connection_strength
                    
        W_in_expanded = W_in_expanded * eps 

        self.W_in_sparse = csr_matrix(W_in_expanded)

        #Generate adjacency matrix describing both in-ring connections (W_rot) and combined small-world and
        #in-ring connections (W_comb)
        W_rot = np.zeros((self.n_h,self.n_h), dtype=int)
        for i in range(n_rings):
            W_rot[self.nrn1_idx[i] + np.arange(self.R[i]),self.nrn1_idx[i] + np.roll(np.arange(self.R[i]), 1)] = 1

        W_comb = np.zeros((self.n_h, self.n_h), dtype=float)
        W_comb = W_comb + W_rot
        # Iterate through each outgoing ring (i) and target ring (j)
        for i in range(n_rings):
            for j in range(n_rings):
                # Get the indices for the first neuron in the outgoing ring (i) and the second neuron in the target ring (j)
                first_neuron_idx_i = self.nrn1_idx[i]
                second_neuron_idx_j = self.nrn1_idx[j] + 1  # Second neuron in ring j

                # Get the connection strength from W
                connection_strength = self.W[i, j]

                # Set the connection in W_comb
                if connection_strength > 0:
                    W_comb[first_neuron_idx_i, second_neuron_idx_j] = connection_strength

        self.W_comb_sparse = csr_matrix(W_comb)


    #Helper Function for Clustering
    def hierarchical_clustering(self, H, alpha=0.7):
        """
        Standard hierarchical clustering, using scipy

        ARGUMENTS

            H       :   (# hidden states, # neurons)-numpy array representing
                        hidden states of the RNN

        RETURNS

            order   :   (# neurons,)-numpy array. order[i] denotes the index of
                        neuron i, based on the order imposed by the hierarchical
                        clustering
        """
        # Covariance matrix
        corr = (H.T @ H) / H.shape[0]

        # Pairwise distanced, based on neuron-neuron correlation vectors
        pdist = spc.distance.pdist(corr)
        # Hierarchical Clustering
        linkage = spc.linkage(pdist, method='complete')
        # Convert to cluster indices
        idx = spc.fcluster(linkage, alpha * pdist.max(), 'distance')
        order = np.argsort(idx)

        # Return the new neuron ordering
        return order
        
    # Schmidt trigger "Activation function"
    def f_out(self,o, v):
        return 1 - np.logical_or( v >= self.v_thh, np.logical_and(v >= self.v_thl, o == 0)  )


    def initialize(self):
        # Set the initial state
        init_phase = np.random.rand(self.n_rings)
        self.v_cap[0].fill(0.9)
        self.v_out[0].fill(1)

        # Randomly select a single node in each ring to be firing initially
        firing_nodes = np.floor(np.random.random(size=self.n_rings) * (self.R-1)).astype(int) + self.nrn1_idx

        # Set the selected nodes to have output 1 and random low capacitor values
        self.v_cap[0,firing_nodes] = 0.1 + self.dv * np.random.rand(self.n_rings)
        self.v_out[0,firing_nodes] = 0

        # Simulate each ring *on its own* for a random amount of time
        T_init = np.repeat( np.random.rand(self.n_rings) * self.T_init_max, self.R )
        alph_init = np.exp(-self.dt_init / T_init * self.T_init_max)
        n_ts_init = int(self.T_init_max / self.dt_init)
        for t in range(n_ts_init):
            # Update capacitors
            u = self.v_out[0,self.shifted_idx] 
            self.v_cap[0] = self.v_cap[0] * alph_init + u * (1-alph_init)
            # Update outputs
            self.v_out[0] = self.f_out(self.v_out[0], u - self.v_cap[0])


    #Simulates the network n_ts timesteps
    #Note: v_in must be of size (n_ts, n_in)
    def simulate(self, v_in):
        alph = np.exp(-self.dt)
        for t in range(self.n_ts-1):
            # Update capacitors
            # Input from *within* each ring
            #Matrix multiplication implementation
            
            #Vector describing inputs from in-ring and small-world connections
            u = self.W_comb_sparse @ self.v_out[t, :] + self.W_in_sparse @ v_in[t, :]
            #Update capacitor voltage
            self.v_cap[t+1] = alph * self.v_cap[t] + (1-alph) * u
            #Update outputs
            self.v_out[t+1] = self.f_out(self.v_out[t], u - self.v_cap[t+1])
        return self.v_out, self.v_cap
    
    def reset(self):
        self.v_cap = np.empty((self.n_ts,self.n_h), dtype=float)
        self.v_out = np.empty((self.n_ts,self.n_h), dtype=float)

"""
Generate input: MNIST Data according to Hilbert Curve
Creates v_in
"""
(train_X, train_y), (test_X, test_y) = mnist.load_data()

def create_v_in(mnist_digit, n_in, n_ts):
    mnist_digit = ndimage.zoom(mnist_digit, 32/28, mode = 'constant')
    mnist_digit = mnist_digit * 0.01
    N = mnist_digit.shape[0]
    #Add some noise
    mnist_digit += 0.001 * np.random.randn(*mnist_digit.shape)
    # Hilbert curve positions
    logN = int(np.log2(N))
    locs = decode(np.arange(N**2), 2, logN)

    # Creating the sliding window embedding
    sw_dim = n_in  # Sliding window dimension
    sw = np.zeros((N**2 - sw_dim + 1,sw_dim))
    # Traversal of the image A along the Hilbert curve
    A_hilbert = mnist_digit[locs[:,0], locs[:,1]]
    # Sliding window of the Hilbert curve traversal
    for i in range(sw_dim):
        sw[:,sw_dim-i-1] = A_hilbert[i:][:sw.shape[0]]


    n, d  = sw.shape
    ratio = n_ts // n
    remainder = n_ts % n

    # Repeat each entry of sw n_in // n times
    sw_repeated = np.repeat(sw, ratio, axis=0)

    # Repeat the beginning sequences of sw to fill the remaining elements
    if remainder > 0:
        sw_remainder = sw_repeated[:remainder]
        sw_repeated = np.vstack((sw_repeated, sw_remainder))

    return sw_repeated, locs

"""
Run simulation over training imgs to create
training set for classifier
"""
train_dataset = []
n = CoupledOscillator(n_rings=size)
for img in train_X:
    n.initialize()
    v_in, locs = create_v_in(img, n.n_in, n.n_ts)
    outs, caps = n.simulate(v_in)
    v_os = outs[::n.n_ts//n.n_in]
    v_os = v_os.reshape(-1) 
    train_dataset.append(v_os)
    n.reset()

# Post simulation

# Save the final state of the network
train_dir = '%s/train' % sim_dir
if not os.path.isdir(train_dir): os.makedirs(train_dir)
s = f'{train_dir}/train_dataset_{n.n_rings}rings{n.seed}'
train_dataset = np.array(train_dataset)
np.savez_compressed(s, train_dataset)




test_dataset = []
for img in test_X:
    n.initialize()
    v_in, locs = create_v_in(img, n.n_in, n.n_ts)
    outs, caps = n.simulate(v_in)
    v_os = outs[::n.n_ts//n.n_in]
    v_os = v_os.reshape(-1) 
    test_dataset.append(v_os)
    n.reset()

# Save the final state of the network
test_dir = '%s/test' % sim_dir
if not os.path.isdir(test_dir): os.makedirs(test_dir)
s = f'{test_dir}/test_dataset_{n.n_rings}rings{n.seed}'
test_dataset = np.array(test_dataset)
np.savez_compressed(s, test_dataset)