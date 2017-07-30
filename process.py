"""
Usage:  python process.py <coupling_strength> <slope_parameter> <label>
  where <coupling_strength> globally scales the coupling,
        <slope_parameter> determines the slope of the sigmoid, and
        <label> is an integer to label this parameter set.
"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy import signal
import scipy.io as sio
import numpy as np
import numpy.lib.scimath as sm
import sys
import os
import errno
import glob

datadir = './datafiles'
imgdir = './img'

def load_mat(filename):
    """load array from a MATLAB .mat file"""
    return sio.loadmat(filename)['data']

def sigmoid(x, a, midpoint):
    return 1.0/(1.0 + np.exp(-a*(x-midpoint)))

def siggain(x, a, midpoint):
    """first derivative of sigmoid"""
    return a/(2.0 + 2.0*np.cosh(a*(x-midpoint)))

def inverse_sigmoid(phi, gain, midpoint):
    return midpoint - np.log(1.0/phi - 1.0)/gain
   
def hilbert_phase(ts, axis=0):
    """Phase of the analytic signal, using the Hilbert transform"""
    return np.angle(signal.hilbert(signal.detrend(ts, axis=axis), axis=axis))

def mod2pi(ts):
    """For a timeseries where all variables represent phases (in radians),
    return an equivalent timeseries where all values are in the range (-pi, pi]
    """
    return np.pi - np.mod(np.pi - ts, 2*np.pi)

def order_param(ts, axis=1):
    """Order parameter of phase synchronization.
    ts is assumed to hold phase values in radians."""
    return np.abs(np.exp(1.0j * ts).mean(axis=axis))
 
def phase_entropy(ts, bins=20):
    """Entropy of phase distrib estimated by partition of (-pi,pi] into m bins.
    ts is assumed to be a 1D vector of phase values in the range (-pi, pi]."""
    n = ts.shape[0]
    (h, _) = np.histogram(ts, bins, range=(-np.pi, np.pi), density=False)
    p = np.true_divide(h, n)
    q = p.copy()
    q[h==0] = 1.0 # For empty bins, contribution to the sum should be zero.
    return -np.sum(p*np.log(q))

_third = np.float64(1)/3
_ca = (np.float64(686)/3)**_third
_cb = np.float64(18)**_third
# Equilibrium V, as a function of local input I.
def Veq(I):
    s = np.power(72 - 9*I + np.sqrt(9300.0 + 81*I**2 -1296*I), _third)
    return 1.0 + _ca/s - s/_cb

# Jacobian first eigenvalue, as a function of equilibrium V.
def eig(V):
    return -10 + 60*V -30*V**2 + 10*sm.sqrt(-39+12*V +30*V**2 -36*V**3 +9*V**4)

# Make dirs without complaining if they already exist
def mkdirp(path):
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

gcs=float(sys.argv[1]) # global coupling strength
slope=float(sys.argv[2]) # sigmoid slope
label=int(sys.argv[3]) # parameter set label
midpoint = 1.5 # horizontal shift of sigmoid

# Create subdirectories for output
for d in ['meancoup', 'meangains', 'meansigin', 'meaninstab', 'meanop',
          'meanpe']:
    mkdirp(datadir + '/' + d)
for d in ['corrgain_a', 'corrgain_g', 'coup', 'gain', 'orbit', 'sig', 'instab',
          'sigin']:
    mkdirp(imgdir + '/' + d)

filenames = glob.glob(datadir + '/raw/Raw_SimplePreSigmoidal_%06f_%06f_*.mat' %
        ( gcs, slope))

# First plot orbits
def plot_orbit(ar0, ar1, num, square=False, color='blue', alpha=0.01):
    """draw a translucent plot of the orbit"""
    index_partition = np.linspace(0, ar0.shape[0], num=num, dtype=int)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    for k in range(num - 1):
        i0 = index_partition[k]
        i1 = index_partition[k+1]
        ax.plot(ar0[i0:i1], ar1[i0:i1], color=color, alpha=alpha)
    (xmin, xmax) = (ar0.min(), ar0.max())
    (ymin, ymax) = (ar1.min(), ar1.max())
    if square:
        width = max(xmax - xmin, ymax - ymin)
        xpad = max(0.0, (width - (xmax - xmin))/2.0)
        ypad = max(0.0, (width - (ymax - ymin))/2.0)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    return fig

# Plotting orbits from first simulation only:
firstarray = load_mat(filenames[0])
# take a slice of 20 seconds from the middle of the 10 minute simulation
# with oscillations of ~10Hz that is about 200 cycles of each node
start = 5*60*2000
end = start + 20*2000
# append time series of all 76 nodes and plot together
toplot = np.vstack([firstarray[start:end,:,i] for i in range(76)])
fig00 = plot_orbit(toplot[:,0], toplot[:,1], num=50*76, alpha=0.006)
# find means
mean_V = firstarray[:,0,:].mean(axis=0)
mean_W = firstarray[:,1,:].mean(axis=0)
plt.scatter(mean_V, mean_W, c='red', alpha=0.1)
ax = fig00.axes[0]
ax.set_xlim(-2.0, 4.5)
ax.set_ylim(-35.0, 2.0)
ax.set_xlabel('V')
ax.set_ylabel('W')
ax.set_title('Orbits of all 76 nodes, $\\gamma$=%.02f, a=%.03f' % (gcs,slope))
fig00.savefig(imgdir + '/orbit/img' + str(label).zfill(4) +'.png', dpi=100)
plt.close()
del firstarray, toplot

# Estimate synchrony
print('Processing %d files for gcs %f, slope %f' % (
	len(filenames), gcs, slope))
# concatenate all trials, V variable only.
arrays = [load_mat(fn)[:,0,:] for fn in filenames]
allv = np.concatenate(arrays, axis=0) # shape (nreps*ntimepoints, nregions)
del arrays
allphase = hilbert_phase(allv)
mean_op = order_param(allphase, axis=1).mean()
ntimepoints = allphase.shape[0] # across all repetitions concatenated
pe = np.ones(ntimepoints) * np.nan
for i in range(ntimepoints):
    pe[i] = phase_entropy(allphase[i,:], bins=19)
mean_pe = pe.mean()
del allphase
np.savetxt(datadir + '/meanop/meanop_SimplePreSigmoidal_%06f_%06f.txt' % (
        gcs, slope), np.array([mean_op]))
np.savetxt(datadir + '/meanpe/meanpe_SimplePreSigmoidal_%06f_%06f.txt' % (
        gcs, slope), np.array([mean_pe]))

# Get mean gain for each node.
allgain = siggain(allv, slope, midpoint)
meangains = allgain.mean(axis=0) # mean over time only. shape (nregions,)
np.savetxt(datadir + '/meangains/meangains_SimplePreSigmoidal_%06f_%06f.txt' % (
        gcs, slope), meangains)
meangain = meangains.mean() # mean over all time and all regions.

# Now plot histograms.
# All trials and all nodes concatenated, V variable only.
allsigin = np.ravel(allv)
meansigin = allsigin.mean()
(min, max) = (-20.0, 20.0)

# sigmoid input distribution juxtaposed with sigmoid and gain curves
fig0 = plt.figure(figsize=(16,9))
# plot sigmoid and its derivative
domain = np.linspace(min, max, num=100)
sigvals = sigmoid(domain, slope, midpoint)
siggainvals = siggain(domain, slope, midpoint)
print('For gcs=%.02f a=%.03f, mean gain = %.03g' % (gcs, slope, meangain))
plt.plot(domain, sigvals, 'b-', linewidth=2.0, label="Sigmoid $S(V)$")
plt.plot(domain, siggainvals*4.5, color='magenta', alpha=0.2,
	 label="gain $S'(V)$ (rescaled)")
w = np.repeat(0.2*(max - min)/allsigin.size, len(allsigin))
plt.hist(allsigin, normed=False, bins=300, range=(min, max), weights=w,
	 histtype='stepfilled', color='grey', alpha=0.3,
	 label="distribution $p(V)$")
ax = fig0.axes[0]
plt.text(0.01, 0.70, ("For $\\gamma$=%.02f,a=%.03f, mean gain "%(gcs,slope)) +
	 "$\int_{-\infty}^{\infty} p(V) S'(V)dV$ = %.03g" %
	 meangain, ha='left', va='bottom', transform=ax.transAxes)
plt.legend(loc='upper left')
plt.title('distribution of input $V$ to Sigmoidal, $\\gamma$=%.02f, a=%.03f' %
        (gcs, slope))
plt.xlabel("sigmoid input $V$")
plt.xlim(min, max)
plt.ylim(0.0, 1.0)
fig0.savefig(imgdir + '/sig/img' + str(label).zfill(4) +'.png', dpi=100)
plt.close()

# sigmoid input distribution
fig1 = plt.figure(figsize=(16,9))
plt.hist(allsigin, normed=True, bins=300, range=(min, max), 
         histtype='stepfilled', color='grey', alpha=0.3)
plt.title('distribution of Sigmoidal input, $\\gamma$=%.02f, a=%.03f' % (
	gcs, slope))
plt.xlabel("sigmoid input $V$")
plt.xlim(min, max)
plt.ylim(0.0, 15.0/(max - min))
fig1.savefig(imgdir + '/sigin/img' + str(label).zfill(4) +'.png', dpi=100)
plt.close()
np.savetxt(datadir + '/meansigin/meansigin_SimplePreSigmoidal_%06f_%06f.txt' % (
        gcs, slope), np.array([meansigin]))
del allsigin

# sigmoid gain distribution
(min, max) = (0.0, 0.30)
fig2 = plt.figure(figsize=(16,9))
plt.hist(allgain.ravel(), normed=True, bins=300, range=(min, max),
	 histtype='stepfilled', color='magenta', alpha=0.3)
plt.title('distribution of Sigmoidal gain , $\\gamma$=%.02f, a=%.03f' % (
	gcs, slope))
plt.xlabel("sigmoid gain $S'(V)$")
plt.xlim(min, max)
plt.ylim(0.0, 80.0/(max - min))
fig2.savefig(imgdir + '/gain/img' + str(label).zfill(4) +'.png', dpi=100)
plt.close()
del allgain

# retain only the first simulation for each parameter set
for fn in filenames[1:]:
    os.unlink(fn)

# TEMP: Now do the same for coupling input
filenames = glob.glob('datafiles/coupling/coup_SimplePreSigmoidal_%06f_%06f_*.mat' % (gcs, slope))
print('Processing %d files for coupling, gcs %f, slope %f' % (
	len(filenames), gcs, slope))
# concatenate all trials and all nodes together
arrays = [load_mat(fn).ravel() for fn in filenames]
allcoup = gcs * np.concatenate(arrays, axis=0)
del arrays

# coupling distribution
meancoup = allcoup.mean()
#(min, max) = (allcoup.min(), allcoup.max())
(min, max) = (0.0, 60.0)
fig3 = plt.figure(figsize=(16,9))
plt.hist(allcoup, normed=True, bins=300, range=(min, max), 
         histtype='stepfilled', color='grey', alpha=0.3)
plt.title('distribution of coupling input, $\\gamma$=%.02f, a=%.03f' % (
	gcs, slope))
plt.xlabel("coupling input $\\gamma I_{net}$")
plt.xlim(min, max)
#plt.ylim(0.0, 15.0/(max - min))
plt.ylim(0.0, 0.4)
fig3.savefig(imgdir + '/coup/img' + str(label).zfill(4) +'.png', dpi=100)
plt.close()
np.savetxt('datafiles/meancoup/meancoup_SimplePreSigmoidal_%06f_%06f.txt' % (
        gcs, slope), np.array([meancoup]))

# distribution of instability measure
allinstab = np.reciprocal(np.abs(np.real(eig(Veq(allcoup)))))
del allcoup
meaninstab = allinstab.mean()
(min, max) = (0.0, 100.0)
fig4 = plt.figure(figsize=(16,9))
plt.hist(allinstab, normed=True, bins=300, range=(min, max), 
         histtype='stepfilled', color='blueviolet', alpha=0.3)
plt.title('distribution of $|Re(eig)|^{-1}$, $\\gamma$=%.02f, a=%.03f' % (
	gcs, slope))
plt.xlabel("$|Re(eig)|^{-1}$ at equilibrium point")
plt.xlim(min, max)
#plt.ylim(0.0, 15.0/(max - min))
plt.ylim(0.0, 0.4)
fig4.savefig(imgdir + '/instab/img' + str(label).zfill(4) +'.png', dpi=100)
plt.close()
np.savetxt('datafiles/meaninstab/meaninstab_SimplePreSigmoidal_%06f_%06f.txt' % (
        gcs, slope), np.array([meaninstab]))
del allinstab

# retain only the first simulation for each parameter set
for fn in filenames[1:]:
    os.unlink(fn)
