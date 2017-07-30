"""
TVB model for resting state fMRI
based on http://nbviewer.jupyter.org/url/docs.thevirtualbrain.org/tutorials/tutorial_s5_ModelingRestingStateNetworks.ipynb

Usage:  python model_PreSigmoidal.py <gain_parameter> <label>
  where <gain_parameter> determines the slope of the sigmoid, and
        <label> is a number to identify individual trials.
"""

from tvb.simulator.lab import (
    simulator, connectivity, models, coupling, integrators, monitors, noise)
import scipy.io as sio
import sys
import numpy

# Get master seed from /dev/urandom. Report it, for reproducibility if desired.
rs = numpy.random.RandomState(None)
master_seed = rs.randint(2**32)
print('Using initial seed %d.' % master_seed)
rs.seed(master_seed)

def save_mat(ar, filename):
    """save array to a MATLAB .mat file"""
    mat_dict = {'data': ar}
    sio.savemat(filename, mat_dict, do_compression=True)
    return

def run_sim(conn, gain, D, dt=0.5, simlen=1e3):
    """
    Run a single instance of the simulation.
    Returns a list of pairs (times, values). There will be one entry
    in the list for each output monitor used.
    """
    sim = simulator.Simulator(
        model=models.Generic2dOscillator(a=0.0),
        connectivity=conn,  # use the connectivity structure defined below
        coupling=coupling.PreSigmoidal(G=gain*0.5,theta=0.0,dynamic=False),
        integrator=integrators.HeunStochastic(dt=dt,
                           noise=noise.Additive(nsig=numpy.array([D]))),
##      monitors=[monitors.Bold(period=1000.0), monitors.Raw()]
        monitors=[monitors.Bold(period=1000.0)]
    )
    sim.configure()
    # Give each sim instance a different random state
    resultgen = sim(simulation_length=simlen, random_state=rs.get_state())
    nmon = len(sim.monitors)
    times = [[] for i in range(nmon)]
    values = [[] for i in range(nmon)]
    tempres = resultgen.next()
    for r in resultgen:
        for i in range(nmon):
            if r[i] is not None:
                times[i].append(r[i][0])
                values[i].append(r[i][1])
    for i in range(nmon):
        times[i] = numpy.array(times[i])
        values[i] = numpy.concatenate(numpy.expand_dims(values[i], 0))
    return zip(times, values)
    

# this is the connectivity structure based on cocomac
conn = connectivity.Connectivity.from_file('/home/matthewa/tvb/tvb-data/tvb_data/connectivity/connectivity_76.zip')
conn.configure()

simlen=660000 # 11 mins, expressed in ms - 10 mins, plus one for burnin
burnin=60 # expressed in seconds

# using the gain parameter specified on the command line,
# run one instance of the simulation
# the label is there simply to allow running multiple of these in parallel
# without clobbering each other's output files
gain=float(sys.argv[1])
label=int(sys.argv[2])

print('Running model for coupling strength %f' % gain)
all_results = run_sim(conn, gain, 5e-4, simlen=simlen)

bold_timepoints = all_results[0][0]
bold = all_results[0][1]
##raw_timepoints = all_results[1][0]
##raw = all_results[1][1]

# reshape the data which are output with some empty dimensions
bold = bold[:,0,:,0]
##raw = raw[:,:,:,0]

simdata=bold[burnin:,:] # remove the burnin period

# export MATLAB files
save_mat(simdata, 'datafiles/BOLD/BOLD_PreSigmoidal_%06f_%05d.mat' % (gain, label))
##save_mat(raw[(burnin*1000):,:], 'datafiles/Raw_%06f_%05d.mat' % (gain, label))


# compute the correlation between the model's connectome
# and the weighted strucural backbone
cc=numpy.corrcoef(simdata.T)
sw=conn.scaled_weights()
simcorr=numpy.corrcoef(cc.ravel(),sw.ravel())[0,1]

print((label,gain,simcorr))
numpy.savetxt('datafiles/corr/simcorr_PreSigmoidal_%06f_%05d.txt'%(gain,label),numpy.array([simcorr]))
numpy.savetxt('datafiles/txt/BOLD_PreSigmoidal_%06f_%05d.txt'%(gain,label),simdata)
