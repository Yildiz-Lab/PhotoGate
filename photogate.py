# Collection of functions for simulating photogate experiments

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import datetime
import os
import csv
import pickle
from matplotlib.colors import LogNorm

### Class definitions #########################################################

class DiffusionSimulation(object):
    """A single square cell filled with fluorescent particles diffusing in 2D. 

    Attributes:
        particles (numpy.array of int): N by 3 array storing positions and 
            fluorescence bleaching states of all simulated particles. First
            and second column are X and Y values, and third column is number of
            remaining unbleached fluorophores.
        tau (float): Time step, in s, of the simulation.
        grid_delta (float): Grid spacing (unitary step size), in um.   
        grid_num_el (int): Number of discrete steps covering the grid.
        steps_per_frame (int): Number of simulation steps per frame saved
            to file.
        
        _escaping_bounds (numpy.array bool): N by 4 array storing information
            about whether each particle attempted to escape the bounds of the
            cell in the last iteration. The columns represent particles 
            escaping to the left, right, down, and up, respect.
        _tracking (numpy.array of int): N by 2 array (where N is the number of 
            particles) for storing the tracking history of particles. The first
            column contains the starting frame of the most recent tracking
            period of the particle's tracking, and the second column contains
            the oligomeric state of the particle at the start of tracking.
        _tracking_times (list): individual particle tracking times, terminated 
            by either bleaching or escaping from tracking zone.
    
        **kwargs: 
            grid_size (float): Width, in um, of the simulated cell. 
                Defaults to 40.
            num_particles (int): Number of particles to simulate. 
                Defaults to 80000. 
            oligo_state (int): Fluorescent oligomeric state of each particle. 
                Defaults to 2.
            diffusion_const(float): Diffusion constant, in um^2/s. 
                Defaults to 0.1.
            frame_rate (float): Frequency, in Hz, with which snapshots of the
                simulation are saved to file. Defaults to 10.
            sim_time (float): Time, in seconds, to run the simulation. 
                Defaults to 80. 
            num_time_steps(int): Number of time steps in simulation. 
                Defaults to 5000.
            tracking_roi (float): radius, in um, of the ROI used for finding
                trackable molecules. Defaults to 6.5.
            track_min_distance(float): minimum distance, in um, between two 
                molecules before they're indistinguishable. Defaults to 0.5.
    """
    
    def __init__(self, **kwargs):
        defaults = {
            'grid_size' : 40,
            'num_particles' : 80000, 
            'oligo_state' : 2, 
            'diffusion_const' : 0.1,
            'frame_rate' : 10,
            'sim_time' : 80, 
            'num_time_steps' : 5000,
            'tracking_roi' : 6.5,
            'track_min_distance' : 0.5
        }

        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        
        self.tau = self.sim_time / self.num_time_steps
        self.steps_per_frame = int(1 / (self.frame_rate * self.tau)) 
        self.grid_delta = np.sqrt(4 * self.diffusion_const * self.tau) 
        # ensure grid_num_el is a multiple of 2:
        self.grid_num_el = int(self.grid_size / (2 * self.grid_delta)) * 2      
        
        # Initialize particles
        self.particles = np.zeros((self.num_particles, 3), dtype = int)
        self.particles[:,0:2] = np.round( \
            np.random.rand(self.num_particles, 2) * (self.grid_num_el - 1))
        self.particles[:,2:3] = np.ones((self.num_particles,1)) * \
            int(self.oligo_state)
            
        # Initialize tracking variables
        self._tracking = np.zeros((self.num_particles,2), dtype = int)
        self._tracking_times = []
        
        # initialize _escaping_bounds:
        self._escaping_bounds = np.zeros((self.num_particles,4), dtype = bool)
    
    def calc_square_radii(self):
        """ Return the squared r (distance from ROI center) of particles"""
        box_offset = int(self.grid_num_el / 2)    
        particle_radii_sq = ((self.particles[:,0] - box_offset) \
        * self.grid_delta) ** 2 + ((self.particles[:,1] - box_offset) \
        * self.grid_delta) ** 2
        
        return particle_radii_sq
    
    def get_unbleached_particles(self, target_radius=None):
        """ Return the indices of all unbleached particles within the ROI.
        
        If target_radius is not explicitly set, uses self.tracking_roi
        
         Returns:
            indices (array of int64): the row indices of all particles in 
                self.particles that are within tracking_roi of the cell
                center and contain at least one unbleached fluorophore
        """
        if target_radius is None:
            target_radius = self.tracking_roi
        
        r_sq = target_radius ** 2
        particle_radii_sq = self.calc_square_radii()
        unbleached = np.logical_and(particle_radii_sq < r_sq,
                                    self.particles[:,2] > 0)
        indices = np.flatnonzero(unbleached)
        
        return indices
                           
    def unbleached_particle_density(self):
        """ Return density of unbleached fluorophores within tracking_roi.
        
        Returns:
            density_unbleached (float): density, in um^-2, of particles within
                tracking_roi that have at least one unbleached fluorophore            
            percent_full (float): percentage of fully unbleached particles
                inside the ROI. Returns -100 if all particles in ROI are
                completely bleached.
        """
        area = np.pi * (self.tracking_roi ** 2)  
        indices = self.get_unbleached_particles()        
        
        oligo_state_in_target = self.particles[indices, 2] 
        num_unbleached = np.sum(oligo_state_in_target > 0)
        density_unbleached = num_unbleached / area
        part_unbleached = oligo_state_in_target[oligo_state_in_target > 0]
        unbleached = part_unbleached[part_unbleached == self.oligo_state]
                                                       
        if len(part_unbleached) > 0:
            percent_full = (len(unbleached) / len(part_unbleached)) * 100
        else:
            # percent of fully unbleached particles is undefined
            percent_full = -100        
        
        return density_unbleached, percent_full
    
    def get_trackable(self):
        """ Return indices of trackable fluorophores inside tracking_roi.
        
        Returns:
            indices (array of int64): the row indices of unbleached particles 
                in self.particles that are within tracking_roi of the cell
                center and are at least min_distance away from the closest
                other unbleached particle, making them unambigously trackable.
        """
        candidate_ind = self.get_unbleached_particles()

        # Since all candidate particles are within tracking_roi, all potential
        # interfering particles will be within min_distance + tracking_roi
        interfere_radius = self.tracking_roi + self.track_min_distance
        interfere_ind = self.get_unbleached_particles(interfere_radius)
        interfere_coords = self.particles[interfere_ind, 0:2]

        if interfere_coords.size > 0:
            # Build a matrix of pairwise square distances between particles
            x1, x2 = np.meshgrid(interfere_coords[:,0], interfere_coords[:,0])
            y1, y2 = np.meshgrid(interfere_coords[:,1], interfere_coords[:,1])
            pairwise_dist_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            
            min_distance_grid = self.track_min_distance / self.grid_delta
            untrackable_pairs = pairwise_dist_sq < min_distance_grid ** 2

            # exclude the zero distances between the same particles:
            np.fill_diagonal(untrackable_pairs, False)

            trackable = np.invert(np.any(untrackable_pairs, axis=0))

            indices = np.intersect1d(interfere_ind[trackable], candidate_ind)
        
        # if no candidate particles available, return an empty array:
        else:
            indices = np.array([])
        
        return indices
    
    def move_particles(self):
        """Perform one step of the simulation, moving each particle."""
        
        # randomly move every particle along x and y:
        rand_n = np.random.rand(self.num_particles,2)
        left = rand_n[:,0] < 0.5
        right = rand_n[:,0] >= 0.5
        down = rand_n[:,1] < 0.5
        up = rand_n[:,1] >= 0.5
          
        self.particles[left,0] = self.particles[left,0] - 1
        self.particles[right,0] = self.particles[right,0] + 1
        self.particles[down,1] = self.particles[down,1] - 1
        self.particles[up,1] = self.particles[up,1] + 1
        
        # determine which particles tried to go out of bounds
        self._escaping_bounds[:,:] = False
        self._escaping_bounds[left,0] = self.particles[left,0] < 0
        self._escaping_bounds[right,1] = \
            self.particles[right,0] > (self.grid_num_el - 1)
        self._escaping_bounds[down,2] = self.particles[down,1] < 0
        self._escaping_bounds[up,3] = \
            self.particles[up,1] > (self.grid_num_el - 1)
            
        # bounce particles that tried to go out of bounds
        self.particles[self._escaping_bounds[:,0],0] = \
             self.particles[self._escaping_bounds[:,0],0] + 2
        self.particles[self._escaping_bounds[:,1],0] = \
             self.particles[self._escaping_bounds[:,1],0] - 2
        self.particles[self._escaping_bounds[:,2],1] = \
            self.particles[self._escaping_bounds[:,2],1] + 2
        self.particles[self._escaping_bounds[:,3],1] = \
            self.particles[self._escaping_bounds[:,3],1] - 2
 
    def bleach_particles(self, p_bleach_inv, p_bleach_inv_tirf, 
                         gate_or_tirf='gate'):
        """ Perform one step of Monte Carlo photobleaching simulation."""                  
     
        particle_radii_sq = self.calc_square_radii()
        
        if gate_or_tirf == 'gate': # ring-shaped bleaching beam    
            num_int = p_bleach_inv.shape[0]
            bleach_indices = np.searchsorted(p_bleach_inv[:num_int-1,2], 
                                            particle_radii_sq)
            p_bleach_per_particle = (p_bleach_inv[bleach_indices, 0]).reshape(\
                                    self.num_particles,1)
    
        else: # TIRF case: circular bleaching beam
            p_bleach_per_particle = np.ones([self.num_particles,1]) * \
                                                p_bleach_inv_tirf['p_inv']
            r_sq = p_bleach_inv_tirf['r_tirf'] ** 2                                           
            p_bleach_per_particle[particle_radii_sq > r_sq] = 1
    
        # get random numbers to determine bleaching of each fluorophore
        rand_p = np.random.rand(self.num_particles, self.oligo_state)
        bleaching_this_frame = rand_p>p_bleach_per_particle
        cum_bleaching = np.sum(bleaching_this_frame, axis=1)
        
        # bleach particles that pass the cutoff criterion:
        self.particles[:,2] = np.maximum(self.particles[:,2] - cum_bleaching, 
                                      np.zeros_like(cum_bleaching))
    
    def update_tracking_history(self, curr_sim_iteration):
        """ Find trackable particles in current frame and update history.
        
        """
        
        # only execute this function if current iteration is a frame multiple
        new_frame = (curr_sim_iteration % self.steps_per_frame == 0)   
        if curr_sim_iteration == 0 or not new_frame:
            return
        
        curr_frame = int(curr_sim_iteration / self.steps_per_frame)
        prev_tracking = self._tracking

        # get indices of currently and previously trackable particles
        curr_trackable = self.get_trackable()
        prev_trackable = np.flatnonzero(prev_tracking[:,0])

        # find indices of particles that appeared or disappeared in this frame        
        new = np.setdiff1d(curr_trackable, prev_trackable)
        gone = np.setdiff1d(prev_trackable, curr_trackable)
        bleached = gone[self.particles[gone, 2] == 0]

        for i in new:
            self._tracking[i,0] = curr_frame
            self._tracking[i,1] = self.particles[i,2]

        for i in gone:
            frames_tracked = (curr_frame - self._tracking[i,0]) 
            time_tracked = frames_tracked * self.tau * self.steps_per_frame
            bleach_steps = self._tracking[i,1] - self.particles[i,2]
            self._tracking[i,:] = 0

            if i in bleached:
                fully_bleached = True
            else:
                fully_bleached = False

            particle_summary = [time_tracked, bleach_steps, fully_bleached]
            self._tracking_times.append(particle_summary)
        
        return

class FluorescenceProfile(object):
    """ Compute and store a radially symmetric photobleaching profile. 

    Attributes:
        gate_intensities: Computed intensity values over a range of radii. 1st 
            column contains intensities, 2nd column is radii, and 3rd is radii 
            squared.
    
        **kwargs: 
            intensity_tirf (int): Number of photons emitted by the
                fluorophore per second due to the TIRF beam. Defaults to 5000.
            intensity_gate (int): Number of photons emitted per second due to 
                the gate beam. Defaults to 3e6.
            photon_budget (int): Mean number of photons emitted by fluorophore
                before it photobleaches. Defaults to 50000.
            r_ring (float): Radius, in um, of the ring swept out by the gate
                beam. Defaults to 7.5.
            r_tirf (float): Radius, in um, of TIRF beam. Defaults to 7.5.    
            gauss_width (float): Width, in um, of the Gaussian bleaching beam.
                Defaults to 0.4.
            num_steps (int): Number of computed discrete intensities from the 
                swept gate beam. Defaults to 500. 
            box_width (float): Width of the square box, in um, enclosing the
                region for which intensity is computed. Defaults to 40.
    """
    def __init__(self, **kwargs):
        defaults = {
            'intensity_tirf' : 5000,
            'intensity_gate' : 3000000,
            'photon_budget' : 50000,
            'r_ring' :  7.5,
            'r_tirf' :  7.5,
            'gauss_width' : 0.4,
            'num_steps' : 500,
            'box_width' : 40
        }

        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.update_gate_intensities() # populates gate_intensities
        
    def gauss_intensity(self, theta, R, d, c): 
        """Compute a Gaussian intensity from a single point's contribution."""    
        intensity = np.exp(-(R**2 + d**2 -2*R*d*np.cos(theta))/(2*c**2))   
        return intensity
    
    def update_gate_intensities(self):
        """Calculate a Gaussian ring as a function of distance from center."""
        
        max_radius = np.sqrt(2 * (self.box_width / 2 )**2)
        self.gate_intensities = np.zeros([self.num_steps, 3])
        self.gate_intensities[:,1] = np.linspace(0,max_radius, 
                                    self.num_steps, endpoint = True)
        self.gate_intensities[:,2] = self.gate_intensities[:,1]**2
        num_int = self.gate_intensities.shape[0]
    
        # Run numerical integration for every point along the intensity vector
        for r in range(0, num_int-1):
            int_result = integrate.quad(self.gauss_intensity, 0, np.pi*2, 
                                       args=(self.r_ring, 
                                             self.gate_intensities[r,1], 
                                             self.gauss_width))                                            
            self.gate_intensities[r,0] = int_result[0]
            
    def gate_bleaching_profile(self, time_step):
        """Calculate bleaching profile given current gate settings.
        
        Returns p_bleach_inv_gate, an array containing inverse probabilities of 
        bleaching per time_step as a function of distance from center.
        The 1st column contains inverse probabilities, 2nd column is radii, 
        and 3rd is radii squared. 
        """
     
        p_bleach_inv_gate = self.gate_intensities.copy()
        photons_per_tau_gate = self.intensity_gate * time_step
        
        # normalize and scale gate beam
        p_bleach_inv_gate[:,0] = self.gate_intensities[:,0] \
            * photons_per_tau_gate / np.amax(self.gate_intensities[:,0]) 
        p_bleach_inv_gate[:,0] = np.exp(math.log(0.5) * \
            p_bleach_inv_gate[:,0] / self.photon_budget)
        
        return p_bleach_inv_gate
    
    def tirf_bleaching_profile(self, time_step):
        """Calculate bleaching profile given current TIRF settings.
        
        Returns:
            p_bleach_inv_tirf [dict]: Uniform inverse bleaching probability 
            under TIRF illumination and radius of the TIRF spot.
        """
        photons_per_tau_tirf = self.intensity_tirf * time_step
        p_inv = math.exp(math.log(0.5) * \
            photons_per_tau_tirf / self.photon_budget)
        p_bleach_inv_tirf = {'r_tirf': self.r_tirf, 'p_inv' : p_inv}
        
        return p_bleach_inv_tirf
    
    def build_heat_map (self, num_pixels=400):
        """Return a square heatmatp of computed intensities.
        
        Args:
            num_pixels (int): resolution of the heatmap. Defaults to 400.
        """
        
        side_roi = self.box_width
        num_pixels = round(num_pixels / 2) * 2 # must be a multiple of 2
        side = np.linspace(-side_roi,side_roi,num_pixels)
        x,y = np.meshgrid(side,side)
        quad_side = round(len(x)/2)
    
        # we only need to compute one quadrant of the full ROI (technically
        # half a quadrant would be enough, but more convoluted to implement)
        quad_x, quad_y = x[quad_side:,quad_side:], y[quad_side:,quad_side:] 
    
        # calculate intensities in the top right quadrant
        r_z = np.sqrt(quad_x**2+quad_y**2)
        num_int = self.gate_intensities.shape[0]
        ind_z = np.searchsorted(
            self.gate_intensities[:num_int-1,1], r_z)
    
        quad_z = self.gate_intensities[ind_z, 0]
    
        # Populate final intensity array by reflecting the top right quadrant
        z = np.empty_like(x)
        z[quad_side:,quad_side:] = quad_z
        z[:quad_side,quad_side:] = quad_z[::-1,:]
        z[quad_side:,:quad_side] = quad_z[:,::-1]
        z[:quad_side,:quad_side] = quad_z[::-1,::-1]
        
        return x,y,z

### Simulation functions ######################################################

def init_gate_timers(tau, bleach_cycle_time=0.1, num_pre_bleach_cycles=50, 
                   gate_freq=0.5, fluo_params={}): 
    """Initialize timers used for the full-cycle photogate simulation.
    
    Returns the dictionary of parameters gate_timing that can then be used to
    keep track of the simulation's position within the bleaching cycle.
    
    Args:
        tau (float): time_step of the simulation, in seconds
        bleach_cycle_time (float): Time, in seconds, that the photogate remains 
            on in one cycle. Defaults to 0.1.
        num_pre_bleach_cycles (int): Number of initial photogate cycles used to 
            pre-bleach the ROI. Defaults to 50.
        gate_freq (float): Frequency, in Hz, of the gating cycle. 
            Defaults to 0.5.
        fluo_params (dict): Any additional fluorescence parameters (see allowed
            **kwargs for the FluorescenceProfile class). Defaults to empty.
    
    Returns:
        gate_timing (dict): Timer settings and counter variables to keep track 
            of the current state of the full PhotoGate experiment
        fluo_profiles (list): instances of FluorescenceProfile spanning the
            full range of requested ring radii.
        p_bleach_inv (numpy.array): gate bleaching profile from the first
            instance of FluorescenceProfile
    """
    
    gate_period = 1 / gate_freq # in seconds
    ctr_pre_bleach_cycle = 0
    ctr_cycle = 0
    steps_in_gate = int(bleach_cycle_time / tau)
    steps_in_tirf = int((gate_period - bleach_cycle_time) / tau)
    
    # determine required maximum radius:
    dummy_params = fluo_params.copy()
    dummy_params['num_steps'] = 1   
    dummy_profile = FluorescenceProfile(**dummy_params)
    r_ring_max = dummy_profile.r_ring
    
    # Create an array of fluorescence profiles for the entire required range
    scale = r_ring_max / num_pre_bleach_cycles
      
    fluo_params_tmp = fluo_params.copy()  
    del fluo_params_tmp['r_ring']
    fluo_profiles = [FluorescenceProfile(r_ring=r*scale, **fluo_params_tmp) \
                for r in range(num_pre_bleach_cycles+1)]    
    
    gate_timing = {'steps_in_gate': steps_in_gate, 
                  'steps_in_tirf': steps_in_tirf,
                  'num_pre_bleach_cycles': num_pre_bleach_cycles,
                  'ctr_pre_bleach_cycle': ctr_pre_bleach_cycle, 
                  'ctr_cycle': ctr_cycle}
    
    p_bleach_inv = fluo_profiles[0].gate_bleaching_profile(tau)
            
    return gate_timing, p_bleach_inv, fluo_profiles
    
def gate_update(gate_timing, tau, p_bleach_inv, fluo_profiles):
    """Determine which part of the bleach cycle the program is in.

    Updates counters in gate_timing and re-calculates the Gaussian profile of
    the pre-bleaching beam if needed.
    """
    
    num_cycles = gate_timing['num_pre_bleach_cycles']
    i, ii = gate_timing['ctr_pre_bleach_cycle'], gate_timing['ctr_cycle'] 
    
    # pre-bleaching phase
    if i <= num_cycles: 
        gate_or_tirf = 'gate'
        if ii == 0: # update photoGate radius
             # add later: intensities should depend on radius
            p_bleach_inv[:] = fluo_profiles[i].gate_bleaching_profile(tau)
            ii += 1
        elif ii < gate_timing['steps_in_gate']:
            ii += 1
        else:
            ii = 0
            i += 1

    # normal bleaching phase
    else: 
        if ii < gate_timing['steps_in_gate']: # gating
            gate_or_tirf = 'gate'
            ii += 1
        elif (ii < (gate_timing['steps_in_gate'] \
                    + gate_timing['steps_in_tirf'])): # TIRF
            gate_or_tirf = 'TIRF'
            ii += 1
        else: # start new bleaching cycle
            ii = 1
            gate_or_tirf = 'gate'
    
    gate_timing['num_pre_bleach_cycles'] = num_cycles
    gate_timing['ctr_pre_bleach_cycle'], gate_timing['ctr_cycle'] = i, ii
    
    return gate_or_tirf 

### Saving and loading functions ##############################################

def init_save_file(sim, fluo, gate_timing, base_file_name='data', 
                   result_dir='simResults'):
    """Write inf file for the simulation and  open csv file for writing
    
    Args:
        sim (class): Instance of the DiffusionSimulation class.
        fluo (class): Instance of the FluorescenceProfile class.
        gate_timing (dict): Timer settings controlling the full experiment.
        base_file_name (str): Base name for output files (without extension).
            Defaults to 'data'.
        result_dir (str): Directory for storing result files. 
            Defaults to 'simResults'
    
    Returns:
        file_writer (csv.writer object): writer object for new data file
        data_file (_io.TextIOWrapper object): data file being written
        dirs (dict): Directories used for saving data
    """
    
    def make(dir_name):
        " Make directory if it doesn't already exist"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    dirs = {}
    dirs['data'] = os.path.join(os.getcwd(), result_dir, 'data') 
    dirs['inf'] = os.path.join(dirs['data'], 'inf')
    dirs['raw'] = os.path.join(dirs['data'], 'raw')
    dirs['results'] = os.path.join(dirs['data'], 'results')
    dirs['figs'] = os.path.join(os.getcwd(), result_dir, 'figs')
    dirs['svg'] = os.path.join(dirs['figs'], 'svg')
    dirs['png'] = os.path.join(dirs['figs'], 'png')
     
    [make(d) for d in dirs.values()]
    
    data_file_name = os.path.join(dirs['raw'], base_file_name)
    inf_file_name = os.path.join(dirs['inf'], base_file_name)
    
    write_inf_file(inf_file_name, sim, fluo, gate_timing)
    
    data_file = open(data_file_name+'.csv', 'w', newline='')
    file_writer = csv.writer(data_file)
    
    return file_writer, data_file, dirs

def write_positions_to_file(sim, gate_or_tirf, file_writer):
    """Write current frame to the csv data file.
        
    All particles get flattened to a single line, each particle being
    represented by three numbers (x position, y position, and bleaching state).
    The final cell in the line contains the value of 'gate_or_tirf'.
    
    Args:
        sim (class): Instance of the DiffusionSimulation class.
        gate_or_tirf (str): Indicator of the type of illumination used in the
            current frame.
        file_writer (csv.writer object): writer object for new data file
    """    
    part_list = sim.particles.flatten().tolist()
    part_list.append(gate_or_tirf)    
    file_writer.writerow(part_list)    

def write_inf_file(write_file_name, sim, fluo, gate_timing):
    """Save all relevant simulation parameters into a text file.
    
    Args:
        write_file_name (str): Base output data file name (without extension).
        sim (class): Instance of the DiffusionSimulation class.
        fluo (class): Instance of the FluorescenceProfile class.
        gate_timing (dict): Timer settings controlling the full experiment.
    """
    def get_relevant_attributes(cls): 
        """ Return dictionary of needed class attributes """
        # exclude any private attributes starting with '_'
        atts = [i for i in cls.__dict__.items() if i[0][:1] != '_']    
        # exclude numpy arrays
        valid_atts = [i for i in atts if not isinstance(i[1], np.ndarray)]
        #valid_atts = dict(valid_atts)
        return valid_atts    
    
    sim_params = get_relevant_attributes(sim)
    fluo_params = get_relevant_attributes(fluo)   
    
    with open(write_file_name+'.inf', 'w') as inf_file:
        inf_file.write('Timestamp: {:%Y-%m-%d %H:%M:%S}\n\n'.\
            format(datetime.datetime.now()))
        for x in sim_params:
            inf_file.write(x[0] + ' : ' + str(x[1]) + '\n')
        for x in fluo_params:
            inf_file.write(x[0] + ' : ' + str(x[1]) + '\n')
        for x in gate_timing:
            # exclude counters
            if (x != 'ctr_cycle') and (x != 'ctr_pre_bleach_cycle'): 
                inf_file.write(x + ' : ' + str(gate_timing[x]) + '\n')

def write_tracking_times_csv(sim, file):
    " Get the _tracking_times list from simulation and save it in csv format."
    
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time tracked (s)', 'Bleaching steps', 
                         'Fully bleached'])        
        for item in sim._tracking_times:
            writer.writerow(item)

def read_tracking_times_csv(file):
   " Read _tracking_times from csv file and return it as a single dictionary."
   
   with open(file, 'r') as data_file:
        reader = csv.reader(data_file)
        data = [[],[],[]]
        next(reader, None) # skip header
        for row in reader:
            data[0].append(float(row[0]))
            data[1].append(int(row[1]))
            data[2].append(row[2].lower() in ('true', 't', '1'))
   return data

def read_params_from_inf_file(inf_file):
    """ Return parameter names and values from an .inf file as a dictionary."""
    
    def str_to_num(num):
        try:
            return int(num)
        except ValueError:
            return float(num)
    
    param_names, param_values = [], []
    
    for line in open(inf_file, 'r'):
        if " : " in line:
            curr_line= line.rstrip()
            param = curr_line.split(" : ", 1)
            param_names.append(param[0])
            param_values.append(str_to_num(param[1]))

    params = dict(zip(param_names, param_values))
    return params

def load_results(result_name, var_name, dirs):
    """Load given type of result across entire range of parameters.
    
    Args:
        result_name (str): Measurement of interest. Must match existing result
            type in dirs['results'].
        var_name (str): Variable that the data is iterating over. Must match
            an existing key in the .inf files, e.g. 'r_ring'.
        dirs (dict): Directories used for saving data
    
    Returns:
        var (numpy.array): values of the variable var_name for each data set.
        data (numpy.array): saved data for result_name for each value of var.
        time_ax (numpy.array): time axis that can be used to plot any row in
            data. t = 0 at the end of pre-bleaching cycle.
    """
    
    res_path = dirs['results']
    inf_path = dirs['inf']
    
    all_files = os.listdir(res_path)
    
    ext = '_'+result_name+'.txt'
    
    result_files = [x for x in all_files if ext in x]
    result_full = [os.path.join(res_path, x) for x in result_files]
    inf_files = [x[:-len(ext)]+'.inf' for x in result_files]
    inf_full = [os.path.join(inf_path, x) for x in inf_files]
    
    params_0 = read_params_from_inf_file(inf_full[0])
    
    var = np.zeros(len(result_files))
    data = np.zeros([len(result_files),params_0['num_time_steps']])
    
    # the +2 and +1 below are added because the program currently adds extra
    # steps during the pre-bleaching cycle because of how gate_timing counters
    # are handled. While minor, this should be fixed in the future.
    pre_end_ind = (params_0['num_pre_bleach_cycles'] + 2) \
        * (params_0['steps_in_gate'] + 1)
    pre_end_t = pre_end_ind * params_0['tau']    
    time_ax = np.linspace(0,params_0['sim_time'], 
                          num=params_0['num_time_steps'])
    time_ax = time_ax - pre_end_t                     
    for i in range(len(result_files)):
        params = read_params_from_inf_file(inf_full[i])
        var[i] = params[var_name]
        data[i,:] = np.loadtxt(result_full[i])
    
    # sort results by ascending var
    ind = np.argsort(var)
    var = var[ind]
    data = data[ind, :]
    
    return var, data, time_ax

def load_tracking_results(result_files):
    " Load results from multiple tracking files and return them as a list. "    
    
    bleached = []
    bleach_steps = []
    track_times = []
    data = []
    for file in result_files:
        data_tmp = read_tracking_times_csv(file)
        track_times = np.asarray(data_tmp[0])
        bleach_steps = np.asarray(data_tmp[1])
        bleached = np.asarray(data_tmp[2])
        data.append([track_times, bleach_steps, bleached])
    
    return data

def make_result_heatmap(target_dir, result_name, var_name, t_steps=40, 
                        save_result=True):
    """Prepare heatmap of a simulation as a time vs. parameter of choice.
    
    Args:
        target_dir (str): Base directory holding the results of a simulation.
        result_name (str): Result to be plotted on the heatmap. Must exist in
            target_dir/data/results.
        var_name (str): Name of the parameter over which the selected result is
            iterated. Must be a valid dictionary key in the .inf file.
        t_steps (int): Number of time steps (i.e. x-axis resolution) of the
            heatmap. Defaults to 40.
        save_result (bool): Should the data be saved to file? Defaults to True.
    
    Returns:
        data (list):
            info (dict): Metadata and annotations for the plot.
            x_plot, y_plot, z_plot (numpy.arrays): Data to be plotted.
    
    """
    
    log_offset = 0.00001 # small value to replace zeros in z for log plot    
    
    dirs = {'results' : os.path.join(target_dir, 'data', 'results'),
            'inf' : os.path.join(target_dir, 'data', 'inf'),
            'save' : os.path.join(target_dir, 'figs', 'analysis')}
    
    if not os.path.exists(dirs['save']):
                os.makedirs(dirs['save']) 
          
    inf_names = os.listdir(dirs['inf'])
    inf_files = [os.path.join(dirs['inf'], x) for x in inf_names]
    params = read_params_from_inf_file(inf_files[0])
    
    val, d, t = load_results(result_name, var_name, dirs)
    if var_name == 'steps_in_tirf':
        val = 1 / ((val + params['steps_in_gate']) * params['tau'])
    
    tmax = t[-1]
    t_vals = np.linspace(0, tmax, t_steps)
    inds = np.searchsorted(t, t_vals)
    
    x_plot = t[inds]
    y_plot = val
    z_plot = d[:,inds]
    
    # replace zeros in z_plot with small values to avoid undefined log values
    z_plot[z_plot == 0] = log_offset
    
    title = os.path.split(target_dir)[1]
    full_name = title + '_' + result_name + '_vs_' + var_name    
    
    info = {'title' : title,
              'x_axis' : 'Time after bleaching (s)',
              'y_axis' : var_name,
              'save_dir' : dirs['save'],
              'full_name' : full_name}
    
    data = [info, x_plot, y_plot, z_plot]
    
    # Save the x, y, and z components of the plot as a pickle file
    if save_result:
        file_name = full_name + '.pickle'
        file_path = os.path.join(dirs['save'], file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    return data


### Data plotting functions ###################################################

def init_anim_plot(source_dir='simResults', base_file_name = 'data'):
    """Initialize and prepare figure for plotting animation"""
    
    dir_data = os.path.join(os.getcwd(), source_dir ,'data')
    if not os.path.exists(dir_data):
        print("Source directory does not exist")
        
    dir_out = os.path.join(os.getcwd(), source_dir, 'video')
    if not os.path.exists(dir_out):
            os.makedirs(dir_out)
    
    data_file_name = os.path.join(dir_data, 'raw', base_file_name)
    data_file = open(os.path.join(os.getcwd(), data_file_name+'.csv'), 'r')
    file_reader = csv.reader(data_file)    
    
    inf_file_name = os.path.join(dir_data, 'inf', base_file_name)
    inf_file = os.path.join(os.getcwd(), inf_file_name+'.inf')
    
    params = read_params_from_inf_file(inf_file)
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure('animation')
    axes = fig.add_subplot(111, aspect='equal', autoscale_on=False)
    
    axes.set_xlabel(r'Distance ($\mu m$)')
    axes.set_ylabel(r'Distance ($\mu m$)')
    extra_space = params['grid_size'] * 0.1
    axes.set_xlim([-extra_space, params['grid_size'] + extra_space])
    axes.set_ylim([-extra_space, params['grid_size'] + extra_space])
    
    return file_reader, data_file, params, fig, axes

def plot_anim_frame(data_row, params, axes, ctr, marker_size=5,
                        plot_style='simple'):
    """ Plot a single name of animation and return the list of artists"""
    
    gate_or_tirf = data_row[-1]
    curr_row = np.asarray(data_row[:-1], dtype='float32')  
    data = curr_row.reshape(params['num_particles'], 3)
    data[:,0:2] = data[:,0:2] * params['grid_delta'] # scale x and y
    
    # Only keep unbleached particles for plotting:
    data = data[(data[:,2] > 0), :]
     
    curr_time = format(ctr * params['tau'] * 
                           params['steps_per_frame'], '7.3f') 
    
    text_time = axes.text(0.95, 0.01, 'Time: '+curr_time+' s',
            verticalalignment='bottom', horizontalalignment='right',
            transform=axes.transAxes,
            color='black', fontsize=15)
    
    text_gate = axes.text(0.95, 0.93, gate_or_tirf,
            verticalalignment='bottom', horizontalalignment='right',
            transform=axes.transAxes,
            color='black', fontsize=15)
    
    if plot_style == 'fancy':
        colors_bleach_tmp = data[:,2] * 0.55 / params['oligo_state'] + 0.1
        colors_bleach_tmp = 1 - colors_bleach_tmp
        colorsBleach = np.transpose([colors_bleach_tmp,
                                     colors_bleach_tmp, colors_bleach_tmp])
        frame = axes.scatter(data[:,0],data[:,1], 
                           color=colorsBleach, alpha=1)
    elif plot_style == 'simple':
        frame, = axes.plot(data[:,0],data[:,1], 'o', 
                         color='black', markersize=marker_size)
    artists = [frame, text_time, text_gate]
    
    return artists

def plot_current_state(sim, marker_size=6, plot_trackable=False,
                       figure_name='live_plot'):
    """Plot current positions of all particles in sim to figure.
    
    Args:
        sim (class): Instance of the DiffusionSimulation class.
        marker_size (int): Marker size for plotting particles. Defaults to 6.
        plot_trackable (bool): show particles that are far enough from each
            other to be tracked using a different color. Defaults to False.
        rROI (float): radius of the ROI, in um. Used only for plot_trackable.
            Defaults to 7.
        figure_name (str): Name of the figure which is used for live plotting.
            Defaults to 'live_plot'.
    """

    plot_type = 'regular'
    
    fig = plt.figure(figure_name)
    
    unbleached = sim.particles[(sim.particles[:,2] > 0), :] * sim.grid_delta
    
    fig.add_subplot(111, aspect='equal', autoscale_on=True)
    
    plt.plot(unbleached[:,0],unbleached[:,1], 'o', 
                     color='black', markersize=marker_size)
    
    if plot_trackable:
        indices = sim.get_trackable()
        if indices.size > 0:
            plt.plot(sim.particles[indices,0] * sim.grid_delta, 
                     sim.particles[indices,1] * sim.grid_delta, 'o', 
                     color='red', markersize=(marker_size+2))
                         
    if plot_type == 'fancy':
        colors_bleach_tmp = unbleached[:,2] * 0.55 / sim.oligo_state + 0.1
        colors_bleach_tmp = 1 - colors_bleach_tmp
        colors_bleach = np.transpose([colors_bleach_tmp,
                                 colors_bleach_tmp, colors_bleach_tmp])
        plt.scatter(unbleached[:,0] * sim.grid_delta, 
                    unbleached[:,1] * sim.grid_delta,
                    color=colors_bleach, alpha=0.5)  

def plot_result_heatmap(data, save_result=True, disp_result=True, 
                        file_ext='svg', zlim = (), color_map='gist_ncar'):
    """Plot result heatmap.
    
    Args:
        data (list): labels and x,y, and z data components of the heatmap plot
        file_name (str): Destination file for saving the plot. Defaults to 
            'spam'.
        save_result (bool): Should the plot be saved to file? Defaults to True.
        disp_result (bool): Should the plot be displayed immediately? Defaults
            to True.
        file_ext (str): extension of the saved plot. Defaults to 'svg'.
        zlim (tuple): lower and upper limits of the logarithmic z-axis 
            normalization. Defaults to (), which means the min and max of data
            will be used.
        color_map (str): Matplotlib colormap to use for plotting. Defaults to
            'gist_ncar'.
    """
    
    # unpack data
    info, x_plot, y_plot, z_plot = [element for element in data]   
    
    #initialize plot
    fig, ax = plt.subplots()
    ax.set_title(info['title'])
    ax.set_xlabel(info['x_axis'])
    ax.set_ylabel(info['y_axis'])

    # Determine z limits in the heatmap plot
    if not zlim:    
        zlim = (z_plot.min(), z_plot.max())
        
    plt.pcolormesh(x_plot,y_plot,z_plot, norm=LogNorm(*zlim), cmap=color_map)
    plt.colorbar()
    
    
    if save_result:
        file_name = info['full_name'] + '.' + file_ext
        file_path = os.path.join(info['save_dir'], file_name)
        fig.savefig(file_path)       
        
    if disp_result:
        plt.show()

def plot_fluo_density_results(sim, fluo_density_in_roi, percent_oligomers):
    "Plot fluorophore density data and return the figure."
    time_vect = np.linspace(0,sim.sim_time, num=sim.num_time_steps)
    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(time_vect, fluo_density_in_roi)
    axarr[0].set_title('Density of fluorophores and % oligomers')
    axarr[0].set_ylabel(r'Fluorophores per $\mu m^2$')
    axarr[1].plot(time_vect, percent_oligomers)
    axarr[1].set_xlabel('time(s)')
    axarr[1].set_ylabel('% unbleached in ROI')
    fig.set_size_inches(15,10, forward=True)
    
    return fig

def plot_tracking_results(sim):
    "Plot data from _tracking_times as three histograms and return the figure."
    
    bleached, unbleached, steps = [], [], []
    for time in sim._tracking_times:
            
        if time[2]:
            bleached.append(time[0])
            steps.append(time[1]) 
        else:
            unbleached.append(time[0])
    
    max_steps = max(steps)
    step_bins = np.arange(max_steps+1) + 0.5
    
    fig, axarr = plt.subplots(1,3)
    axarr[0].hist(unbleached, normed=0, facecolor='green', alpha=0.75)
    axarr[1].hist(bleached, normed=0, facecolor='black', alpha=0.75)
    axarr[2].hist(steps, bins=step_bins, range=(0.5,max_steps+0.5), 
                    align='mid', facecolor='blue', rwidth=0.5, alpha=0.75)
    axarr[0].set_xlabel('Unbleached (s)')
    axarr[0].set_ylabel('Counts')
    axarr[1].set_xlabel('Bleached (s)')
    axarr[2].set_xlabel('Bleach steps') 
    fig.set_size_inches(15,5, forward=True)
    
    return fig

def plot_tracking_statistics(track_data, print_results=False):
    """Plot statistics from multiple iterations of tracking simulations.
    
    Returns:
        fig (matplotlib figure): Figure plotting the results
        bleach_track_times (numpy.array): tracking times of fluorophores whose
            tracking trajectories end in bleaching.
        all_track_times (numpy.array): tracking times of all fluorophores
    """
    
    bleached, unbleached = [], []
    all_steps = []
    max_steps = 0
    
    for data in track_data:
        steps = []
        
        for i in range(len(data[0])):
            if data[2][i]:          
                bleached.append(data[0][i])
                steps.append(data[1][i]) 
            else:
                unbleached.append(data[0][i])
        max_steps = max(max_steps, max(steps))
        all_steps.append(np.asarray(steps))
    
    steps_by_trial = np.zeros([len(all_steps), max_steps+1], dtype='int')
   
    for i, trial in enumerate(all_steps):
        for n in range(max_steps+1):
            steps_by_trial[i,n] = sum(all_steps[i] == n)
    
    bincenters = np.arange(1,max_steps+1)
    binvalues = np.mean(steps_by_trial, axis=0)
    err_std = np.std(steps_by_trial, axis=0)
    width = 0.5
    
    fig, axarr = plt.subplots(1,3)
    axarr[0].hist(unbleached, normed=0, facecolor='green', alpha=0.75)
    axarr[1].hist(bleached, normed=0, facecolor='black', alpha=0.75)
    axarr[2].bar(bincenters, binvalues[1:], width=width, yerr=err_std[1:],
                align='center', ecolor='black')
    axarr[0].set_xlabel('Unbleached (s)')
    axarr[0].set_ylabel('Counts')
    axarr[1].set_xlabel('Bleached (s)')
    axarr[2].set_xlabel('Bleach steps') 
    fig.set_size_inches(15,5, forward=True)
    
    bleach_track_times = bleached
    all_track_times = np.concatenate((bleached, unbleached), axis=0)
    
    if print_results:
        print('Bin values:', binvalues)
        print('std errors:', err_std)
    
    return fig, all_track_times, bleach_track_times

if __name__ == "__main__":
    #import sys
    print("Running as a script - currently not configured to do anything")