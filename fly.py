from __future__ import print_function # for python3-compatibility

import numpy as np
from scipy.interpolate import UnivariateSpline

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '\\..\\')


from simioniser.EField2D import EField2D
from simioniser.EField3D import EField3D

import ctypes
from ctypes import c_double, c_ulong, c_uint
c_double_p = ctypes.POINTER(c_double)




class ion_flyer(object):
	def __init__(self, verbose=False):
		target = 'coulomb'
		self.verbose = verbose
		
		self.localdir = os.path.dirname(os.path.realpath(__file__)) + '/'
		localdir = self.localdir

		if sys.platform.startswith('linux'):
			compiler = 'gcc'
			commonopts = ['-c', '-fPIC', '-Ofast', '-march=native', '-std=c99', '-fno-exceptions', '-fomit-frame-pointer']
			extension = '.so'
		elif sys.platform == 'win32':
			commonopts = ['-c', '-Ofast', '-march=native', '-std=c99', '-fno-exceptions', '-fomit-frame-pointer']
			compiler = 'C:\\MinGW\\bin\\gcc'
			extension = '.dll'
		else:
			raise RuntimeError('Platform not supported!')


		libpath = localdir + target + extension

		if not os.path.exists(libpath) or os.stat(localdir + target + '.c').st_mtime > os.stat(libpath).st_mtime: # we need to recompile
			from subprocess import call
			# include branch prediction generation. compile final version with only -fprofile-use
			profcommand = [compiler, target + '.c']
			profcommand[1:1] = commonopts
	
			print()
			print()
			print('===================================')
			print('compilation target: ', target)
			call(profcommand, cwd=localdir)
			call([compiler, '-shared', target + '.o', '-o', target + extension], cwd=localdir)
			print('COMPILATION: PROFILING RUN')
			print('===================================')
			print()
			print()
		elif self.verbose:
			print('library up to date, not recompiling field accelerator')
		
		self.coulomb = ctypes.cdll.LoadLibrary(libpath)
		self.coulomb.get_coulomb_force.argtypes = [c_uint, c_double_p, c_double_p]
		self.coulomb.get_coulomb_force.restype = None

		
	def import_crystal(self, foldername):
		masses = {'Calcium': 40.,'Xenon': 130.0,'Pump1': 28.3,'Pump2': 30.2,'Xenon128': 127.9,'Xenon129': 128.9,'Xenon130': 129.9,'Xenon131': 130.9,'Xenon132': 131.9,'Xenon134': 133.9,'Xenon136': 135.9,'Calcium2': 42., 'Calcium3': 44.,'NH3': 17., 'CalciumFluoride': 59., 'Ammonia-d3' : 20, 'Ytterbium171': 171, 'Ytterbium172': 172, 'Ytterbium173': 173, 'Ytterbium174': 174, 'CalciumOH': 57.,'CalciumOD': 58.}
		files = [x for x in os.listdir(foldername) if x.endswith('_pos.csv')]
		self.pos = []
		self.vel = []
		self.mass = []
		self.nIons = 0
		self.types = []
		for f in files:
			data = np.genfromtxt(foldername + f, skip_header=1, delimiter=',')
			itype = f[:-8]
			mass = masses[itype]
			if len(data.shape) == 1:
				data.resize((1, 7))
			self.pos.extend(data[:, :3] + np.array([-202, -0.02689, 0])*0.2e-3) # [-32.05e-3, 2.5e-4, 0]) # shift center coordinates

			self.vel.extend(data[:, 3:6])
			self.mass.extend([mass*1.6605389e-27]*data.shape[0])
			self.nIons += data.shape[0]
			#self.nIonsMy.append(data.shape[0])
			self.types.extend([itype]*data.shape[0])
			
			if self.verbose: print('loading', data.shape[0], 'ions from', f)
			
		self.mass = np.array(self.mass).reshape((self.nIons, 1))
		self.pos = np.array(self.pos)
		self.vel = np.array(self.vel)
		self.totalTime = np.zeros(self.nIons)
		self.types = np.array(self.types)

		if self.verbose: print('loaded a total of ', self.nIons, 'ions')
	
	def calc_acceleration(self, pos, ef, mass):
		q = 1.60217657e-19
		de = np.zeros_like(pos)
		
		for f in ef:
			imove = f.inArray3(pos)
			
			if imove.any():
				de[imove, :] = -f.getField3(pos[imove, :])

		# this is rather dodgy, as we have to match the right mass for multi-component crystals!
		# fm = self.coulomb_force(pos)
		fm = np.zeros_like(pos)
		self.coulomb.get_coulomb_force(pos.shape[0], pos.ctypes.data_as(c_double_p), fm.ctypes.data_as(c_double_p))
		
		return q*de/mass + fm/mass
	
	def coulomb_force(self, pos):
		k = 2.30707735e-28
		force = np.zeros_like(pos)
		for i, p1 in enumerate(pos):
			for p2 in pos:
				dist = p1 - p2
				if dist.any():
					force[i, :] += k*dist/np.sqrt(dist[0]**2+dist[1]**2+dist[2]**2)**3
		return force
	
	def collision(self, pos, ef):
		collision_index = np.ones((self.nIons, 1), dtype=np.bool)
		aindex = np.zeros((self.nIons, ), dtype=np.bool)
		for f in ef:
			inArray = f.inArray3(pos)
			
			inIndex = np.where(inArray)[0]
			aindex |= inArray
			eindex = f.isElectrode3(pos)
			collision_index[inArray & eindex, 0] = 0
		collision_index[~aindex, 0] = 0
		return collision_index
		
	def load_waveform(self, wavefile):
		#self.wave = np.genfromtxt(wavefile, delimiter=',', skip_header=30)
		#x = np.arange(self.wave.shape[0])*2e-9
		#self.B = UnivariateSpline(x, self.wave[:, 2], s=0) # repeller
		#self.C = UnivariateSpline(x, self.wave[:, 1], s=0) # extractor
		                
		wave = np.genfromtxt('./Waveforms/r-el4.txt', skiprows=1, delimiter='\t') #repeller negative phase, electrode 4
		self.B1 = UnivariateSpline(wave[:, 0], wave[:, 1], s=0)
		wave = np.genfromtxt('./Waveforms/r+el2.txt', skiprows=1, delimiter='\t') #repeller positive phase, electrode 2
		self.B2 = UnivariateSpline(wave[:, 0], wave[:, 1], s=0)
		wave = np.genfromtxt('./Waveforms/e+el1.txt', skiprows=10, delimiter='\t') #extractor positive phase, electrode 1
		self.C1 = UnivariateSpline(wave[:, 0], wave[:, 1], s=0)
		wave = np.genfromtxt('./Waveforms/e-el3.txt', skiprows=10, delimiter='\t') #extractor negative phase, electrode 3
		self.C2 = UnivariateSpline(wave[:, 0], wave[:, 1], s=0)

		#import pdb; pdb.set_trace()
		#reading in 2 waveforms
		#wave = np.genfromtxt('./Waveforms/C22014-12-17-r100000.txt', skiprows=5, delimiter=',')
		#self.C = UnivariateSpline(wave[:, 0], wave[:, 1], s=0)
		#wave = np.genfromtxt('./Waveforms/C22014-12-17-e100000.txt', skiprows=5, delimiter=',')
		#self.B = UnivariateSpline(wave[:, 0], wave[:, 1], s=0)
	
	
	def fly_ions(self, H):
		localdir = self.localdir
                #new fields with different waveforms for all 4 electrodes
		ef1 = EField3D(localdir + 'Simion Fields/StarkTrap7/StarkTrap7', [0, 0, 0, 0], 1./2.e-4, np.array([-402.5, -68, -125.5])*2e-4, use_accelerator = True)
		ef2 = EField2D(localdir + 'Simion Fields/StarkTof/starkTof', [-1900, 0], 1./1.e-4, use_accelerator = True, prune_electrodes=True)
		ef = [ef1, ef2]
		self.ef = ef
		#scaling factor = 2, grid unit = e-4 m, spatial origin offset of the Trap = -402.5, -68, -125.5

		Td = 0 # some weird time delay, true for WASC0628
		
		acc = self.calc_acceleration(self.pos, ef, self.mass)
		
		# the first loop in Alex' code does nothing except adjust the time:
		# t1 += T_eject - Td
		# T = 5.48999999999999999e-6
		
		T = 2.251e-6 #1.258e-6 # offset between scope trigger and extraction pulse (for sinusoidal ejection with damping)
		self.totalTime = 0
				
		# the second loop does nothing for Td=0, so we ignore it for now.
		# we'll just implement any hold times in ccmd, and then shouldn't need this
		# loop, ever.
		
		# finally the third loop is where things are happening!
		
		self.plotpos_x = []
		self.plotpos_y = []
		self.plotpos_z = []
		
		step = 0
		while True:
			step += 1
		# the programm can't read out the amplitude out of the waveform files properly, one has to set the values manually 
			V1 = 100*self.B1(T)
			V2 = 100*self.C2(T)
			V3 = 100*self.B2(T)
			V4 = 100*self.C1(T)
			
		#	print (ef1)
		#fast adjustment for all 4 electrodes
			ef1.fastAdjust(0, V1) 
			ef1.fastAdjust(1, V2)
			ef1.fastAdjust(2, V3)
			ef1.fastAdjust(3, V4) 
			ef1.fastAdjust(4, 0) 
			
			#for only 2 waveforms
			#V1 = 100*self.B(T) # repeller
			#V2 = 100*self.C(T) # extractor
            
            # for only 2 waveforms
			#ef1.fastAdjust(0, V1)
			#ef1.fastAdjust(1, V2)
			#ef1.fastAdjust(2, 0) #check if this is for the grounded electrodes
			
			
			pos_f = self.pos + self.vel*H + 0.5*acc*H**2
			acc_f = self.calc_acceleration(pos_f, ef, self.mass)
			vel_f = self.vel + 0.5*(acc+acc_f)*H
			

			for i in np.arange(self.nIons):		# Loop to ckeck if any particle has moved out of an array, and adjust any back to edge of array
				if pos_f[i, 0] > ef1.xmax and self.pos[i, 0] < ef1.xmax and self.vel[i, :].any() and acc[i, :].any():
					h = ((ef1.xmax)-self.pos[i, 0])/self.vel[i, 0]
					pos_f[i, :] = self.pos[i, :] + self.vel[i, :]*h + 0.5*acc[i, :]*h**2
					acc_f[i, :] = self.calc_acceleration(pos_f[i:i+1, :], ef, self.mass[i:i+1])
					vel_f[i, :] = self.vel[i, :] + 0.5*(acc[i, :] + acc_f[i, :])*h
					pos_f[i, :] += np.array([1, 0, 0])*1e-7
				elif pos_f[i, 0] > ef2.xmax and self.pos[i, 0] < ef2.xmax and self.vel[i, :].any() and acc[i , :].any():
					print(i)
					print (self.pos[i, :])
					print (pos_f[i, :])
					print (ef2.xmax)
					raise RuntimeError
					
			# Update Positions
			self.pos = pos_f
			self.vel = vel_f
			acc = acc_f
			
			collision_map = self.collision(self.pos, ef) # Ionmove is a function that calculates if an ion has hit an electrode
			self.vel *= collision_map										 # If ion can move, the vel_a and acc_a gets a corresponding row of 1's, and 0's if it can't.
			acc *= collision_map												 # This step sets vel, acc and h to 0 for each ion that has hit an electrode, halting their movement
			
			T += H
			self.totalTime += H*collision_map
			
			
			
			if step % 20 == 0:
				self.plotpos_x.append(self.pos[:, 0])
				self.plotpos_y.append(self.pos[:, 1])
				self.plotpos_z.append(self.pos[:, 2])
				if self.verbose: print('current time: ', T)
			if not self.vel.any() or T > 5e-5:
				break
		
	
if __name__ == '__main__':
	__file__ = 'ion flying'
	from matplotlib import pyplot as plt
	from time import time
	
	__file__ = 'ion flying'

	crystdat = 'sample crystal/'
	wavefile = 'Waveforms/WAVE/WASC0628.CSV'
	
	# Time step size. This must be small enough that an ion only moves by 1 Simion grid space in a single step.
	H = 2.e-9 
	#H = 1.e-10 
	
	runs = 1
	
	flyer = ion_flyer(verbose=True)
	flyer.import_crystal(crystdat)
	flyer.load_waveform(wavefile)
	t0 = time()
	flyer.fly_ions(H)
	print("Execution took %5.2f seconds" %(time() - t0))
	px = np.array(flyer.plotpos_x)
	py = np.array(flyer.plotpos_y)
	pz = np.array(flyer.plotpos_z)
	
	pr = np.sqrt(py**2 + pz**2)
	
	#these are just for plotting the trajectories
	ef2 = EField2D('Simion Fields/StarkTof/starkTof', [-1900, 0], 1./1.e-4, use_accelerator = False)
	ef1 = EField3D('Simion Fields/StarkTrap7/StarkTrap7', [220, 97, 220, 97], 1./2.e-4, np.array([-402.5, -68., -125.5])*2e-4, use_accelerator = False)

#	ef2.plotPotential()
#	plt.plot(px, pr)
#	ef1.plotPotential()
#	plt.plot(px, py, 'b')
#	plt.plot(flyer.pos[:, 0], flyer.pos[:, 1], 'x')
#	plt.plot(px, pz, 'r')
	
	plt.figure()
	xs = np.linspace(0, 0.35, 400)
	ys = np.linspace(0, 0.025, 100)
	pos = np.zeros((40000, 3))
	
	for i, x in enumerate(xs):
		for j, y in enumerate(ys):
			pos[100*i+j, :] = [x, y, 0]
	
	fig0 = plt.figure(0)		
	ax0 = fig0.add_subplot(111, aspect='equal')
	ef1.plotPotential(plane=1)
	ef2.plotPotential()
	fig1 = plt.figure(1)
	ax1 = fig1.add_subplot(111, aspect='equal')
	ef1.plotPotential()
	ef2.plotPotential()

	types = set(flyer.types)
	styles = ['r', 'g', 'b', 'k']
	for i, t in enumerate(types):
		ind = np.where(flyer.types == t)[0]
		r = np.sqrt(flyer.pos[ind, 1]**2 + flyer.pos[ind, 2]**2)
		count = len(np.where((flyer.pos[ind, 0] > 0.12) & (r < 0.013))[0]) #0.124 is the length of the TOF tube and 0.013 is the radius of the MCPs
		print(count, 'ions of type', t, 'detected on MCP')
		#print(count, 'ions of type', t, 'detected on MCP')
		ax0.plot(px[:, ind], pz[:, ind], styles[i])
		ax1.plot(px[:, ind], py[:, ind], styles[i])
	
	ax0.set_aspect('equal', 'datalim')
	ax1.set_aspect('equal', 'datalim')


	hist_nbins = 100
	[hist_n, hist_bins] = np.histogram (flyer.totalTime * 1e6, hist_nbins)
	hist_bins = hist_bins[:-1] # The array of bins includes the right-most edge, so is hist_nbins+1 in length.
	plt.figure()
	plt.plot(hist_bins, hist_n)

	hist_data = np.transpose(np.vstack((hist_bins, hist_n)))
	np.savetxt(crystdat + "tof.csv", hist_data, delimiter=',')
	np.savetxt(crystdat + "flightTimes.csv", flyer.totalTime * 1e6, delimiter=',')
	np.savetxt(crystdat + "MCPposition.csv", flyer.pos[:, :], delimiter=',')
	
	
	plt.show()

	
