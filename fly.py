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
		masses = {'Calcium': 40., 'Xenon': 131.293, 'NH3': 17., 'CalciumFluoride': 59., 'Ammonia-d3' : 20, 'Ytterbium171': 171, 'Ytterbium172': 172, 'Ytterbium173': 173, 'Ytterbium174': 174, 'CalciumOH': 57.,'CalciumOD': 58.}
		files = [x for x in os.listdir(foldername) if x.endswith('_pos.csv')]
		self.pos = []
		self.vel = []
		self.mass = []
		self.nIons = 0
		self.types = []
		for f in files:
			data = np.genfromtxt(foldername + f, skip_header=0, delimiter=',')
			itype = f[:-8]
			mass = masses[itype]
			if len(data.shape) == 1:
				data.resize((1, 7))
			self.pos.extend(data[:, :3] + np.array([-85, 0, 0])*15.82e-3) # [-32.05e-3, 2.5e-4, 0]) # shift center coordinates
			self.vel.extend(data[:, 3:6])
			self.mass.extend([mass*1.6605389e-27]*data.shape[0])
			self.nIons += data.shape[0]
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
		self.wave = np.genfromtxt(wavefile, delimiter=',', skip_header=30)
		x = np.arange(self.wave.shape[0])*2e-9
		self.B = UnivariateSpline(x, self.wave[:, 2], s=0) # repeller
		self.C = UnivariateSpline(x, self.wave[:, 1], s=0) # extractor
	
	def fly_ions(self, H):
		localdir = self.localdir
		ef1 = EField3D(localdir + 'Simion Fields/StarkTrap/starkTrap', [0, 0, 0], 1./15.82e-3, np.array([-2123., -428., -950.])*1e-3, use_accelerator = True)
		ef2 = EField2D(localdir + 'Simion Fields/StarkTof/starkTof', [-1900, 0], 1./3.95e-3, use_accelerator = True, prune_electrodes=True)
		ef = [ef1, ef2]
		self.ef = ef
		
		Td = 0 # some weird time delay, true for WASC0628
		T_eject = (self.wave[:, 2].argmax() + 1)*2e-9-2*H # position of ejection voltage maximum, coincides with the flank of ejection pulse
		
		acc = self.calc_acceleration(self.pos, ef, self.mass)
		
		# the first loop in Alex' code does nothing except adjust the time:
		# t1 += T_eject - Td
		T = 5.48999999999999999e-6
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
			
			V1 = self.B(T)*305 # repeller
			V2 = self.C(T)*176 # extractor		
			
			ef1.fastAdjust(0, V1)
			ef1.fastAdjust(1, V2)
			ef1.fastAdjust(2, 0)
			
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
	H = 1.e-8
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
	
	
	ef2 = EField2D('Simion Fields/StarkTof/starkTof', [-1900, 500], 1./3.95e-3, use_accelerator = False)
#	ef2.plotPotential()
#	plt.plot(px, pr)
	ef1 = EField3D('Simion Fields/StarkTrap/starkTrap', [200, 100, 0], 1./15.82e-3, np.array([-2123., -428., -950.])*1e-3, use_accelerator = False)
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
		count = len(np.where((flyer.pos[ind, 0] > 0.23) & (r < 0.05))[0])
		print(count, 'ions of type', t, 'detected on MCP')
		ax0.plot(px[:, ind], pz[:, ind], styles[i])
		ax1.plot(px[:, ind], py[:, ind], styles[i])
	
	ax0.set_aspect('equal', 'datalim')
	ax1.set_aspect('equal', 'datalim')

	plt.show()


	


	
