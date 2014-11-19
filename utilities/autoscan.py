import os, subprocess, sys

sys.path.append('../../ion flying/')

from concurrent.futures import ProcessPoolExecutor
from fly import ion_flyer
import numpy as np

#natoms = range(50, 501, 50)
phase = np.linspace(0, 1, 21)
# cool = np.linspace(0, 700, 8)
dirname = '../mixedscan/Xe_decay_050/' # remember to change master.info along with folder!
binfolder = os.getcwd()

def runner(n):
	outfolder = dirname + str(n) + '/'
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)
	else:
		print("folder exists, skipping!")
		return
	
	with open('master.info', 'r') as source , open(outfolder + '/trap.info', 'w+') as dest:
		changed = False
		for line in source:
#			if line.startswith('    Ca') and not changed:
#				dest.write('    Ca ' + str(n) + '\n')
#			if line.startswith('    coolperiods') and not changed:
#				dest.write('    coolperiods ' + str(700+n) + '\n')
#			if line.startswith('    coolperiods') and not changed:
#				dest.write('    coolperiods ' + str(n) + '\n')
				changed = True
			else:
				dest.write(line)
				
	# simulate the actual ion trapping
	print("running for ", outfolder)
	with open(os.devnull, 'w') as devnull:
		subprocess.call([binfolder + '/ccmd', '.'], cwd=outfolder, stdout=devnull)
	
	# and fly ions to the detector
	if not os.path.exists(outfolder):
		print('folder does not exist, skipping flight')
		return
	crystdat = outfolder
	wavefile = '/home/atreju/Documents/Uni/projects/sin eject/sim/ion flying/Waveforms/WAVE/WASC0628.CSV'
	H = 1.e-8
	runs = 1
	flyer = ion_flyer()
	flyer.import_crystal(crystdat)
	flyer.load_waveform(wavefile)
	flyer.fly_ions(H)
	
	print('saving data')
	np.savetxt(outfolder + 'ejected.csv', flyer.pos)
	np.savetxt(outfolder + 'arrivaltimes.csv', flyer.totalTime)
	np.savetxt(outfolder + 'types.csv', flyer.types, fmt="%s")

with ProcessPoolExecutor(8) as executor:
	executor.map(runner, natoms)


