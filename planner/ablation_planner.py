"""
This script is used to plan a series of cuts for a given object.
"""
import numpy as np
from cut_simulator import Simulator
import pyvista as pv
import matplotlib.pyplot as plt

#%% Grid Generation (Obj, Constraint, Initial)
np.random.seed(1)
x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
x = x.reshape((-1, 1))
y = y.reshape((-1, 1))
z = np.zeros_like(x)
state = np.hstack([x, y, z])
objFxn = np.hstack([x, y, z])
objFxn[np.where(np.logical_and(np.abs(objFxn[:,0]) < 3, np.abs(objFxn[:,1]) < 3))[0], 2] -= 2
constraintFxn = objFxn.copy()
constraintFxn[:,2] = -5

# Initial Plotting
pl = pv.Plotter()
pl.add_mesh(state, color = "green")
pl.add_mesh(objFxn, color = "blue")
pl.add_mesh(constraintFxn, color = "red")
pl.show()

#%% Run Simulator
#sim = Simulator(1, 0, 13.5613, 2.25, 0.40707, 7.09, objFxn, constraintFxn)
sim = Simulator(1, 1.939, 1/0.333885349, 0.483187452, 11.071, 12.72661, objFxn, constraintFxn)
volume = np.sum(np.maximum(0, state[:,2] - objFxn[:,2]))
inputCode = ""
inputSeq = []
inputSeqLength = [0]
iteration = 1
volList = [volume]
overcutList = [0]
cutFrac = 1
#while inputCode.lower() != "stop":
while cutFrac > 0.15:
    test = sim.graphPlan(state, 0, maxiter = 10000, method = 'supergauss') # Change maxiter to search more or less per cut sequence
    # pl = pv.Plotter()
    # pl.add_mesh(test[1], color = "green")
    # pl.add_mesh(objFxn, color = "blue")
    # pl.add_mesh(constraintFxn, color = "red")
    # pl.show()
    inputSeq += test[0]
    state = test[1].copy()
    curVol = np.sum(np.maximum(0, -sim.objFxn(test[1][:,0], test[1][:,1]) + test[1][:,2]))
    overcutVol = np.sum(np.minimum(0, -sim.objFxn(test[1][:,0], test[1][:,1]) + test[1][:,2]))
    volList.append(curVol)
    overcutList.append(overcutVol)
    inputSeqLength.append(len(inputSeq))
    cutFrac = curVol/ volume
    print("Iteration {}: --Volume Remaining: {:.4f}%".format(iteration, cutFrac * 100))
    print("Iteration {}: --Overcut Percentage: {:.4f}%".format(iteration, overcutVol * 100 / volume))
    print("Iteration {}: --Input Length: {}".format(iteration, len(inputSeq)))
    #inputCode = input("Continue searching? Enter 'stop' to stop: ")
    iteration += 1
    
#%% Plotting and Data Analysis
inputSeq = np.array(inputSeq)
inputSeq = inputSeq[~np.all(inputSeq == 0, axis = 1)]
np.save("../data/laser_inputs/inputSeq.npy", inputSeq)

plt.figure(1)
plt.bar(["B4 Cut {}".format(i) for i in np.arange(1, len(volList) + 1, 1)], volList)
plt.bar(["B4 Cut {}".format(i) for i in np.arange(1, len(volList) + 1, 1)], overcutList, color = 'g')
plt.ylabel("Remaining Volume")
plt.grid(axis='y')
plt.show()

plt.figure(2)
plt.bar(["B4 Cut {}".format(i) for i in np.arange(1, len(volList) + 1, 1)], inputSeqLength)

plt.ylabel("Input Sequence Length")
plt.grid(axis='y')
plt.show()

pl = pv.Plotter()
pl.add_mesh(state, color = "green")
pl.add_mesh(objFxn, color = "blue")
pl.add_mesh(constraintFxn, color = "red")
pl.show()

