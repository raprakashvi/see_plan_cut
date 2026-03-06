"""
This script is used to simulate a laser cut on a given object.
"""
import numpy as np
import pyvista as pv
import time
from scipy.interpolate import CloughTocher2DInterpolator
from line_profiler import profile

# Add parallel threads implementation

class Simulator():
    
    def __init__(self, density, threshold, enthalpy, spotSize, maxEnergy, P = 1, objPCD=None, constraintPCD=None):
        self.p = density
        self.phi = threshold
        self.h = enthalpy
        self.w = spotSize
        self.maxEnergy = maxEnergy
        self.P = P
        self.objPCD = objPCD
        self.constraintPCD = constraintPCD
        if self.objPCD is not None:
            self.objInterp = CloughTocher2DInterpolator(objPCD[:,0:2], objPCD[:,2],
                                 fill_value=0)
        if self.constraintPCD is not None:
            self.constraintInterp = CloughTocher2DInterpolator(constraintPCD[:,0:2], constraintPCD[:,2],
                                                          fill_value=0)
        
    def graphPlan(self, state, targetCost, maxiter = 10000, maxtime = 300, method = 'gauss'):
        inputSeq = []
        
        minEnergy = self.phi
        ### TO DO -------------------------
        success, u = self._motionPlan(state, targetCost, np.linspace(minEnergy, self.maxEnergy, 100), maxiter, maxtime, method) 
        for i in range(len(u)):
            if method == 'gauss':
                state = self.simulate(state, u[i])
            elif method == 'supergauss':
                state = self.simulate_SG(state, u[i])
            else:
                print("Method not known!")
                return
        inputSeq += u
        return inputSeq, state, success
     
    @profile
    def _motionPlan(self, state, targetCost, energyRange, maxiter = 10000, maxtime = 300, method = 'gauss'):
        rng = np.random.default_rng()
        tx_Range = np.linspace(-0.1754, 0.1754, 20)
        ty_Range = np.linspace(-0.1754, 0.1754, 20)
        
        G = {"id":[0], "state":[state], "prevState":[-1], "objZ":[self.objFxn(state[:,0], state[:,1])], "cost":[self.cost(state[:,2], self.objFxn(state[:,0], state[:,1]))], "input":[[0, 0, 0, 0, 0]]}

        lowestCost = [0, G["cost"][0]]
        highestCost = [0, G["cost"][0]]
        iteration = 1
        counter = 1
        exitcode = 1
        start = time.time()
        while lowestCost[1] > targetCost:
            if iteration > maxiter:
                exitcode = 0
                print("Exiting after max retries...")
                break
            if time.time() - start > maxtime:
                exitcode = 0
                print("Exiting after max alloted time...")
                break
                
            nodeSelectWeights = np.maximum((highestCost[1] - np.array(G["cost"])) ** 6, 0.000001)
            nodeID = rng.choice(G["id"], p=nodeSelectWeights / np.sum(nodeSelectWeights, dtype=float))
            
            tempState = G["state"][nodeID]
            tempObjZ = G["objZ"][nodeID]
            
            if np.all(tempState[:,2] < tempObjZ): #% Skip if all points are overcut
                iteration = iteration + 1
                continue
            
            pointSelectWeights = np.maximum(10e-8, tempState[:,2] - tempObjZ)
            laserPointIndex = rng.choice(tempState.shape[0], p=pointSelectWeights / np.sum(pointSelectWeights, dtype=float))
            tx = rng.choice(tx_Range)
            ty = rng.choice(ty_Range)
            
            
            
            normVec = np.array([-np.sin(ty), np.sin(tx) * np.cos(ty), -np.cos(tx) * np.cos(ty)])
            
            
            xL = tempState[laserPointIndex, 0] - normVec[0] * tempState[laserPointIndex, 2] / normVec[2]; # Sample from points that have more remaining to cut with higher probability. Project point back onto z=0 plane so that the laser is properly pointed at the xL/yL of the point with highest depth. 
            yL = tempState[laserPointIndex, 1] - normVec[1] * tempState[laserPointIndex, 2] / normVec[2]; # Sample from points that have more remaining to cut with higher probability. Project point back onto z=0 plane so that the laser is properly pointed at the xL/yL of the point with highest depth. 
            
            energyPred = (np.abs(tempState[laserPointIndex, 2] - tempObjZ[laserPointIndex]) * (self.p * self.h) + self.phi)
            energyWeights = np.exp(0.5 * (1 - np.abs(energyRange - energyPred)))
            tempInput = [xL, yL, tx, ty, rng.choice(energyRange, p = energyWeights / np.sum(energyWeights, dtype = float))]
            if method == 'gauss':
                newState = self.simulate(tempState, tempInput)
            elif method == 'supergauss':
                newState = self.simulate_SG(tempState, tempInput)
            else:
                print("Method not known!")
                return

            nonViolatedPoints = newState[np.where(tempState[:,2] > self.constraintFxn(tempState[:,0], tempState[:,1]))[0],:] # Check for new constraint violations ONLY, so that the function doesn't get "stuck" if it already made a cut that violated constraints (will just return failure for everything otherwise)
            if np.any(nonViolatedPoints[:,2] < self.constraintFxn(nonViolatedPoints[:,0], nonViolatedPoints[:,1])): # Skip if cut violates constraints
                iteration = iteration + 1
                continue
            
            # If cut does not violate constraints:
            iteration = iteration + 1
            counter = counter + 1
            objZ = self.objFxn(newState[:,0], newState[:,1])
            
            # pl = pv.Plotter()
            # pl.add_mesh(newState)
            # pl.add_mesh(np.vstack((newState[:,0], newState[:,1], objZ)).T)
            # pl.show()
            
            curCost = self.cost(newState[:,2], objZ); # Version of 2-norm with some additional scaling, for heuristic picking of search node
            newID = max(G["id"]) + 1
            G["id"].append(newID)
            G["state"].append(newState)
            G["prevState"].append(nodeID)
            G["objZ"].append(objZ)
            G["cost"].append(curCost)
            G["input"].append(tempInput)
            
            if curCost < lowestCost[1]:
                lowestCost = [newID, curCost]
            if curCost > highestCost[1]:
                highestCost = [newID, curCost]
            
            if counter >= 500:
                print("Iteration {}-{:.4f} seconds elapsed-{:.4f} 2Norm Cost".format(iteration, time.time() - start, curCost))
                counter = 0
            
            
        bestID = np.where(np.array(G["cost"]) == min(G["cost"]))[0][0]
        inputSeq = []
        while bestID != -1:
            inputSeq.append(G["input"][bestID])
            bestID = G["prevState"][bestID]
        
        return exitcode, inputSeq
    
    def objFxn(self, X, Y):
        return self.objInterp(X, Y)
        
    def constraintFxn(self, X, Y):
        return self.constraintInterp(X, Y)
    
    def cost(self, y_actual, y_pred, overcutWeight = 2):
        residual = y_actual - y_pred 
        cost = np.linalg.norm(residual * (residual > 0) + overcutWeight * residual * (residual < 0))
        return cost
    
    def simulate(self, state, u):
        '''
        Simulation with a regular Gaussian Model
        '''
        # Setting input parameters
        pt = np.array([u[0], u[1], 0]) # Laser incident position
        angle = np.array([u[2], u[3]]) # Laser incident angle in rad (x and y rotation respectively)
        energy = u[4]
        
        # Calculating simulated ablation
        normVec = np.array([-np.sin(angle[1]), np.sin(angle[0]) * np.cos(angle[1]), -np.cos(angle[0]) * np.cos(angle[1])])
        ptMat = np.repeat(pt[np.newaxis,:], state.shape[0], axis = 0)
        normVecMat = np.repeat(normVec[np.newaxis,:], state.shape[0], axis = 0)
        # https://math.stackexchange.com/questions/100761/how-do-i-find-the-projection-of-a-point-onto-a-plane
        ptProjMat = state - np.repeat((np.sum((state - ptMat) * normVecMat, axis = 1) / np.dot(normVec, normVec))[:,np.newaxis], 3, axis = 1) * normVecMat; 
        distanceVecMat = ptProjMat - ptMat
        gaussVec = np.linalg.norm(distanceVecMat, axis = 1)
        zTranslationVec = 1/(self.p * self.h) * np.maximum(0, energy * np.exp(-0.5/(self.w ** 2) * (gaussVec ** 2)) - self.phi)
        return state + normVecMat * np.repeat(zTranslationVec[:, np.newaxis], 3, axis = 1)
    
    def simulate_SG(self, state, u):
        '''
        Simulation with a Super-Gaussian Model
        '''
        # Setting input parameters
        pt = np.array([u[0], u[1], 0]) # Laser incident position
        angle = np.array([u[2], u[3]]) # Laser incident angle in rad (x and y rotation respectively)
        energy = u[4]
        
        # Calculating simulated ablation
        normVec = np.array([-np.sin(angle[1]), np.sin(angle[0]) * np.cos(angle[1]), -np.cos(angle[0]) * np.cos(angle[1])])
        ptMat = np.repeat(pt[np.newaxis,:], state.shape[0], axis = 0)
        normVecMat = np.repeat(normVec[np.newaxis,:], state.shape[0], axis = 0)
        # https://math.stackexchange.com/questions/100761/how-do-i-find-the-projection-of-a-point-onto-a-plane
        ptProjMat = state - np.repeat((np.sum((state - ptMat) * normVecMat, axis = 1) / np.dot(normVec, normVec))[:,np.newaxis], 3, axis = 1) * normVecMat; 
        distanceVecMat = ptProjMat - ptMat
        gaussVec = np.linalg.norm(distanceVecMat, axis = 1)
        zTranslationVec = 1/(self.p * self.h) * np.maximum(0, energy * np.exp(-(0.5/(self.w ** 2) * (gaussVec ** 2)) ** self.P) - self.phi)
        return state + normVecMat * np.repeat(zTranslationVec[:, np.newaxis], 3, axis = 1)
    
if __name__ == "__main__":
    x, y = np.mgrid[-2:2:0.03, -2:2:0.03]
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = np.zeros_like(x)
    state = np.hstack([x, y, z])
    objFxn = np.hstack([x, y, z])
    objFxn[np.where(np.logical_and(np.abs(objFxn[:,0]) < 2, np.abs(objFxn[:,1]) < 2))[0], 2] -= 0.8
    constraintFxn = objFxn.copy()
    constraintFxn[:,2] -= 10
    pl = pv.Plotter()
    pl.add_mesh(state, color="blue")
    pl.add_mesh(objFxn, color="red")
    pl.add_mesh(constraintFxn)
    pl.show()
    
    sim = Simulator(1, 2.5, 4.96, 0.5, 0.40456, 10, objFxn, constraintFxn)
    
    testCode = int(input("Enter Test Number: "))
    if testCode == 1:
        start = time.time()
        newState = sim.simulate(state, [0, 0, 0.1, 0.1, 0.9])
        newState = sim.simulate(newState, [-0.1, 0.1, -0.1, 0.1, 0.7])
        newState = sim.simulate(newState, [0.15, 0.1, -0.1, 0, 0.8])
        print("Time: {}".format(time.time() - start))
        pl = pv.Plotter()
        pl.add_mesh(newState, color="blue")
        #pl.add_mesh(objFxn, color="red")
        pl.show()
    if testCode == 2:
        test = sim.graphPlan(state, 0)
        pl = pv.Plotter()
        pl.add_mesh(test[1], color = "blue")
        pl.add_mesh(objFxn, color = "red")
        pl.show()