import util 
from util import Belief, pdf
from engine.const import Const
import collections
import math
import random
import os

# this function gets a particle dictionary
# and does samplig of particles.
def weightedRandomChoice(weightDict):
    weights = []
    elems = []
    # elem is (x,y) position
    # weightDict[elem] is number of particles at that position
    for elem in weightDict:
        weights.append(weightDict[elem])
        elems.append(elem)
    total = sum(weights)
    key = random.uniform(0, total)
    runningTotal = 0.0
    for i in range(len(weights)):
        weight = weights[i]
        runningTotal += weight
        if runningTotal > key:
            return elems[i]


# Class: Estimator
#----------------------
# Maintain and update a belief distribution over the probability of a car being in a tile.
class Estimator(object):
    def __init__(self, numRows: int, numCols: int):
        self.NUM_PARTICLES = 500

        self.belief = util.Belief(numRows, numCols) 
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly.
        # self.particles is dictionary (int, int) -> int
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()
            

    # normalize and update self.belief
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    def parked(self, agentX, agentY, observedDist):
        for row in range(self.belief.numRows):
            for col in range(self.belief.numCols):
                dist = math.sqrt((util.colToX(col) - agentX) ** 2 + (util.rowToY(row) - agentY) ** 2)
                prob_distr = util.pdf(dist, Const.SONAR_STD, observedDist)
                self.belief.setProb(row, col, self.belief.getProb(row, col) * prob_distr)
        self.belief.normalize()

        newBelief = util.Belief(self.belief.numRows, self.belief.numCols, value=0)
        for oldTile, newTile in self.transProb:
            newBelief.addProb(newTile[0], newTile[1], self.belief.getProb(*oldTile) * self.transProb[(oldTile, newTile)])
        newBelief.normalize()
        self.belief = newBelief

    def estimate(self, posX: float, posY: float, observedDist: float, isParked: bool) -> None:

        if isParked:
            self.parked(posX, posY, observedDist)
            return

        proposed = collections.defaultdict(float)
        for row, col in self.particles:
            dist = math.sqrt((util.colToX(col) - posX) ** 2 + (util.rowToY(row) - posY) ** 2)
            prob_distr = util.pdf(dist, Const.SONAR_STD, observedDist)
            proposed[(row, col)] = self.particles[(row, col)] * prob_distr
        newParticles = collections.defaultdict(int)
        for i in range(self.NUM_PARTICLES):
            particle = weightedRandomChoice(proposed)
            newParticles[particle] += 1
        self.particles = newParticles

        newParticles = collections.defaultdict(int)
        for tile, value in self.particles.items():
            if tile in self.transProbDict:
                for _ in range(value):
                    newWeightDict = self.transProbDict[tile]
                    particle = weightedRandomChoice(newWeightDict)
                    newParticles[particle] += 1
        self.particles = newParticles
        self.updateBelief()
        # END_YOUR_CODE
        return

    def getBelief(self) -> Belief:
        return self.belief

   