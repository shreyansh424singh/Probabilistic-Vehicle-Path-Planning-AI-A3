import util 
from util import Belief, pdf
from engine.const import Const
import collections
import math
import random

# this function gets a particle dictionary
# and does samplig of particles.
def sampling(prob_belief):
    list1 = []
    list2 = []
    # loc is (x,y) position
    # prob_belief[loc] is number of particles at that position
    for loc in prob_belief:
        list2.append(loc)
        list1.append(prob_belief[loc])
    f = random.uniform(0, sum(list1))
    s = 0.0
    for i in range(len(list1)):
        a = list1[i]
        s += a
        if s > f:
            return list2[i]


# Class: Estimator
#----------------------
# Maintain and update a belief distribution over the probability of a car being in a block.
class Estimator(object):
    def __init__(self, numRows: int, numCols: int):
        self.total_particles = 200

        self.belief = util.Belief(numRows, numCols) 

        self.prob = util.loadTransProb()
        self.dict_transition_prob = dict()
        for (old_block, new_block) in self.prob:
            if not old_block in self.dict_transition_prob:
                self.dict_transition_prob[old_block] = collections.defaultdict(int)
            self.dict_transition_prob[old_block][new_block] = self.prob[(old_block, new_block)]

        # Initialize the particles randomly.
        # self.map_particle is dictionary (int, int) -> int
        self.map_particle = collections.defaultdict(int)
        old_particles = list(self.dict_transition_prob.keys())
        for i in range(self.total_particles):
            i = int(random.random() * len(old_particles))
            self.map_particle[old_particles[i]] += 1

        self.normalizeBelief()
            

    # normalize and update self.belief
    def normalizeBelief(self):
        updated_Belief = Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for block in self.map_particle:
            updated_Belief.setProb(block[0], block[1], self.map_particle[block])
        updated_Belief.normalize()
        self.belief = updated_Belief

    def parked(self, posX, posY, observedDist):
        for i in range(self.belief.numRows):
            for j in range(self.belief.numCols):
                dist = math.sqrt((util.colToX(j) - posX) ** 2 + (util.rowToY(i) - posY) ** 2)
                prob_distr = pdf(dist, Const.SONAR_STD, observedDist)
                self.belief.setProb(i, j, self.belief.getProb(i, j)* self.map_particle[(i, j)] * prob_distr)
        self.belief.normalize()

        updated_Belief = util.Belief(self.belief.numRows, self.belief.numCols, value=0)
        for old_block, new_block in self.prob:
            updated_Belief.addProb(new_block[0], new_block[1], self.belief.getProb(*old_block) * self.prob[(old_block, new_block)])
        updated_Belief.normalize()
        self.belief = updated_Belief

    def estimate(self, posX: float, posY: float, observedDist: float, isParked: bool) -> None:

        if isParked:
            self.parked(posX, posY, observedDist)
            return

        prob_Belief = collections.defaultdict(float)
        for i, j in self.map_particle:
            dist = math.sqrt((util.colToX(j) - posX) ** 2 + (util.rowToY(i) - posY) ** 2)
            prob_distr = pdf(dist, Const.SONAR_STD, observedDist)
            prob_Belief[(i, j)] = self.map_particle[(i, j)] * prob_distr
        updated_belief = collections.defaultdict(int)
        for i in range(self.total_particles):
            i = sampling(prob_Belief)
            updated_belief[i] += 1
        self.map_particle = updated_belief

        updated_belief = collections.defaultdict(int)
        for block, v in self.map_particle.items():
            if block in self.dict_transition_prob:
                for _ in range(v):
                    new_prob_belief = self.dict_transition_prob[block]
                    i = sampling(new_prob_belief)
                    updated_belief[i] += 1
        self.map_particle = updated_belief
        self.normalizeBelief()
        return

    def getBelief(self) -> Belief:
        return self.belief

   