'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
import util
import itertools
from turtle import Vec2D
from engine.const import Const
from engine.vector import Vec2d
from engine.model.car.car import Car
from engine.model.layout import Layout
from engine.model.car.junior import Junior
from configparser import InterpolationMissingOptionError
import random
import math

# Class: Graph
# -------------
# Utility class
class Graph(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

class PolicyIteration:
    def __init__(self, reward_function, transition_model, gamma, state_action_map):
        self.num_states = len(transition_model)
        self.reward_function = reward_function
        self.state_action_map = state_action_map
        self.transition_model = transition_model
        self.gamma = gamma
        self.values = reward_function.copy()
        self.policy = [0 for _ in range(self.num_states)]
        for i in range(self.num_states):
            self.policy[i] = random.choice(self.state_action_map[i])


    def update_val(self):
        for s in range(self.num_states):
            temp = 0
            act = self.policy[s]
            t_prob = 1
            temp = t_prob*(self.reward_function[act] + self.gamma*self.values[act])
            self.values[s] = temp

    def update_policy(self):
        change = 0
        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = {}
            for a in self.state_action_map[s]:
                p = 1
                v_list[a] = p*(self.reward_function[a] + self.gamma*self.values[a])
            max = -1; loc = 0
            for key, val in v_list.items():
                if(val > max):
                    max = val
                    loc = key
            if loc != 0:
                self.policy[s] = loc            

            if temp != self.policy[s]:
                change += 1

        return change

    def train(self, iterations, ini_state, states_map):
        c = 0
        self.update_val()
        self.update_policy()
        while c < iterations:
            c += 1
            self.update_val()
            new_policy_change = self.update_policy()
            if new_policy_change == 0:
                break

        return self.policy[ini_state]

# Class: IntelligentDriver
# ---------------------
# An intelligent driver that avoids collisions while visiting the given goal locations (or checkpoints) sequentially. 
class IntelligentDriver(Junior):

    # Funciton: Init
    def __init__(self, layout: Layout):
        self.burnInIterations = 30
        self.layout = layout
        self.states = []
        self.worldGraph = self.createWorldGraph()
        self.checkPoints = self.layout.getCheckPoints() # a list of single tile locations corresponding to each checkpoint
        self.transProb = util.loadTransProb()
        self.transitions = [[0.0 for _ in range(len(self.states))] for _ in range(len(self.states))]

        c=0
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if (self.states[i], self.states[j]) in self.transProb.keys():
                    self.transitions[i][j] = self.transProb[(self.states[i], self.states[j])]
                    if self.transitions[i][j] > 0: 
                        c+=1

        self.action_map = []
        for i in range(len(self.states)):
            x, y = self.states[i]
            actions = []
            if (x, y+1) in self.states:
                actions.append(self.states.index((x, y+1)))
            if (x+1, y) in self.states:
                actions.append(self.states.index((x+1, y)))
            if (x, y-1) in self.states:
                actions.append(self.states.index((x, y-1)))
            if (x-1, y) in self.states:
                actions.append(self.states.index((x-1, y)))
            if (x+1, y+1) in self.states:
                actions.append(self.states.index((x+1, y+1)))
            if (x+1, y-1) in self.states:
                actions.append(self.states.index((x+1, y-1)))
            if (x-1, y+1) in self.states:
                actions.append(self.states.index((x-1, y+1)))
            if (x-1, y-1) in self.states:
                actions.append(self.states.index((x-1, y-1)))
            self.action_map.append(actions)


                
    # ONE POSSIBLE WAY OF REPRESENTING THE GRID WORLD. FEEL FREE TO CREATE YOUR OWN REPRESENTATION.
    # Function: Create World Graph
    # ---------------------
    # Using self.layout of IntelligentDriver, create a graph representing the given layout.
    def createWorldGraph(self):
        nodes = []
        edges = []
        # create self.worldGraph using self.layout
        numRows, numCols = self.layout.getBeliefRows(), self.layout.getBeliefCols()

        # NODES #
        ## each tile represents a node
        nodes = [(x, y) for x, y in itertools.product(range(numRows), range(numCols))]
        
        # EDGES #
        ## We create an edge between adjacent nodes (nodes at a distance of 1 tile)
        ## avoid the tiles representing walls or blocks#
        ## YOU MAY WANT DIFFERENT NODE CONNECTIONS FOR YOUR OWN IMPLEMENTATION,
        ## FEEL FREE TO MODIFY THE EDGES ACCORDINGLY.

        ## Get the tiles corresponding to the blocks (or obstacles):
        blocks = self.layout.getBlockData()
        blockTiles = []
        bT_new = []
        for block in blocks:
            row1, col1, row2, col2 = block[1], block[0], block[3], block[2] 
            row11, col11, row12, col12 = block[1], block[0], block[3], block[2] 
            # some padding to ensure the AutoCar doesn't crash into the blocks due to its size. (optional)
            row1, col1, row2, col2 = row1-1, col1-1, row2+1, col2+1
            blockWidth = col2-col1 
            blockHeight = row2-row1 

            blockWidth1 = col12-col11 
            blockHeight1 = row12-row11 

            for i in range(blockHeight):
                for j in range(blockWidth):
                    blockTile = (row1+i, col1+j)
                    blockTiles.append(blockTile)

            for i in range(blockHeight1):
                for j in range(blockWidth1):
                    bT = (row11+i, col11+j)
                    bT_new.append(bT)

        ## Remove blockTiles from 'nodes'
        self.states = [(y, x) for (x, y) in nodes if (x, y) not in bT_new]
        nodes = [x for x in nodes if x not in blockTiles]

        
        for node in nodes:
            x, y = node[0], node[1]
            adjNodes = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
            
            # only keep allowed (within boundary) adjacent nodes
            adjacentNodes = []
            for tile in adjNodes:
                if tile[0]>=0 and tile[1]>=0 and tile[0]<numRows and tile[1]<numCols:
                    if tile not in blockTiles:
                        adjacentNodes.append(tile)

            for tile in adjacentNodes:
                edges.append((node, tile))
                edges.append((tile, node))
        return Graph(nodes, edges)

    #######################################################################################
    # Function: Get Next Goal Position
    # ---------------------
    # Given the current belief about where other cars are and a graph of how
    # one can driver around the world, chose the next position.
    #######################################################################################
    def getNextGoalPos(self, beliefOfOtherCars: list, parkedCars:list , chkPtsSoFar: int):
        '''
        Input:
        - beliefOfOtherCars: list of beliefs corresponding to all cars
        - parkedCars: list of booleans representing which cars are parked
        - chkPtsSoFar: the number of checkpoints that have been visited so far 
                       Note that chkPtsSoFar will only be updated when the checkpoints are updated in sequential order!
        
        Output:
        - goalPos: The position of the next tile on the path to the next goal location.
        - moveForward: Unset this to make the AutoCar stop and wait.

        Notes:
        - You can explore some files "layout.py", "model.py", "controller.py", etc.
         to find some methods that might help in your implementation. 
        '''
        moveForward = True

        currPos = self.getPos() # the current 2D location of the AutoCar (refer util.py to convert it to tile (or grid cell) coordinate)
        currPos = (util.xToCol(currPos[0]), util.yToRow(currPos[1]))
        goalPos = self.checkPoints[chkPtsSoFar]
        goalPos = (goalPos[1], goalPos[0])
        chkPos = self.checkPoints[chkPtsSoFar]
        chkPos = (chkPos[1], chkPos[0])

        rewards = [0 for i in range(len(self.states))]

        closeness = 3

        for belief in beliefOfOtherCars:
            for row in range(self.layout.getBeliefRows()):
                for col in range(self.layout.getBeliefCols()):
                    if ((row, col) in self.states and math.sqrt(abs(row-currPos[0])**2 + abs(col-currPos[1])**2 < closeness)):
                        # rewards[self.states.index((row, col))] -= 50*belief.getProb(row, col) * (closeness - (abs(row-currPos[0]) + abs(col-currPos[1])))
                        rewards[self.states.index((row, col))] -= 50*belief.getProb(col, row) * (closeness - (abs(row-currPos[0]) + abs(col-currPos[1])))
                        if (row+1, col) in self.states:
                            rewards[self.states.index((row+1, col))] -= 5*belief.getProb(col, row+1) * (closeness - (abs(row+1-currPos[0]) + abs(col-currPos[1])))
                        if (row-1, col) in self.states:
                            rewards[self.states.index((row-1, col))] -= 5*belief.getProb(col, row-1) * (closeness - (abs(row-1-currPos[0]) + abs(col-currPos[1])))
                        if (row, col+1) in self.states:
                            rewards[self.states.index((row, col+1))] -= 5*belief.getProb(col+1, row) * (closeness - (abs(row-currPos[0]) + abs(col+1-currPos[1])))
                        if (row, col-1) in self.states:
                            rewards[self.states.index((row, col-1))] -= 5*belief.getProb(col-1, row) * (closeness - (abs(row-currPos[0]) + abs(col-1-currPos[1])))
                        # if (row+1, col+1) in self.states:
                        #     rewards[self.states.index((row+1, col+1))] -= 10*belief.getProb(col+1, row+1) * (closeness - (abs(row+1-currPos[0]) + abs(col+1-currPos[1])))
                        # if (row-1, col-1) in self.states:
                        #     rewards[self.states.index((row-1, col-1))] -= 10*belief.getProb(col-1, row-1) * (closeness - (abs(row-1-currPos[0]) + abs(col-1-currPos[1])))
                        # if (row-1, col+1) in self.states:
                        #     rewards[self.states.index((row-1, col+1))] -= 10*belief.getProb(col+1, row-1) * (closeness - (abs(row-1-currPos[0]) + abs(col+1-currPos[1])))
                        # if (row+1, col-1) in self.states:
                        #     rewards[self.states.index((row+1, col-1))] -= 10*belief.getProb(col-1, row+1) * (closeness - (abs(row+1-currPos[0]) + abs(col-1-currPos[1])))
    
        
        rewards[self.states.index(goalPos)] = 15 + abs(goalPos[0]-currPos[0])**2 + abs(goalPos[1]-currPos[1])**2
        if self.layout.getBeliefRows()*self.layout.getBeliefCols() < 300:
            rewards[self.states.index(goalPos)] = 10.0

        l = [1, 0, -1]
        if(goalPos == (0, 0)): l = [2, 1, 0, -1, -2]
        for d1 in l:
            for d2 in l:
                pos = (goalPos[0]+d1, goalPos[1]+d2)
                if pos in self.states:
                    rewards[self.states.index(pos)] += 8

        policy_ite = PolicyIteration(rewards, self.transitions, 0.1, self.action_map)

        goal_state = policy_ite.train(1000, self.states.index(currPos), self.states)

        goalPos = self.states[goal_state]

        if abs(currPos[0] - goalPos[0]) == 1 and abs(currPos[1] - goalPos[1]) == 1:
            if (goalPos[0], currPos[1]) not in self.states:
                goalPos = (currPos[0], goalPos[1])

            if (currPos[0], goalPos[1]) not in self.states:
                goalPos = (goalPos[0], currPos[1])


        if( abs(currPos[0]-chkPos[0]) + abs(currPos[1] - chkPos[1]) < 4 and (currPos[0] == chkPos[0] == 0 or currPos[1] == chkPos[1] == 0 or (currPos[0]==1 and chkPos[0]==0) or (currPos[1]==1 and chkPos[1]==0))):
            goalPos = chkPos


        tt = [True, False, False]
        if self.layout.getBeliefRows()*self.layout.getBeliefCols() <=625:
            tt = [True, False, False, False, False]
        moveForward = random.choice(tt)
        # try:
        #     if belief.getProb(currPos[1], currPos[0]) > belief.getProb(currPos[1], currPos[0]+1) + belief.getProb(currPos[1], currPos[0]-1) + belief.getProb(currPos[1]+1, currPos[0]) + belief.getProb(currPos[1]-1, currPos[0]) +belief.getProb(currPos[1]+1, currPos[0]) + belief.getProb(currPos[1], currPos[0]-1):
        #         moveForward = True
        # except:
        #     moveForward = random.choice(tt)



        goalPos = (util.colToX(goalPos[0]), util.rowToY(goalPos[1]))
        return goalPos, moveForward

    # DO NOT MODIFY THIS METHOD !
    # Function: Get Autonomous Actions
    # --------------------------------
    def getAutonomousActions(self, beliefOfOtherCars: list, parkedCars: list, chkPtsSoFar: int):
        # Don't start until after your burn in iterations have expired
        if self.burnInIterations > 0:
            self.burnInIterations -= 1
            return[]
       
        goalPos, df = self.getNextGoalPos(beliefOfOtherCars, parkedCars, chkPtsSoFar)
        vectorToGoal = goalPos - self.pos
        wheelAngle = -vectorToGoal.get_angle_between(self.dir)
        driveForward = df
        actions = {
            Car.TURN_WHEEL: wheelAngle
        }
        if driveForward:
            actions[Car.DRIVE_FORWARD] = 1.0
        return actions
    
    
