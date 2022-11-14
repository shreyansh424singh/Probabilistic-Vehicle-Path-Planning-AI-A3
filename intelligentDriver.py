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
import time
# import numpy as np

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
        # self.num_actions = len(transition_model)
        # Check reward nan to num
        self.reward_function = reward_function

        self.state_action_map = state_action_map

        self.transition_model = transition_model
        self.gamma = gamma

        # self.values = [0 for _ in range(self.num_states)]
        self.values = reward_function.copy()

        self.policy = [0 for _ in range(self.num_states)]

        for i in range(self.num_states):
            self.policy[i] = random.choice(self.state_action_map[i])


    def one_policy_evaluation(self):
        change = 0
        for s in range(self.num_states):
            # if(self.reward_function[])

            t1 = self.values[s]
            temp = 0
            act = self.policy[s]
            t_prob = 1
            temp = t_prob*(self.reward_function[act] + self.gamma*self.values[act])


            self.values[s] = temp
            change = max(change, abs(temp - t1))
        return change

    def run_policy_evaluation(self, threshold):
        change = self.one_policy_evaluation()
        c = 0
        while c<1:
            c+=1
            change = self.one_policy_evaluation()
            if change < threshold:
                return

    def run_policy_improvement(self):
        update_policy_count = 0
        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = {}

            for a in self.state_action_map[s]:
                # p = self.transition_model[s][a]
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
                update_policy_count += 1

        return update_policy_count

    def train(self, iterations, ini_state, states_map):
        c = 0
        threshold = 1e-10
        self.run_policy_evaluation(threshold)
        self.run_policy_improvement()
        while c < iterations:
            c += 1
            self.run_policy_evaluation(threshold)
            new_policy_change = self.run_policy_improvement()
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # for i in range(len(states_map)):
            #     print(f"loc: {states_map[i]} policy: {states_map[self.policy[i]]} value: {self.values[i]} ")
            # print(f"change {new_policy_change} ))))))))) ")
            if new_policy_change == 0:
                break

        # print(f"************************************\n{self.policy}\n{self.values}\n************************************")

        # for i in range(len(states_map)):
        #     x1, y1 = states_map[i]
        #     x2, y2 = states_map[ini_state]
        #     if(abs(x1-x2)+abs(y1-y2) < 4):
        #         print(f"loc: {states_map[i]} policy: {states_map[self.policy[i]]} value: {self.values[i]} reward: {self.reward_function[i]} ")
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
        # self.worldGraph = None
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

        closeness = 4

        for belief in beliefOfOtherCars:
            for row in range(self.layout.getBeliefRows()):
                for col in range(self.layout.getBeliefCols()):
                    if ((row, col) in self.states and (abs(row-currPos[0]) + abs(col-currPos[1]) < closeness)):
                        # rewards[self.states.index((row, col))] -= 50*belief.getProb(row, col) * (closeness - (abs(row-currPos[0]) + abs(col-currPos[1])))
                        rewards[self.states.index((row, col))] -= 50*belief.getProb(col, row) * (closeness - (abs(row-currPos[0]) + abs(col-currPos[1])))
                        if ((row, col+1) not in self.states) or ((row, col-1) not in self.states) or ((row+1, col) not in self.states) or ((row-1, col) not in self.states):
                            # rewards[self.states.index((row, col))] = -10
                            rewards[self.states.index((row, col))] = -4 * (closeness - (abs(row-currPos[0]) + abs(col-currPos[1])))

        
        # rewards[self.states.index(goalPos)] = 10.0
        rewards[self.states.index(goalPos)] = 15 + abs(goalPos[0]-currPos[0])**3 + abs(goalPos[1]-currPos[1])**3
        for d1 in [1, 0, -1]:
            for d2 in [1, 0, -1]:
                pos = (goalPos[0]+d1, goalPos[1]+d2)
                if pos in self.states:
                    rewards[self.states.index(pos)] += 8


        # print(f"currPos: {currPos} ckpt: {goalPos}")
        # for i in range(len(self.states)):
        #     x1, y1 = self.states[i]
        #     x2, y2 = currPos
        #     if(abs(x1-x2)+abs(y1-y2) < closeness):

        #         val = 0 
        #         for b in beliefOfOtherCars:
        #             val += b.getProb(y1, x1)
        #         if rewards[i] != 0.0:
        #             print(f"loc: {self.states[i]} reward: {rewards[i]} ")

        # print("$$$$$$$$$$$$$$$$$$$$$$$")
        # print(rewards)

        policy_ite = PolicyIteration(rewards, self.transitions, 0.1, self.action_map)

        goal_state = policy_ite.train(1000, self.states.index(currPos), self.states)

        goalPos = self.states[goal_state]

        # if abPos[0] - goalPos[0] == 1 and srr

        if abs(currPos[0] - goalPos[0]) == 1 and abs(currPos[1] - goalPos[1]) == 1:
            if (goalPos[0], currPos[1]) not in self.states:
                goalPos = (currPos[0], goalPos[1])

            if (currPos[0], goalPos[1]) not in self.states:
                goalPos = (goalPos[0], currPos[1])


        if( abs(currPos[0]-chkPos[0]) + abs(currPos[1] - chkPos[1]) < 4 and (currPos[0] == chkPos[0] or currPos[1] == chkPos[1])):
            goalPos = chkPos



        # print(f"currPos: {currPos} goal: {goalPos}")

        tt = [True, False, False]
        moveForward = random.choice(tt)

        # print(moveForward)
        # time.sleep(10000)

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
    
    

