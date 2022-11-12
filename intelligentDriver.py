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
# import numpy as np

# Class: Graph
# -------------
# Utility class
class Graph(object):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

class PolicyIteration:
    def __init__(self, reward_function, transition_model, gamma, init_policy=None):
        self.num_states = len(transition_model)
        self.num_actions = len(transition_model)
        # Check reward nan to num
        self.reward_function = reward_function

        self.transition_model = transition_model
        self.gamma = gamma

        self.values = [0 for i in range(self.num_states)]
        if init_policy is None:
            # self.policy = np.random.randint(0, self.num_actions, self.num_states)
            # go up for all states initially
            self.policy = [0 for i in range(self.num_states)]
        else:
            self.policy = init_policy

    def one_policy_evaluation(self):
        change = 0
        print({f"policy {self.policy}"})
        for s in range(self.num_states):
            temp = self.values[s]
            a = self.policy[s]
            print("$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(f's = {s}')
            print(f'a = {a}')
            p = self.transition_model[s][a]

            temp = 0
            for v in self.values:
                temp += v
            temp *= p
            self.values[s] = self.reward_function[s] + self.gamma * temp
            change = max(change, abs(temp - self.values[s]))
        return change

    def run_policy_evaluation(self, threshold):
        change = self.one_policy_evaluation()
        c = 0
        while c<30:
            c+=1
            change = self.one_policy_evaluation()
            print(change)
            if change < threshold:
                return

    def run_policy_improvement(self):
        update_policy_count = 0
        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = [0 for i in range(self.num_states)]
            for a in range(self.num_actions):

                p = self.transition_model[s][a]

                # v_list[a] = np.sum(p * self.values)
                for v in self.values:
                    v_list[a] += v
                v_list[a] *= p

            # self.policy[s] = np.argmax(v_list)
            max = v_list[0]; loc = 0
            for i in range(len(v_list)):
                if(v_list[i] > max):
                    max = v_list[0]
                    loc = i
            print(f" {loc} ))))))))((((((((((((((((((())))))))))))))))))) ")
            self.policy[s] = loc            

            if temp != self.policy[s]:
                update_policy_count += 1
        return update_policy_count

    def train(self, iterations, ini_state):
        c = 0
        threshold = 1e-3
        self.run_policy_evaluation(threshold)
        self.run_policy_improvement()
        while c < iterations:
            c += 1
            self.run_policy_evaluation(threshold)
            new_policy_change = self.run_policy_improvement()
            if new_policy_change == 0:
                break

        # return self.values, self.policy
        print(self.policy)
        print(self.values)

        return self.policy[ini_state]






# Class: IntelligentDriver
# ---------------------
# An intelligent driver that avoids collisions while visiting the given goal locations (or checkpoints) sequentially. 
class IntelligentDriver(Junior):

    # Funciton: Init
    def __init__(self, layout: Layout):
        self.burnInIterations = 30
        self.layout = layout 
        # self.worldGraph = None
        self.worldGraph = self.createWorldGraph()
        self.checkPoints = self.layout.getCheckPoints() # a list of single tile locations corresponding to each checkpoint
        self.transProb = util.loadTransProb()

        self.states = self.worldGraph.nodes.copy()
        # self.transitions = [[0 for _ in range(4)] for _ in range(len(self.states))]
        # self.trans_actions = self.transProb.keys()
        # for i in range(len(self.transitions)):
        #     x, y = self.states[i]
        #     if ((x, y), (x, y+1)) in self.transProb.keys():
        #         self.transitions[i][0] = self.transProb[((x, y), (x, y+1))]
        #     if ((x, y), (x+1, y)) in self.transProb.keys():
        #         self.transitions[i][1] = self.transProb[((x, y), (x+1, y))]
        #     if ((x, y), (x, y-1)) in self.transProb.keys():
        #         self.transitions[i][2] = self.transProb[((x, y), (x, y-1))]
        #     if ((x, y), (x-1, y)) in self.transProb.keys():
        #         self.transitions[i][3] = self.transProb[((x, y), (x-1, y))]
        self.transitions = [[0.0 for _ in range(len(self.states))] for _ in range(len(self.states))]
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if (self.states[i], self.states[j]) in self.transProb.keys():
                    self.transitions[i][j] = self.transProb[(self.states[i], self.states[j])]


                
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
        print(f'Blocks = {blocks}')
        blockTiles = []
        for block in blocks:
            row1, col1, row2, col2 = block[1], block[0], block[3], block[2] 
            # some padding to ensure the AutoCar doesn't crash into the blocks due to its size. (optional)
            row1, col1, row2, col2 = row1-1, col1-1, row2+1, col2+1
            blockWidth = col2-col1 
            blockHeight = row2-row1 

            for i in range(blockHeight):
                for j in range(blockWidth):
                    blockTile = (row1+i, col1+j)
                    blockTiles.append(blockTile)

        ## Remove blockTiles from 'nodes'
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
        print(currPos)
        goalPos = self.checkPoints[chkPtsSoFar]
        # goalPos = (util.colToX(goalPos[0]), util.rowToY(goalPos[1]))



        # print(beliefOfOtherCars)
        rewards = [0 for i in range(len(self.states))]

        for belief in beliefOfOtherCars:
            for row in range(self.layout.getBeliefRows()):
                for col in range(self.layout.getBeliefCols()):
                    if ((row, col) in self.states):
                        rewards[self.states.index((row, col))] -= 10*belief.getProb(row, col)
        
        if goalPos in self.states:
            rewards[self.states.index(goalPos)] = 1
        else:
            print("next checkpoint not present")
        # print(goalPos)
        # print(self.states)

        # print("$$$$$$$$$$$$$$$$$$$$$$$")
        # print(rewards)

        policy_ite = PolicyIteration(rewards, self.transitions, 0.1)

        goal_state = PolicyIteration.train(policy_ite, 10, self.states.index(currPos))

        goalPos = self.states[goal_state]

        print(f"currPos: {currPos} goal: {goalPos}")        

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
        # print(goalPos)
        vectorToGoal = goalPos - self.pos
        # print(vectorToGoal)
        # print((10,10) - self.pos)
        wheelAngle = -vectorToGoal.get_angle_between(self.dir)
        # print(wheelAngle)
        # print(-((10,10) - self.pos).get_angle_between(self.dir))
        # print(f"goal: {goalPos} vector {vectorToGoal} wheel angle: {wheelAngle} ")
        # print(self.dir)
        driveForward = df
        actions = {
            Car.TURN_WHEEL: wheelAngle
        }
        if driveForward:
            actions[Car.DRIVE_FORWARD] = 1.0
        # print(actions)
        return actions
    
    

