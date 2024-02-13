from __future__ import division
from tqdm import tqdm
import time
import math
import random


def UCTPolicy(node, explorationConstant):
    """UCT (Upper Confidence Bound applied to Trees) policy function."""
    # Check if the node has unexplored children
    if not node.isFullyExpanded():
        return node.expand()
    
    # Calculate the UCT value for each child node
    uctValues = [child.getUCTValue(explorationConstant) for child in node.children.values()]
    
    # Select the child node with the highest UCT value
    bestChildIndex = uctValues.index(max(uctValues))
    return node.children.values()[bestChildIndex]

def randomWeightedPolicy(state, policy_weights):
    while not state.isTerminal():
        try:
            possible_actions = state.getPossibleActions()
            #get current layer
            layer = possible_actions[0].x[1]
            #take a random action based on the weights for the current layer
            action = random.choices(possible_actions, policy_weights[layer])[0]
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()    

def randomPolicy(state, policy_weights=[0]):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()

def get_final_child(Node):
    bestChild = mcts.getBestChild(Node, Node, 0)
    while not bestChild.state.isTerminal():
        try:
            bestChild = mcts.getBestChild(bestChild, bestChild, 0)
        except IndexError:
            print('Could not reach leaf')
            break
    return  bestChild.state

class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant= 1 / (2 * math.sqrt(2)),
                 rolloutPolicy=randomPolicy, policy_weights = [0]):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.policy_weights = policy_weights

    def create_initialNode(self, initialState):
        self.root = treeNode(initialState, None)
        
    def search(self, needDetails=False):
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            pbar = tqdm(total=timeLimit)
            while time.time() < timeLimit:
                self.executeRound()
                pbar.update(1)
        else:
            verbose_iteration = self.searchLimit // 10
            pbar = tqdm(total=self.searchLimit)
            for i in range(self.searchLimit):
                self.executeRound()
                pbar.update(1)
                #if(i % verbose_iteration == 0):  
                #   for action, node in self.root.children.items():
                #     print('Child: ', node.state.current, node.totalReward)
                #   print('Best child: ', self.getBestChild(self.root, 0).state.current)
                #   print('----------------------------------------------------------------')
        pbar.close()
        bestChild = self.getBestChild(self.root, 0)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
             for action, node in self.root.children.items():
                 print('num_visits: ', node.numVisits, 'state: ', node.state.current, 'reward: ', node.totalReward)
        return get_final_child(self.root) #traverse the best tree path
             
    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state, self.policy_weights)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        #print(bestNodes[0].state.current)        
        return random.choice(bestNodes)