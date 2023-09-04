# myTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import heapq
# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import math
from ast import Raise
from typing import List, Tuple

from numpy import true_divide
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent,Grid
from util import nearestPoint
import sys,os

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
                             first = 'MixedAgent', second = 'MixedAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class Node:
    def __init__(self,position,t,f_value = 0):
        self.position = position
        self.t = t
        self.g = t
        self.f = f_value
        self.parent = None

    def __lt__(self,other):
        return self.f < other.f

    def __hash__(self):
        return hash((self.position[0],self.position[1],self.t))
    def __eq__(self,other):
        if not isinstance(other,Node):
            return False
        return self.position[0] == other.position[0] and self.position[1] == other.position[1] and self.t == other.t



class MixedAgent(CaptureAgent):
    """
    This is an agent that use pddl to guide the high level actions of Pacman
    """
    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {
            "offensiveWeights":{'closest-food': -1, 
                                        'bias': 1, 
                                        '#-of-ghosts-1-step-away': -100, 
                                        'successorScore': 100, 
                                        'chance-return-food': 10,
                                        },
            "defensiveWeights": {'numInvaders': -1000, 'onDefense': 100,'teamDistance':0 ,'invaderDistance': -10, 'stop': -100, 'reverse': -2},
            "escapeWeights": {'onDefense': 1000, 'enemyDistance': 30, 'stop': -100, 'distanceToHome': -20}
        }
    QLWeightsFile = BASE_FOLDER+'/QLWeightsMyTeam_update.txt'

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}


    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER+'/myTeam.pddl')
        self.highLevelPlan: List[Tuple[Action,pddl_state]] = None # Plan is a list Action and pddl_state
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = 0 # index of action in self.highLevelPlan should be execute next

        self.startPosition = gameState.getAgentPosition(self.index) # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str,Tuple]] = []
        self.lowLevelActionIndex = 0



        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.trainning = True # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.1 #default exploration prob, change to take a random step
        self.alpha = 0.02 #default learning rate
        self.discountRate = 0.9 # default discount rate on successor state q value when update
        
        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index]={}
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION

        """
        if os.path.exists(MixedAgent.QLWeightsFile):
            with open(MixedAgent.QLWeightsFile, "r") as file:
                MixedAgent.QLWeights = eval(file.read())
            print("Load QLWeights:",MixedAgent.QLWeights )
        
    
    def final(self, gameState : GameState):
        """
        This function write weights into files after the game is over. 
        You may want to comment (disallow) this function when submit to contest server.
        """
        if self.trainning:
            print("Write QLWeights:", MixedAgent.QLWeights)
            file = open(MixedAgent.QLWeightsFile, 'w')
            file.write(str(MixedAgent.QLWeights))
            file.close()
    

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action ("North", "South", "East", "West", "Stop") to achieve the high-level action.
        """

        #-------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects,initState)

        # Check if we can stick to current plan 
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agnet:",self.index,"compute plan:")
            print("\tOBJ:"+str(objects),"\tINIT:"+str(initState), "\tPOSITIVE_GOAL:"+str(positiveGoal), "\tNEGTIVE_GOAL:"+str(negtiveGoal),sep="\n")
            self.highLevelPlan: List[Tuple[Action,pddl_state]] = self.getHighLevelPlan(objects, initState,positiveGoal, negtiveGoal) # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = [] # reset low level plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            print("\tPLAN:",self.highLevelPlan)
        if len(self.highLevelPlan)==0:
            raise Exception("Solver retuned empty plan, you need to think how you handle this situation or how you modify your model ")
        
        # Get next action from the plan
        highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction
        print("Agent:", self.index, highLevelAction)

        #-------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            self.lowLevelPlan = self.getLowLevelPlanHS(gameState , highLevelAction)
            #self.lowLevelPlan = self.getLowLevelPlanQL(gameState, highLevelAction) #Generate low level plan with q learning
            # you can replace the getLowLevelPlanQL with getLowLevelPlanHS and implement heuristic search planner
            self.lowLevelActionIndex = 0
        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        self.lowLevelActionIndex+=1
        print("\tAgent:", self.index,lowLevelAction)
        return lowLevelAction

    #------------------------------- PDDL and High-Level Action Functions ------------------------------- 
    
    
    def getHighLevelPlan(self, objects, initState, positiveGoal, negtiveGoal) -> List[Tuple[Action,pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        self.pddl_solver.parser_.reset_problem()
        self.pddl_solver.parser_.set_objects(objects)
        self.pddl_solver.parser_.set_state(initState)
        self.pddl_solver.parser_.set_negative_goals(negtiveGoal)
        self.pddl_solver.parser_.set_positive_goals(positiveGoal)
        
        # Solve the problem and return the plan
        return self.pddl_solver.solve()

    def get_pddl_state(self,gameState:GameState) -> Tuple[List[Tuple],List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        # Collect objects and states from the gameState

        states = []
        objects = []


        # Collect available foods on the map
        foodLeft = self.getFood(gameState).asList()
        if len(foodLeft) > 0:
            states.append(("food_available",))
        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)
        cloestFoodDist = self.closestFood(myPos,self.getFood(gameState), gameState.getWalls())
        if cloestFoodDist != None and cloestFoodDist <=CLOSE_DISTANCE:
            states.append(("near_food",myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0 :
            states.append(("capsule_available",))
        for cap in capsules:
            if self.getMazeDistance(cap,myPos) <=CLOSE_DISTANCE:
                states.append(("near_capsule",myObj))
                break
        
        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore> 3:
                states.append(("winning_gt3",))
            if currentScore> 5:
                states.append(("winning_gt5",))
            if currentScore> 10:
                states.append(("winning_gt10",))
            if currentScore> 20:
                states.append(("winning_gt20",))
        else:
            if currentScore < 0:
                states.append(("winning",))
            if currentScore < -3:
                states.append(("winning_gt3",))
            if currentScore < -5:
                states.append(("winning_gt5",))
            if currentScore < -10:
                states.append(("winning_gt10",))
            if currentScore < -20:
                states.append(("winning_gt20",))

        # Collect team agents states
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents :
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if agent_index != self.index and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agent_index)) <= CLOSE_DISTANCE:
                states.append(("near_ally",))
            
            if agent_state.scaredTimer>0:
                states.append(("is_scared",agent_object))

            if agent_state.numCarrying>0:
                states.append(("food_in_backpack",agent_object))
                if agent_state.numCarrying >=20 :
                    states.append(("20_food_in_backpack",agent_object))
                if agent_state.numCarrying >=10 :
                    states.append(("10_food_in_backpack",agent_object))
                if agent_state.numCarrying >=5 :
                    states.append(("5_food_in_backpack",agent_object))
                if agent_state.numCarrying >=3 :
                    states.append(("3_food_in_backpack",agent_object))

            if agent_state.numCarrying<5:
                states.append(("food_less_than_5",agent_object))
            if agent_state.isPacman:
                states.append(("is_pacman",agent_object))
            
            

        # Collect enemy agents states
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer>0:
                states.append(("is_scared",enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if self.getMazeDistance(agent_state.getPosition(), enemy_position) <= CLOSE_DISTANCE:
                        states.append(("enemy_around",enemy_object, "a{}".format(agent_index)))
            else:
                if noisyDistance[enemy_index] >=LONG_DISTANCE :
                    states.append(("enemy_long_distance",enemy_object, "a{}".format(self.index)))
                elif noisyDistance[enemy_index] >=MEDIUM_DISTANCE :
                    states.append(("enemy_medium_distance",enemy_object, "a{}".format(self.index)))
                else:
                    states.append(("enemy_short_distance",enemy_object, "a{}".format(self.index)))                                                                                                                                                                                                 


            if enemy_state.isPacman:
                states.append(("is_pacman",enemy_object))
            typeIndex += 1
            
        return objects, states
    
    def stateSatisfyCurrentPlan(self, init_state: List[Tuple],positiveGoal, negtiveGoal):
        if self.highLevelPlan is None or len(self.highLevelPlan) == 0:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False
        
        if positiveGoal != self.currentPositiveGoalStates or negtiveGoal != self.currentNegativeGoalStates:
            return False
        
        if self.pddl_solver.matchEffect(init_state, self.highLevelPlan[self.currentActionIndex][0] ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(self.highLevelPlan) -1 and self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex+1][0]):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = [] # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            # Current action precondition satisfied, continue executing current action of the plan
            return True
        
        # Current action precondition not satisfied anymore, need new plan
        return False
    
    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        # Check a list of goal functions from high priority to low priority if the goal is applicable
        # Return the pddl goal states for selected goal function
        if (("winning_gt10",) in initState):
            return self.goalDefWinning(objects, initState)
        else:
            return self.goalScoring(objects, initState)

    def goalScoring(self,objects: List[Tuple], initState: List[Tuple]):
        # If we are not winning more than 5 points,
        # we invate enemy land and eat foods, and bring then back.

        positiveGoal = []
        negtiveGoal = []
        obj_curr =  [obj[0] for obj in objects if obj[1]=="current_agent"][0]
        if ("5_food_in_backpack",obj_curr) in initState:
            negtiveGoal += [("is_pacman",obj_curr)]
            return positiveGoal,negtiveGoal

        for obj in objects:
            agent_obj = obj[0]
            agent_type =obj[1]
            if agent_type == "current_agent":
                negtiveGoal += [("food_less_than_5",agent_obj)] # no food avaliable means eat all the food

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.
        
        return positiveGoal, negtiveGoal

    def goalDefWinning(self,objects: List[Tuple], initState: List[Tuple]):
        # If winning greater than 5 points,
        # this example want defend foods only, and let agents patrol on our ground.
        # The "win_the_game" pddl state is only reachable by the "patrol" action in pddl,
        # using it as goal, pddl will generate plan eliminate invading enemy and patrol on our ground.

        positiveGoal = [("defend_foods",)]
        negtiveGoal = []
        
        return positiveGoal, negtiveGoal

    #------------------------------- Heuristic search low level plan Functions -------------------------------
    def getsvalidsuccessor(self,current_node: Node,map,ghosts_obstacle,goal_pos,w):
        dx = [-1,0,1,0,0]
        dy = [0,1,0,-1,0]
        successor_list = []
        for ddx,ddy in zip(dx,dy):
            xx = current_node.position[0] + ddx
            yy = current_node.position[1] + ddy
            if map[xx][yy] == True  or (xx,yy) in ghosts_obstacle:
                continue
            else:
                tmp_node = Node((xx,yy),current_node.t+1,current_node.t+1+w*self.getMazeDistance((xx,yy),goal_pos))
                successor_list.append(tmp_node)
        return successor_list

    def get_sorrounded(self,gamestate,map,ghosts):
        dx = [-1,0,1,0]
        dy = [0,-1,0,1]
        pos_cur = gamestate.getAgentPosition(self.index)
        count_sorrounded = 0
        for i in range(len(dx)):
            if (pos_cur[0]+dx[i],pos_cur[1]+dy[i]) in ghosts or map[pos_cur[0]+dx[i]][pos_cur[1]+dy[i]] == True:
                count_sorrounded+=1
        if count_sorrounded==4:
            return True
        return False

    def quick_move(self,gamestate,map,ghosts):
        dx = [-1,0,1,0]
        dy = [0,-1,0,1]
        pos_cur = gamestate.getAgentPosition(self.index)
        path = [pos_cur]
        next_pos = pos_cur
        min_dist_all = map.width * map.height
        for i in range(len(dx)):
            if (pos_cur[0]+dx[i],pos_cur[1]+dy[i]) in ghosts or map[pos_cur[0]+dx[i]][pos_cur[1]+dy[i]] == True:
                pass
            else:
                dist_all = 0
                for g in ghosts:
                    dist_all += abs(pos_cur[0]+dx[i]-g[0]) + abs(pos_cur[1]+dy[i]-g[1])
                if dist_all<=min_dist_all:
                    next_pos = (pos_cur[0]+dx[i],pos_cur[1]+dy[i])
                    min_dist_all = dist_all

        path.append(next_pos)

        return path


    def attack_mode_hs(self,gamestate:GameState ,map,foods):
        """
        map  : a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        food : a 2d array matrix of food,  foods[x][y] = true if there's a food.

        "" Modified 1.01 : 1. If sorrounded , solution will slow . It can be fixed pre.
                           2. After eaten the capsules , there is no need for afraid the ghost.
                           3. Add quick_move() function to solve the problem that there is the probablity of have no path for the food or some specific loc

            Modified 1.02 : 1. If Two agents aim to the same loc of the cloest food , two will be cathed together      # Excellent update

        """
        # consider two situation : 1. Ghost state : from start to food 2. Pacman state : from food to next food
        start_pos = gamestate.getAgentPosition(self.index)
        # Get the closest food location as the goal
        map_width = map.width
        map_height = map.height
        min_maze_distance = map_height * map_height
        goal_pos = (0,0)
        search_food_mode = True
        if start_pos[0] > int(map_width/2)-1:
            for i in range(map_width):
                for j in range(map_height):
                    if foods[i][j]:
                        if self.getMazeDistance(start_pos,(i,j)) <= min_maze_distance:
                            goal_pos = (i,j)
                            min_maze_distance = self.getMazeDistance(start_pos,(i,j))
        else:
            search_food_mode = False
            if self.index == 0:
                for i in list(range(map.height))[::-1]:
                    if map[int(map_width/2)][i] == False:
                        goal_pos = (int(map_width/2),i)
                        break
            elif self.index == 2:
                for i in range(map_height):
                    if map[int(map_width/2)][i] == False:
                        goal_pos = (int(map_width/2),i)
                        break
        # If enemies is around then avoid
        # For the reason that the cannot predict the enemies next step ,so the current pos of ghost will be the still obstacle
        print('goal_pos:',goal_pos)
        print('dist to food :',min_maze_distance)
        enemies = [gamestate.getAgentState(i) for i in self.getOpponents(gamestate)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0] # Modify 1

        ghosts_obstacle = []
        if len(ghosts) > 0:
            for g in ghosts:
                ghosts_obstacle.append(g.getPosition())

        if search_food_mode:
            for i in range(int(map_width/2)):
                for j in range(map_height):
                    ghosts_obstacle.append((i,j))

        if self.get_sorrounded(gamestate,map,ghosts_obstacle):
            return []

        path = self.WA_star_for_search(gamestate,map,start_pos,goal_pos,ghosts_obstacle)
        print('ghosts:',ghosts_obstacle)
        print('search: ',search_food_mode)
        print('path:',path)
        return path
        # A* search start
    def goback_mode_hs(self,gamestate,map):
        start_pos = gamestate.getAgentPosition(self.index)
        goal_pos = self.startPosition
        enemies = [gamestate.getAgentState(i) for i in self.getOpponents(gamestate)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]  # Modify 1

        ghosts_obstacle = []
        if len(ghosts) > 0:
            for g in ghosts:
                ghosts_obstacle.append(g.getPosition())
        path = self.WA_star_for_search(gamestate,map,start_pos,goal_pos,ghosts_obstacle)
        return path

    def defence_mode_hs(self,gamestate,map,foodneeddefend):
        """
        foodneeddefend:food will be eatan by enemy (food next to enemy)
        In this mode, you only need to focus on defense, and go directly to attack mode if the capsule is eaten.

        """
        start_pos = gamestate.getAgentPosition(self.index)
        goal_pos = (0,0)
        map_width = map.width
        map_height = map.height
        # 3-levels goals , level 1: border position
        if self.index == 0:
            for i in list(range(map.height))[::-1]:
                if map[int(map_width / 2)][i] == False:
                    goal_pos = (int(map_width / 2), i)
                    break
        elif self.index == 2:
            for i in range(map_height):
                if map[int(map_width / 2)][i] == False:
                    goal_pos = (int(map_width / 2), i)
                    break
        # level 2:food will be eaten
        # Check the history of the state to saerch the latest food location to protect
        food_alarm_list = []
        for prestate in self.observationHistory[::-2]:
            foodneeddefendprev = self.getFoodYouAreDefending(prestate)
            for i in range(map_width):
                for j in range(map_height):
                    if foodneeddefendprev[i][j] and not foodneeddefend[i][j]:
                        food_alarm_list.append((i, j))
            if len(food_alarm_list) > 0:
                break

        if len(food_alarm_list) == 0 :
            pass
        elif len(food_alarm_list) ==1:
            goal_pos = food_alarm_list[0]
        else:
            goal_pos = food_alarm_list[int(self.index//2)]

        print('goal pos',goal_pos)
        # level 3: ghosts near
        enemies = [gamestate.getAgentState(i) for i in self.getOpponents(gamestate)]
        ghosts = [a for a in enemies if  a.isPacman and a.getPosition() != None]
        if len(ghosts)>0:
            if len(ghosts) == 1:
                goal_pos = ghosts[0].getPosition()
            else:
                goal_pos = ghosts[int(self.index//2)].getPosition()

        path = self.WA_star_for_search(gamestate,map,start_pos,goal_pos,[])
        return path
    def WA_star_for_search(self,gamestate:GameState ,map,start_pos,goal_pos,ghosts_obstacle):
        """
        map  : a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        food : a 2d array matrix of food,  foods[x][y] = true if there's a food.
        ghosts_obstacle : list of (x,y) ,means the location of the ghosts
        """
        openlist = [] # Priority Queue
        g_score = {}
        closedic = {}

        w = 1
        start_node = Node(start_pos,0,w * self.getMazeDistance(start_pos,goal_pos))
        heapq.heappush(openlist,(start_node.f,start_node))

        g_score[start_node] = start_node
        start_time = time.time()
        while g_score:
            end_time = time.time()
            if end_time-start_time>=1:
                return self.quick_move(gamestate,map,ghosts_obstacle)
            _,current = heapq.heappop(openlist)
            del g_score[current]
            closedic[current] = current
            if current.position == goal_pos:
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent
                path = path[::-1]
                return path
            for s in self.getsvalidsuccessor(current,map,ghosts_obstacle,goal_pos,w):
                if s in closedic:
                    continue
                s.parent = current
                if s in g_score:
                    pre_s = g_score[s]
                    if s.g<pre_s.g:
                        pre_s.g = s.g
                        pre_s.f = s.f
                        pre_s.parent = s.parent
                        heapq.heapify(openlist)
                else:
                    heapq.heappush(openlist,(s.f,s))
                    g_score[s] = s
        return []



    def getdirect(self,loc_after,loc_before):
        if loc_after[0]-loc_before[0] == 0 and loc_after[1] - loc_before[1] ==1:
            return "North"
        elif loc_after[0]-loc_before[0] == 0 and loc_after[1] - loc_before[1] == -1:
            return "South"
        elif loc_after[0]-loc_before[0] == 1 and loc_after[1] - loc_before[1] == 0:
            return "East"
        elif loc_after[0]-loc_before[0] == -1 and loc_after[1] - loc_before[1] == 0:
            return "West"
        else:
            return "Stop"
    def getLowLevelPlanHS(self, gameState: GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.
        #A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}
        map = gameState.getWalls() # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        foods = self.getFood(gameState) # a 2d array matrix of food,  foods[x][y] = true if there's a food.
        capsules = self.getCapsules(gameState) # a list of capsules
        foodNeedDefend = self.getFoodYouAreDefending(gameState) # return food will be eatan by enemy (food next to enemy)
        capsuleNeedDefend = self.getCapsulesYouAreDefending(gameState) # return capsule will be eatan by enemy (capsule next to enemy)

        path = []
        if highLevelAction == "attack":
            path = self.attack_mode_hs(gameState,map,foods)
        elif highLevelAction == 'go_home' or highLevelAction == 'go_home2' or highLevelAction == 'go_home3':
            path = self.goback_mode_hs(gameState,map)
        elif highLevelAction == 'defence' or highLevelAction == 'defence2':
            path = self.defence_mode_hs(gameState,map,foodNeedDefend)
        else:
            path = self.attack_mode_hs(gameState, map, foods)
        if path == []:
            return [("Stop",gameState.getAgentPosition(self.index))]
        elif len(path) == 1:
            return [(self.getdirect(path[0],path[0]),path[0])]
        else:
            return [(self.getdirect(path[1],path[0]),path[1])]


        return [] # You should return a list of tuple of move action and target location (exclude current location).


    def posSatisfyLowLevelPlan(self,gameState: GameState):
        if self.lowLevelPlan == None or len(self.lowLevelPlan)==0 or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,self.lowLevelPlan[self.lowLevelActionIndex][0])
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True

    #------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """
    def getLowLevelPlanQL(self, gameState:GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        values = []
        legalActions = gameState.getLegalActions(self.index)
        rewardFunction = None
        featureFunction = None
        weights = None
        learningRate = 0

        ##########
        # The following classification of high level actions is only a example.
        # You should think and use your own way to design low level planner.
        ##########
        if highLevelAction == "attack":
            # The q learning process for offensive actions are complete, 
            # you can improve getOffensiveFeatures to collect more useful feature to pass more information to Q learning model
            # you can improve the getOffensiveReward function to give reward for new features and improve the trainning process .
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        elif highLevelAction == "go_home":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update,
        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon) # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                        if self.trainning:
                            self.updateWeights(gameState, action, rewardFunction, featureFunction, weights,learningRate)
                        values.append((self.getQValue(featureFunction(gameState, action), weights), action))
                action = max(values)[1]
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,action)
        return [(action, nextPos)]


    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """
    def getQValue(self, features, weights):
        return features * weights
    
    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def updateWeights(self, gameState, action, rewardFunction, featureFunction, weights, learningRate):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState, featureFunction, weights)) - self.getQValue(features, weights)
            weights[feature] =weights[feature] + learningRate*correction * features[feature]
        
    
    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, nextState: GameState, featureFunction, weights):
        qVals = []
        legalActions = nextState.getLegalActions(self.index)

        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                features = featureFunction(nextState, action)
                qVals.append(self.getQValue(features,weights))
            return max(qVals)
    
    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        # Calculate the reward. 
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(g,gameState.getWalls()) for g in ghosts)

        base_reward =  -50 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)

        if ghost_1_step > 0:
            base_reward -= 5
        if score <0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned*10
        
        print("Agent ", self.index," reward ",base_reward)
        return base_reward
    
    def getDefensiveReward(self,gameState:GameState, nextState:GameState):

        currentAgentState: AgentState = gameState.getAgentState(self.index)
        nextAgentState: AgentState = nextState.getAgentState(self.index)
        cur_pos = currentAgentState.getPosition()
        after_pos = nextAgentState.getPosition()
        base_reward = -50

        #Consider the ondefence
        if (currentAgentState.isPacman and not nextAgentState.isPacman) or (not currentAgentState.isPacman and not nextAgentState.isPacman):
            base_reward += 1


        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        pacmans = [a for a in enemies if a.isPacman and a.getPosition() != None]
        for p in pacmans:
            if self.getMazeDistance(cur_pos,p.getPosition()) == 1 and after_pos == p.getPosition():
                base_reward += 50
            elif self.getMazeDistance(cur_pos,p.getPosition()) > 1 and self.getMazeDistance(after_pos,p.getPosition()) <= self.getMazeDistance(cur_pos,p.getPosition()):
                base_reward += 4

        if cur_pos == after_pos:
            base_reward -= 20


        print("Agent ", self.index, " reward ", base_reward)
        #print("Warnning: DefensiveReward not implemented yet, and learnning rate is 0 for defensive ",file=sys.stderr)
        return base_reward
    
    def getEscapeReward(self,gameState, nextState):
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)
        cur_pos = currentAgentState.getPosition()
        after_pos = nextAgentState.getPosition()

        #Encourages movement and imposes penalties for staying in place.
        base_reward = -50
        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(g, gameState.getWalls()) for g in ghosts)
        base_reward -= ghost_1_step*5
        #Similar to the reward composition of the attack mode, but the escape is more focused on evading the opponent.

        if cur_pos == after_pos:
            base_reward -= 50
        if self.getMazeDistance(cur_pos, self.startPosition) > self.getMazeDistance(cur_pos, self.startPosition):
            base_reward += 3
        if not nextAgentState.isPacman:
            base_reward += 30

        print("Agent ", self.index, " reward ", base_reward)
        #print("Warnning: EscapeReward not implemented yet, and learnning rate is 0 for escape",file=sys.stderr)
        return base_reward



    #------------------------------- Feature Related Action Functions -------------------------------


    
    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState) 
        currAgentState = gameState.getAgentState(self.index)

        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)
        
        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(nextState)/(walls.width+walls.height) * 10


        # Bias
        features["bias"] = 1.0
        
        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts) 
        
        
        dist_home =  self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1

        features["chance-return-food"] = (currAgentState.numCarrying)*(1 - dist_home/(walls.width+walls.height)) # The closer to home, the larger food carried, more chance return food
        
        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = dist/(walls.width+walls.height)
        else:
            features["closest-food"] = 0

        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]
    


    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(enemiesAround) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
            features['enemyDistance'] = min(dists)
        if action == Directions.STOP: features['stop'] = 1
        features["distanceToHome"] = self.getMazeDistance(myPos,self.startPosition)

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]
    


    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features['teamDistance'] = team_dist

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]
    
    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def stateClosestFood(self, gameState:GameState):
        pos = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def getSuccessor(self, gameState: GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def getGhostLocs(self, gameState:GameState):
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
                for opponent in opAgents:
                        opPos = gameState.getAgentPosition(opponent)
                        opIsPacman = gameState.getAgentState(opponent).isPacman
                        if opPos and not opIsPacman: 
                                ghosts.append(opPos)
        return ghosts
    

