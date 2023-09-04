from lib_piglet.utils.pddl_parser import *
from typing import List

class pddl_state:

    def __init__(self,state:frozenset,from_action:Action):
        self.state_set_ :frozenset= state
        self.from_action_:Action = from_action

    def __hash__(self):
        return hash(self.state_set_)
    
    def __repr__(self):
        if self.from_action_ is None:
            return str("START")
        return "{}: {}".format(self.from_action_.name,' '.join(self.from_action_.parameters))
    
    def __str__(self):
        if self.from_action_ is None:
            return str("START")
        return str([list(i) for i in self.state_set_])
    
    def __eq__(self, other):
        return self.state_set_ == other.state_set_

class pddl_goal:

    def __init__(self,pos: frozenset,neg: frozenset):
        self.goal_neg_: frozenset= neg
        self.goal_pos_: frozenset = pos
    
    def __str__(self):
        return "GOAL"
        # return "+ {}, - {}".format(str([list(i) for i in self.goal_pos_]), str([list(i) for i in self.goal_neg_]))





class pddl:

    def __init__(self,domain_path:str, problem_path:str = None):
        self.parser_: PDDL_Parser = PDDL_Parser()
        self.parser_.parse_domain(domain_path)
        self.ground_actions_: List[Action] = []
        self.domain_file_ = None
        self.domain_path_ = domain_path
        self.problem_path_ = problem_path
        self.h_dict = {}
        
        self.goal_state_: pddl_goal = None
        self.start_state_: pddl_state = None

        if problem_path is not None:
            self.parser_.parse_problem(problem_path)
            self.set_start_goal()

    def read_problem(self, problem_path):
        self.parser_.reset_problem()
        self.parser_.parse_problem(problem_path)
        self.set_start_goal()
    
    def set_start_goal(self):
        self.ground_actions_ = []
        for action in self.parser_.actions:
            for act in action.groundify(self.parser_.objects, self.parser_.types):
                self.ground_actions_.append(act)
        self.goal_state_: pddl_goal = pddl_goal(self.parser_.positive_goals, self.parser_.negative_goals)
        self.start_state_: pddl_state = pddl_state( self.parser_.state, None)
    
    
    
    def is_goal(self,current_state:pddl_state, goal_state: pddl_goal):
        return self.applicable(current_state.state_set_, goal_state.goal_pos_, goal_state.goal_neg_)

    
    #-----------------------------------------------
    # Applicable
    #-----------------------------------------------
    
    def applicable(self, state:frozenset, positive:frozenset, negative:frozenset):
        return positive.issubset(state) and negative.isdisjoint(state)

    #-----------------------------------------------
    # Apply
    #-----------------------------------------------

    def apply(self, state:frozenset, positive:frozenset, negative:frozenset):
        return state.difference(negative).union(positive)







