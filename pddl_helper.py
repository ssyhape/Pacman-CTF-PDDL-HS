
from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action
from typing import List, Tuple
import os
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))


def getHighLevelPlan(solver, objects, initState, positiveGoalStates, negtiveGoalStates) -> List[Tuple[Action,pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        solver.parser_.reset_problem()
        solver.parser_.set_objects(objects)
        solver.parser_.set_state(initState)
        solver.parser_.set_negative_goals(negtiveGoalStates)
        solver.parser_.set_positive_goals(positiveGoalStates)
        
        # Solve the problem and return the plan
        return solver.solve()

# Specify the pddle model here
solver = pddl_solver(BASE_FOLDER+'/myTeam.pddl')

# Specify the pddl states, objects, goal states here.
positiveGoalStates = [("defend_foods",)]
negtiveGoalStates = []
objects=[("a1","current_agent"),("a2","ally"),("e1","enemy1"),("e2","enemy2")]
initStates = [("is_pacman", "a1"),("winning_gt5",)]

# Solve and print the plan.
plan = getHighLevelPlan(solver, objects, initStates, positiveGoalStates,negtiveGoalStates)
print(plan)
