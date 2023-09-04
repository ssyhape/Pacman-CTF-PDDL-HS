# constraints/grid_constraints.py
# This module defines constraints, constraints table and reservation table for grid map.
#
# @author: mike
# @created: 2020-07-15
#

from lib_piglet.domains.grid_action import Move_Actions


# The constraint data type for 4 connected grid map.
# v_: indicates if the tile is blocked or not
# the default state is unblocked
#
# e_: indicates which of the outgoing directions are constrained / blocked.
# the default state are all False
# 
# timestep_: indicates the timestep when the edge costs apply. 
# the default timestep is 0.
# 
class grid4_constraint:
    v_: bool
    e_: list
    timestep_: int

    # initialize a time_constraint, forbid an agent visit this location
    def __init__(self):
        self.v_ = False
        self.e_ = [False] * len(Move_Actions)
        self.timestep_ = 0


# The constraint table describes the constraints each location holds
class grid_constraint_table:
    width_: int
    height_: int
    table_: list
    constraint_type_ = None

    # initialize an empty constraint table
    # @param width The width of the map
    # @param height The height of the map
    def __init__(self,width: int, height: int, constraint_type):
        self.width_ = width
        self.height_ = height
        self.table_ = [[None] * int(self.width_) for x in range(int(self.height_))]
        self.constraint_type_ = constraint_type

    # add an constraint
    # @param loc A tuple of (x,y) coordinates
    # @param time The timestep the constraint is valid
    # @param constraint A constraint data structure
    def add_constraint(self,loc: tuple, time: int, constraint):
        x = loc[0]
        y = loc[1]
        if self.table_[x][y] is None:
            self.table_[x][y] = {}
        self.table_[x][y][time] = constraint

    # get the constraint hold on a location on a timestep
    # @param loc A tuple of (x,y) coordinates
    # @param time The timestep on this location
    # @return constraint Return the constraint on required timestep and location. Create and return an dummy constraint if no constraint.
    def get_constraint(self,loc: tuple, time: int):
        x = loc[0]
        y = loc[1]
        if self.table_[x][y] is None:
            self.table_[x][y] = {}
        if time not in self.table_[x][y]:
            self.table_[x][y][time] = self.constraint_type_()
            self.table_[x][y][time].timestep_ = time
        return self.table_[x][y][time]

    # clear the constraint table
    def clear(self):
        self.table_ = [[{}] * int(self.width_) for x in range(int(self.height_))]


# An reservation table records does any agent reserved a location at a timestep.
class grid_reservation_table:

    width_: int
    height_: int
    table_: list

    def __init__(self,width, height):
        self.width_ = width
        self.height_ = height
        self.table_ = [[None] * int(self.width_) for x in range(int(self.height_))]

    # Check is an location reserved by any other agent
    # @param loc A tuple of (x,y) coordinates.
    # @patam time The timestep.
    # @param agent_id An int of agent_id.
    # @return bool True if reserved.
    def is_reserved(self, loc: tuple, time: int, current_agent_id: int = -1):
        x = loc[0]
        y = loc[1]
        if self.table_[x][y] is None:
            return False
        if time not in self.table_[x][y]:
            return False

        if self.table_[x][y][time]==-1:
            return False

        if self.table_[x][y][time] == current_agent_id and current_agent_id != -1:
            return False

        return True

    # Add an single reservation to reservation table
    # @param loc A tuple of (x,y) coordinates.
    # @patam time The timestep.
    # @param agent_id An int of agent_id.
    # @return success True if add successful, false if the location reserved by other agent.
    def add_loc(self, loc: tuple, time: int, agent_id: int):
        x = loc[0]
        y = loc[1]
        if self.table_[x][y] is None:
            self.table_[x][y] = {}

        if time not in self.table_[x][y]:
            self.table_[x][y][time] = agent_id
            return True

        if self.table_[x][y][time]!= agent_id and self.table_[x][y][time]!=-1:
            return False
        else:
            self.table_[x][y][time] = agent_id
            return True


    # Delete a reserve from reservation table
    # @param loc A tuple of (x,y) coordinates.
    # @patam time The timestep.
    # @param agent_id An int of agent_id.
    # @return success True if delete successful, False if reserve doesn't exist
    def del_loc(self, loc: tuple, time: int, agent_id: int):
        x = loc[0]
        y = loc[1]
        if self.table_[x][y] is None:
            return False

        if time in self.table_[x][y]:
            if self.table_[x][y][time]== agent_id:
                self.table_[x][y][time] = -1
                return True
            else:
                return False
        else:
            return False

    # clear the reservation table
    def clear(self):
        self.table_ = [[None] * int(self.width_) for x in range(int(self.height_))]


