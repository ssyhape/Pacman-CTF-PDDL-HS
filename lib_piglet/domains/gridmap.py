# gridmap
# 
# Reads and writes 2d grid maps.
#  -----> y
# |
# |
# V x
# For gridmap, state is pair of (x,y) tuple
# @author: dharabor
# @created: 2020-07-14
#

import sys, math

class grid_joint_state:
    
    def __init__(self, locations: list, is_goal =  False):
        self.agent_locations_: dict = {}
        for i in range(0,len(locations)):
            self.agent_locations_[i] = locations[i]
        self.is_goal_: bool = is_goal

    def __eq__(self, other):
        # When comparing with a goal state, as long as all locations for each agent in "a" are same as the location for
        # the agent in "b", we return True. Regardless the length difference between 2 agents. This make sure the codes
        # are compatible with well formed instances when checking if a state is goal.
        #
        # When comparing with a non goal state, we require two state have same length to return True.
        a = self.agent_locations_
        b = other.agent_locations_

        if self.is_goal_:
            a = other.agent_locations_
            b = self.agent_locations_

        if not self.is_goal_ and not other.is_goal_ and len(self.agent_locations_) != len(self.agent_locations_):
            return False

        for key,item in a.items():
            if key not in b:
                raise Exception("Agent {} not exist in both state.".format(key))

            if b[key] != item:
                return False
        return True
    
    def __hash__(self):
        return hash(str(self.agent_locations_))
    
    def __str__(self):
        return str(self.agent_locations_).replace(" ","")
    
    def __repr__(self):
        return str(self.agent_locations_).replace(" ","")


class gridmap:

    
    def __init__(self,filename: str):
        self.map_: list = []
        self.height_ : int= int(0)
        self.width_: int = int(0)
        self.map_size_: int = int(0)
        self.domain_file_: str = filename
        self.load(filename)
    
    def is_goal(self, current_state, goal_state):
        return current_state == goal_state


    # Load map in the map instance
    # @param filename The path to map file.
    def load(self, filename: str):
        self.domain_file_ = filename
        map_fo = open(filename, "r")

        if(self.__parse_header(map_fo) == -1):
            raise Exception("err; invalid map header")

        self.map_ = [ ([None] * int(self.width_)) for x in range(0,int(self.height_)) ]

        i = 0
        while(True):
            char = map_fo.read(1)
            if not char:
                break
            if(char == '\n'):
                continue

            y = int(i % int(self.width_))
            x = int(i / int(self.width_))
            if(char == '.'):
                self.map_[x][y] = True
            else:
                self.map_[x][y] = False
            i += 1

    # Write map tp a file
    def write(self):

        print("type octile")
        print("height " + str(self.height_))
        print("width " + str(self.width_))
        print("map")
 
        for x in range(0, int(self.height_)):
            for y in range(0, int(self.width_)):
                if(self.map_[x][y] == True):
                    print('.', end="")
                else:
                    print('@', end="")
            print()

    # tells whether the tile at location @param index is traversable or not
    # @return True/False
    def get_tile(self, loc: tuple):
        x = loc[0]
        y = loc[1]
        if(x < 0 or x >= self.height_ or y < 0 or y >= self.width_):
            return False
        return self.map_[x][y]

    def __parse_header(self, map_fo):

        tmp = map_fo.readline().strip().split(" ")
        if(tmp[0] != "type" and tmp[1] != "octile"):
            print("not octile map")
            return -1
    
        for i in range(0, 2):
            tmp = map_fo.readline().strip().split(" ")
            if tmp[0] == "height" and len(tmp) == 2:
                self.height_ = int(tmp[1])
            elif tmp[0] == "width" and len(tmp) == 2:
                self.width_ = int(tmp[1])
            else:
                return -1

        tmp = map_fo.readline().strip()
        if(tmp != "map"):
            return -1

    def __str__(self):
        return self.domain_file_

class gridmap_joint(gridmap):
    start_: grid_joint_state
    goal_: grid_joint_state

    def __init__(self, filename: str, start:grid_joint_state , goal:grid_joint_state):
        super(gridmap_joint,self).__init__(filename)
        self.start_ = start
        self.goal_ = goal
