# gridmap
#
# Reads graph maps.
#
# For graph, state is pair of (x,y) tuple
# @author: mike
# @created: 2020-07-14
#

from lib_piglet.utils.tools import eprint

import os,sys

class vertex:

    def __init__(self, id:int, coordinate:tuple):
        self.id:int = id
        self.coordinate: tuple = coordinate
        self.adjacent: dict = {}

    def __str__(self):
        # return str(self.id) + ' adjacent: ' + str([(x.id,cost) for x,cost in self.get_connections()])
        return str(self.id) + ':' + str(self.coordinate)

    def __repr__(self):
        return str(self.id) +": " +str(self.coordinate)

    def print_connections(self):
        print(str(self.id) + ' adjacent: ' + str([(x.id,cost) for x,cost in self.get_connections()]))

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.items()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def get_location(self):
        return self.coordinate

    def set_location(self, coordinate:tuple):
        self.coordinate = coordinate

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __hash__(self):
        return self.get_id()

class graph:
    vert_dict: dict
    num_vertices: int
    domain_file_ :str


    def __init__(self,filename: str == None):
        self.vert_dict = {}
        self.num_vertices = 0
        if filename is not None:
            self.load(filename)
    
    def is_goal(self, current_state, goal_state):
        return current_state == goal_state

    def load(self, filename: str):
        if not os.path.exists(filename):
            eprint("err; file {} not exist".format(filename))
            exit(1)
        self.domain_file_=filename
        print("Loading graph file ... ...")

        f = open(filename)

        for line in f:
            content = line.strip().split()
            if len(content) == 0 or (content[0].strip() != "a" and content[0].strip() != "v"):
                continue

            if len(content) != 4:
                eprint("err; line {} should have 4 element".format(line))
                exit(1)

            if content[0].strip() == "v":
                try:
                    id = int(content[1])
                    x = int(content[2])
                    y = int(content[3])
                except:
                    eprint("err; can not convert elements of {} to integer ".format(line))
                    exit(1)
                if id in self.vert_dict:
                    v: vertex = self.get_vertex(id)
                    v.set_location((x, y))
                else:
                    self.add_vertex(id, (x, y))

            if content[0].strip() == "a":
                try:
                    n1 = int(content[1])
                    n2 = int(content[2])
                    cost = int(content[3])
                except:
                    eprint("err; can not convert elements of {} to integer ".format(line))
                    exit(1)

                self.add_edge(n1, n2, cost)
        sys.stdout.write("\033[F")


    def add_vertex(self, id:int, coordinates:tuple):
        self.num_vertices = self.num_vertices + 1
        new_vertex = vertex(id,coordinates)
        self.vert_dict[id] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost=0):
        if frm not in self.vert_dict:
            self.add_vertex(frm,())
        if to not in self.vert_dict:
            self.add_vertex(to,())

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def __iter__(self):
        return iter(self.vert_dict.values())







