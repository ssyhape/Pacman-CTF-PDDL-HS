# cli/cli_tools.py
# This module provides tools for cli interface
#
# @author: mike
# @created: 2020-07-19

import sys, argparse, os
from enum import IntEnum
from lib_piglet.utils.tools import eprint


# Describe parameters in arg parser result. For IDE convenient.
class args_interface:
    problem: str
    framework: str
    strategy: str
    time_limit: int
    output_file: str
    scenario: str
    depth_limit:int
    cost_limit:int
    heuristic_weight:float
    multi_agent: bool
    problem_number: int
    anytime: bool
    id_threshold_type:int



# Describe domain type enum
class DOMAIN_TYPE(IntEnum):
    gridmap = 0
    n_puzzle = 1
    graph = 2
    pddl = 3


# Describe task class.
class task:
    domain: str = None
    domain_type: int = None
    start_state = None
    goal_state = None
    problem: str = None


framework_choice = ["tree",
                    "graph",
                    "iterative"
                    ]

strategy_choice = ["breadth",
                   "depth",
                   "uniform",
                   "a-star",
                   "greedy-best"
                    ]
domain_types = ["grid4",
                 "n-puzzle",
                "graph",
                "pddl"
                 ]
id_choices=["depth","cost"]

statistic_template = "{0:10}| {1:10}| {2:10}| {3:10}| {4:10}| {5:10}| {6:10}| {7:10}| {8:10}| {9:10}| {10:20}| {11:20}"
csv_template = '"{0}","{1}","{2}","{3}","{4}","{5}","{6}","{7}","{8}","{9}","{10}","{11}"\n'

anytime_statistic_template = "{0:10}| {1:10}| {2:10}| {3:10}| {4:10}| {5:10}| {6:10}| {7:10}| {8:10}| {9:10}| {10:10}| {11:10}| {12:20}| {13:20}"
anytime_csv_template = '"{0}","{1}","{2}","{3}","{4}","{5}","{6}","{7}","{8}","{9}","{10}","{11}","{12}","{13}"\n'


statistic_header = [
    "Framework",
    "Strategy",
    "Status",
    "Cost",
    "Depth",
    "Nodes(exp)",
    "Nodes(gen)",
    "Runtime",
    "start",
    "goal",
    "Problem",
    "Solution"
]

anytime_statistic_header = [
    "Framework",
    "Strategy",
    "Status",
    "Cost",
    "Depth",
    "Nodes(exp)",
    "Nodes(gen)",
    "Re-Exp",
    "1st Sol T",
    "Runtime",
    "start",
    "goal",
    "Problem",
    "Solution"
]


# Print statistic header to screen
def print_header(anytime):
    if anytime:
        print(anytime_statistic_template.format(*anytime_statistic_header))
    else:
        print(statistic_template.format(*statistic_header))


# @return str Statistic header in csv format
def csv_header(anytime):
    if anytime:
        return anytime_csv_template.format(*anytime_statistic_header)
    return csv_template.format(*statistic_header)


# statistic to string
# @return str A string of statistic information
def statistic_string(args,search, anytime):
    template = statistic_template
    if anytime:
        template = anytime_statistic_template
    if args.solution:
        return template.format(str(args.framework), args.strategy,
                                         *[str(x) for x in search.get_statistic()],
                                         str(search.solution_))
    return template.format(str(args.framework), args.strategy,
                                     *[str(x) for x in search.get_statistic()],"Hidden")

# statistic to csv
# @return str A csv format string of statistic information
def statistic_csv(args,search, anytime):
    template = csv_template
    if anytime:
        template = anytime_csv_template
    if args.solution:
        return template.format(str(args.framework), args.strategy,
                                         *[str(x) for x in search.get_statistic()],
                                         search.solution_)
    return template.format(str(args.framework), args.strategy,
                                     *[str(x) for x in search.get_statistic()], "Hidden")

# Parse arguments from cli interface
# @return argument object
def parse_args():
    parser = argparse.ArgumentParser(description="""
     This is piglet commandline interface. You can use piglet-cli run a variety search algorithms. 
     A problem scenario file must be provided with -p, unless you problems are passed in through stdin.
     The framework is graph search on default. You can switch to tree search by -f tree.
     The strategy is uniform-cost search by default. You can switch to breadth first, depth first or A-star by -s.
     """)

    parser.add_argument("-p","--problem", type=str, default=None,
                        help='Specify the problem scenario file. A problem scenario file  ', metavar="/Path/to/scenario_file")

    parser.add_argument("-f", '--framework', type=str, default="graph",
                        choices=framework_choice,
                        help='Specify the search framework you want to use. \
                         Supported frameworks are: [{}].'.format(", ".join(framework_choice)),
                        metavar="graph")

    parser.add_argument("-s", '--strategy', type=str, default=["uniform"],
                        choices=strategy_choice,
                        help='Specify the search strategy you want to use.\
                          Supported strategies are: [{}]. If using strategy "depth" and framework "iterative",\
                           a maximum depth/cost limit also need to be specified after "depth".'.format(", ".join(strategy_choice)), metavar="uniform")
    
    parser.add_argument("-i", '--id-threshold-type', type=str, default=["depth"],
                        choices=id_choices,
                        help='Specify the ID Threshold Type you want to use, when strategy is "depth" and framework is "iterative".\
                          Supported strategies are: [{}]. .'.format(", ".join(strategy_choice)), metavar="depth")
    
    parser.add_argument("-t", '--time-limit', type=float, default=30,
                        help='Specify the time-limit for the search. (seconds) The default setting is 30 second.', metavar=30)

    parser.add_argument('--depth-limit', type=int, default=sys.maxsize,
                        help='Specify the depth-limit for tree search.', metavar=1000)
    parser.add_argument('--cost-limit', type=int, default=sys.maxsize,
                        help='Specify the cost-limit for tree search.', metavar=1000)
    parser.add_argument('--heuristic-weight', type=float, default=1,
                        help='Set a heuristic weight for suboptimal a-star.', metavar=1.0)

    parser.add_argument("-o", "--output-file", type=str, default=None,
                        help="Output results to a file")

    parser.add_argument("--solution", default=False,action="store_true",
                        help="Print/write solution")

    parser.add_argument("-m","--multi-agent", default=False, action="store_true",
                        help="Search in incremental multi-agent mode for grid map")

    parser.add_argument("-a","--anytime", default=False, action="store_true",
                        help="Search in Anytime Weighted A* mode when having graph as framework and a-star as stragety")

    parser.add_argument('-n',"--problem-number", type=int, default=sys.maxsize,
                        help='Solve only top n problem from the scenario file', metavar=1000)


    args , unknown = parser.parse_known_args()
    args:args_interface = args
    if args.framework == "iterative":
        if args.strategy != "depth":
            print("err; With iterative-deepening search, the strategy can only be depth ", file = sys.stderr)
            exit(1)
    if args.anytime and args.strategy != "a-star":
        print("err; anytime search only works with graph framework and a-start stragety", file = sys.stderr)
        exit(1)

    if (args.depth_limit != sys.maxsize or args.cost_limit!=sys.maxsize) and args.framework != "tree":
        eprint("warning; depth limit or cost limit only works with tree search")

    if args.heuristic_weight != 1.0 and args.strategy != "a-star":
        eprint("warning; heuristic weight only works with a-star strategy for suboptimal a-star")

    return args


# Parse individual problem to task
# @param problem. A list of scenario entry
# @return task A task object
def parse_problem(problem: list,domain_type:int):
    ta = task()
    ta.domain_type = domain_type
    if domain_type == DOMAIN_TYPE.n_puzzle:
        try:
            ta.domain = int(problem[0])
        except:
            print("err; Cannot convert {} to puzzle width".format(problem[0]),file=sys.stderr)
            exit(1)
        ta.start_state = problem[1].split(",")
    elif domain_type == DOMAIN_TYPE.gridmap:
        ta.domain = problem[1]
        if len(problem) < 9:
            print("err; the length of an entry of grid problem should be 9. Check the sample grid scenario format",file=sys.stderr)
            exit(1)
        try:
            ta.start_state = (int(problem[5]), int(problem[4]))
            ta.goal_state = (int(problem[7]), int(problem[6]))
        except:
            print("err; Cannot convert {} {} {} {} to coordinates".format(*problem[4:8]),file=sys.stderr)
            exit(1)
    elif domain_type == DOMAIN_TYPE.graph:
        ta.domain = problem[0]
        if len(problem) < 4:
            print("err; the length of an entry of graph problem should be 4. Check the sample graph scenario format",file=sys.stderr)
            exit(1)
        try:
            ta.start_state = int(problem[1])
            ta.goal_state = int(problem[2])
        except:
            print("err; Cannot convert {} {} to coordinates".format(*problem[1:3]),file=sys.stderr)
            exit(1)
    elif domain_type == DOMAIN_TYPE.pddl:
        ta.domain = problem[0]
        ta.problem = problem[1]
        
        if len(problem) != 2:
            print("err; the length of an entry of pddl problem should be 2. Check the sample pddl scenario format",file=sys.stderr)
            exit(1)
    else:
        print("err; Unknown domain type", file=sys.stderr)
        exit(1)
    return ta

def parse_scen_header(content):
    if len(content) != 2 or content[0] != "domain" or content[1] not in domain_types:
        print("err; The first line of input source must be domain type. eg. domain octile", file=sys.stderr)
        print("Supported domains are: [{}]".format(",".join(domain_types)), file=sys.stderr)
        exit(1)
    if content[1] == "n-puzzle":
        domain_type = DOMAIN_TYPE.n_puzzle
    elif content[1] == "grid4":
        domain_type = DOMAIN_TYPE.gridmap
    elif content[1] == "graph":
        domain_type = DOMAIN_TYPE.graph
    elif content[1] == "pddl":
        domain_type = DOMAIN_TYPE.pddl
    else:
        print("err; Unknown domain type: {}".format(content[1]), file=sys.stderr)
        exit(1)
    return domain_type







