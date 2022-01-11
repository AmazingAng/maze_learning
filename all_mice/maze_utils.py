class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

        
class Maze:
    """    
    This constructs a maze with a given graph, represented as a grid of cells.
    The maze consists of square cells the width of a corridor.
    Each cell in the maze has an (x,y) location, with x,y = 0,...,n
    y runs top to bottom.
    Each cell is also given a unique integer cell number.
    The present maze consists of straight runs, each run terminates in a branch point or 
    an end point. 
    
    graph: dict, keys-nodes: values-their neighbors
    maze_map: list of Cells
    decision_node: nodes with decision point
    runs: list of list of nodes
    """

    def __init__(self, nx, ny, graph, ix=0, iy=0, start_node = 1, end_node = 144):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.cell_num = nx * ny
        self.graph = graph
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]
        # Direction: neightbor - current
        self.to_direction = {(-1, 0): 'W',
                 (1, 0): 'E',
                 (0, 1):'S',
                 (0, -1): 'N'}
        self.is_bidirectional = self.check_bidirectional(graph) # check if the graph is bidirectional
        if self.is_bidirectional:
            self.make_maze() # knock down walls according to graph
            self.repeated_parents = set()
            self.decision_nodes = []
            self.runs = self.get_run_from_current(start_node, graph)
            self.shortest_path = self.BFS_SP(graph, start_node, end_node)
            self.run_levels, self.cell_levels = self.get_levels_for_runs()
            self.cell_dists = self.get_distance_to_goal(end_node = 144)
            # self.cell_levels = self.get_levels_for_cells()
            
    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""
        return self.maze_map[x][y]
    
    def idx_to_loc(self, idx):
        # convert index to location (x,y), note: location start from 0
        cell_w = idx % self.nx
        cell_h = idx // self.ny
        return cell_w, cell_h
    
    def loc_to_idx(self, cell_w, cell_h):
        # convert location (x,y) to index
        idx = cell_w * self.nx + cell_h +1
        return idx
    
    def check_bidirectional(self, graph):
        # check if the maze is bidirectional
        is_bidirectional = True
        for node in graph.keys():
            child_nodes = graph[node]
            for child in child_nodes:
                if not node in graph[child]:
                    print("Warning: node " + str(node) + " not in node " + str(child))
                    is_bidirectional = False
        if is_bidirectional:
            print("The maze is bidirectional, Ye!")
        else:
            print("Something is wrong, check log")
        return is_bidirectional

    def get_run_from_current(self, current_node, graph, runs = []):
        # get all runs from current node. Runs are defined as independent decision unit consisting a group of nodes
        # return runs
        done = False
        run = [current_node] # put current node in run
        self.repeated_parents.add(current_node) # add current node to repeated node (each node only belongs to one run)
        children_node = [value for value in graph[current_node] if not value in self.repeated_parents] # delete repeated children

        while not done:
            if len(children_node) >= 2: # child nodes >=2, decision point
                print("Node " + str(current_node) + " is Decision point")
                self.decision_nodes.append(current_node)
                runs.append(run) # 
                # b.set_trace()
                for child in children_node:
                    runs = self.get_run_from_current(child, graph, runs = runs)
                done = True
            elif len(children_node) == 0: # leaf node
                print("Node " + str(current_node) + " is Leaf node")
                runs.append(run)
                done = True
            else:
                current_node = children_node[0]
                # print(current_node)
                self.repeated_parents.add(current_node)
                run.append(current_node)
                # pdb.set_trace()
                children_node = [value for value in graph[current_node] if not value in self.repeated_parents] # parent will not be children's children
        return runs
    
    def get_distance_to_goal(self, end_node = 144):
        # calculate distance from any nodes to goal (default to 144)
        labeled_cells = [144]
        next_cells =[]
        current_dist = 0
        cell_dists = [0]*self.cell_num
        cell_dists[end_node - 1] = 0
        next_cells = self.graph[end_node]
        while(len(labeled_cells) < self.cell_num):
            current_dist += 1
            next_cells_tmp = []
            for i_cell in next_cells:
                if i_cell not in labeled_cells:
                    cell_dists[i_cell-1] = current_dist
                    next_cells_tmp = next_cells_tmp + self.graph[i_cell]
                    labeled_cells.append(i_cell)
            next_cells = next_cells_tmp
        return cell_dists
    
    def get_levels_for_runs(self):
        # calculate levels for all runs (and cells)
        # a level consists a group of cells separated by decision points
        decision_nodes_on_shortest = [x for x in self.decision_nodes if x in  self.shortest_path]
        level_seq = list(range(len(decision_nodes_on_shortest)))
        run_levels = [0]*len(self.runs)
        cell_levels = [0]*self.cell_num
        labeled_cells = []
        for i in range(len(self.runs)):
            # if runs are on shortest path, the last node should be decision nodes and determine its level
            if self.runs[i][-1] in decision_nodes_on_shortest:
                run_levels[i] = level_seq[decision_nodes_on_shortest.index(self.runs[i][-1])]
            # if runs are not on shortest path, the neighbor of the first node should be decision nodes and determine its level
            else:
                first_neighbors = self.graph[self.runs[i][0]]
                decision_node_last = [val for val in first_neighbors if val in labeled_cells][0]
                run_levels[i] = cell_levels[decision_node_last-1]
                
            for i_cell in self.runs[i]:
                cell_levels[i_cell-1] = run_levels[i]
                labeled_cells.append(i_cell)
        return run_levels, cell_levels
        
    # Function to find the shortest
    # path between two nodes of a graph
    def BFS_SP(self,graph, start, goal):
        explored = []

        # Queue for traversing the
        # graph in the BFS
        queue = [[start]]

        # If the desired node is
        # reached
        if start == goal:
            print("Same Node")
            return(None)

        # Loop to traverse the graph
        # with the help of the queue
        while queue:
            path = queue.pop(0)
            node = path[-1]

            # Condition to check if the
            # current node is not visited
            if node not in explored:
                neighbours = graph[node]

                # Loop to iterate over the
                # neighbours of the node
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)

                    # Condition to check if the
                    # neighbour node is the goal
                    if neighbour == goal:
                        print("Shortest path = ", *new_path)
                        return(new_path)
                explored.append(node)

        # Condition when the nodes
        # are not connected
        print("So sorry, but a connecting"\
                    "path doesn't exist :(")
        return(None)
    
    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)
    
    def maze_plot(self, axes = None):
        # plot maze with matplotlib.plot
        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 0.05
        # Height and width of the maze image (excluding padding), in pixels
        height = 6
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx
        # Write the SVG image file for maze
        # SVG preamble and styles.
        if axes == None:
            fig = plt.figure(figsize=[(height + 2 * padding), (width + 2 * padding)])
            axes = plt.gca()
            axes.invert_yaxis()
        
        # Draw the "South" and "East" walls of each cell, if present (these
        # are the "North" and "West" walls of a neighbouring cell in
        # general, of course).
        for x in range(self.nx):
            for y in range(self.ny):
                if self.cell_at(x, y).walls['S']:
                    x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                    axes.plot([x1, x2], [y1, y2], color = "black")
                if self.cell_at(x, y).walls['E']:
                    x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                    axes.plot([x1, x2], [y1, y2], color = "black")
        # Draw the North and West maze border, which won't have been drawn
        # by the procedure above.
        axes.plot([0, 0], [0, height], color = "black")
        axes.plot([0, width], [0, 0], color = "black")
        return axes
    
    def maze_plot_num(self, axes = None, mode='cells',numcol='blue'):
        '''
        adds numbering to an existing maze plot given by axes
        m: maze
        mode: 'cells','runs','nodes': depending on what gets numbered
        numcol: color of the numbers
        '''
        axes = self.maze_plot(axes)
        # plot maze with matplotlib.plot
        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 0.05
        # Height and width of the maze image (excluding padding), in pixels
        height = 6
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        if mode=='nodes':
            for j,r in enumerate(self.runs):
                x, y = self.idx_to_loc(r[-1]-1)
                plt.text((x+0.25)*scx, (y+0.65)*scy, '{:d}'.format(j+1),color=numcol) # number the ends of a run    
        
        if mode=='cells':
            for j in range(self.nx * self.ny):
                x, y = self.idx_to_loc(j)
                plt.text((x+0.25)*scx, (y+0.65)*scy, '{:d}'.format(j+1),color=numcol) # number the cells    
        
        if mode=='dists':
            for j in range(self.nx * self.ny):
                x, y = self.idx_to_loc(j)
                plt.text((x+0.25)*scx, (y+0.65)*scy, '{:d}'.format(self.cell_dists[j]),color=numcol) # number the cells    

        if mode=='decisions':
            for j,d in enumerate(self.decision_nodes):
                x, y = self.idx_to_loc(d-1)
                plt.text((x+0.25)*scx, (y+0.65)*scy, '{:d}'.format(j+1),color=numcol) # number the cells    

    
    def write_svg(self, filename):
        """Write an SVG image of the maze to filename."""

        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 10
        # Height and width of the maze image (excluding padding), in pixels
        height = 500
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG image file handle f."""

            print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'
                  .format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

        # Write the SVG image file for maze
        with open(filename, 'w') as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
                  .format(width + 2 * padding, height + 2 * padding,
                          -padding, -padding, width + 2 * padding, height + 2 * padding),
                  file=f)
            print('<defs>\n<style type="text/css"><![CDATA[', file=f)
            print('line {', file=f)
            print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
            print('    stroke-width: 5;\n}', file=f)
            print(']]></style>\n</defs>', file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls['S']:
                        x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
                    if self.cell_at(x, y).walls['E']:
                        x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)
            print('</svg>', file=f)

    def make_maze(self):
        ##  Knock down walls according to graph
        ## Note: location (x, y) start from 0, while index (idx) start from 1, BE CAREFUL!
        # Total number of cells.
        n_total = self.nx * self.ny
        for i in range(n_total):
            current_idx = i+1
            current_x, current_y = self.idx_to_loc(current_idx-1)
            current_cell = self.cell_at(current_x, current_y)
            neighbor_idxs = self.graph[current_idx]
            for neighbor_idx in neighbor_idxs:
                #pdb.set_trace()
                neighbor_x, neighbor_y = self.idx_to_loc(neighbor_idx-1)
                next_cell = self.cell_at(neighbor_x, neighbor_y)
                # print((current_idx, neighbor_idx) )

                # pdb.set_trace()
                diff_loc = (neighbor_x-current_x, neighbor_y-current_y)
                neighbor_dir = self.to_direction[diff_loc]
                current_cell.knock_down_wall(next_cell, neighbor_dir)
               
            
            
               
            
maze1_graph = {1: [2, 13],
         2: [1],
         3: [4, 15],
         4: [3, 5],
         5: [4, 6],
         6: [5, 7, 18],
         7: [6, 8],
         8: [7],
         9: [10, 21],
         10: [9, 11],
         11: [10, 12],
         12: [11, 24],
         13: [1, 14, 25],
         14: [13, 26],
         15: [3, 27],
         16: [17, 28],
         17: [16, 18, 29],
         18: [6, 17],
         19: [20, 31],
         20: [19, 21],
         21: [9, 20],
         22: [23, 34],
         23: [22, 24],
         24: [12, 23,36],
         25: [13],
         26: [14,27],
         27: [26, 15],
         28: [16],
         29: [17, 30],
         30: [29, 31, 42],
         31: [30, 19],
         32: [33, 44],
         33: [32, 34],
         34: [22, 33],
         35: [36],
         36: [35, 24],
         37: [38, 49],
         38: [37, 39],
         39: [40, 38, 51],
         40: [39],
         41: [42],
         42: [41, 30],
         43: [55],
         44: [32, 45],
         45: [44, 46],
         46: [45, 47],
         47: [46, 48],
         48: [47, 60],
         49: [37, 61],
         50: [51, 62],
         51: [39, 50,52],
         52: [51],
         53: [54],
         54: [53, 55,66],
         55: [43, 54,67],
         56: [57, 68],
         57: [56, 58],
         58: [57, 59],
         59: [58, 60],
         60: [48, 59],
         61: [49, 73],
         62: [50, 74],
         63: [64, 75],
         64: [63, 65],
         65: [64, 66],
         66: [54, 65],
         67: [55, 79],
         68: [56, 69],
         69: [68, 70],
         70: [69, 71],
         71: [70, 72],
         72: [71, 84],
         73: [61, 85],
         74: [62, 75],
         75: [74, 63],
         76: [77, 88],
         77: [76, 89],
         78: [79, 90],
         79: [78, 67],
         80: [81, 92],
         81: [80, 82],
         82: [81, 94],
         83: [95, 84],
         84: [72, 83, 96],
         85: [73, 97],
         86: [98],
         87: [88, 99],
         88: [87, 76],
         89: [77, 101],
         90: [78, 91],
         91: [90, 103],
         92: [80, 104],
         93: [105],
         94: [82, 95,106],
         95: [83, 94],
         96: [84],
         97: [85, 98, 109],
         98: [97, 86],
         99: [87, 100],
         100: [99, 112],
         101: [89, 102],
         102: [101, 114],
         103: [91, 104],
         104: [103, 92],
         105: [93, 106],
         106: [105, 94],
         107: [108, 119],
         108: [107, 120],
         109: [97, 110,121],
         110: [109, 122],
         111: [112, 123],
         112: [100, 111],
         113: [114, 125],
         114: [113, 102],
         115: [116, 127],
         116: [115, 117],
         117: [116, 129],
         118: [119, 130],
         119: [118, 107],
         120: [108],
         121: [109, 133],
         122: [110, 123],
         123: [111, 122],
         124: [125, 136],
         125: [124, 113],
         126: [127, 138],
         127: [115, 126],
         128: [129, 140],
         129: [128, 117,141],
         130: [118, 131,142],
         131: [130, 132],
         132: [131, 144],
         133: [121, 134],
         134: [133, 135],
         135: [134],
         136: [124, 137],
         137: [136, 138],
         138: [137, 126],
         139: [140],
         140: [139, 128],
         141: [129, 142],
         142: [130, 141,143],
         143: [142],
         144: [132],
        }

maze2_graph = {1: [2],
         2: [1,3,14],
         3: [2,15],
         4: [16],
         5: [6, 17],
         6: [5, 7],
         7: [6, 8,19],
         8: [7,20],
         9: [10,21],
         10: [9, 11],
         11: [10, 12],
         12: [11, 24],
         13: [14, 25],
         14: [2,13],
         15: [3, 16],
         16: [4, 15],
         17: [5, 29],
         18: [19, 30],
         19: [7, 18],
         20: [8],
         21: [9, 33],
         22: [23, 34],
         23: [22, 24],
         24: [12, 23,36],
         25: [13,37],
         26: [27],
         27: [26, 39],
         28: [29,40],
         29: [17, 28],
         30: [18, 31],
         31: [30, 43],
         32: [33, 44],
         33: [21, 32],
         34: [22, 46],
         35: [36,47],
         36: [35, 24],
         37: [25, 49],
         38: [50],
         39: [27, 51],
         40: [28,52],
         41: [42,53],
         42: [41],
         43: [31,44],
         44: [32,43, 56],
         45: [46, 57],
         46: [34, 45],
         47: [35],
         48: [ 60],
         49: [37, 61],
         50: [38, 51,62],
         51: [39, 50,52],
         52: [51,40,53],
         53: [41,52,65],
         54: [55,66],
         55: [54,67],
         56: [44],
         57: [45, 69],
         58: [70],
         59: [71, 60],
         60: [48, 59,72],
         61: [49, 73],
         62: [50],
         63: [64, 75],
         64: [63, 65],
         65: [64, 53],
         66: [54, 78],
         67: [55, 68],
         68: [67, 69],
         69: [68, 57],
         70: [58,82, 71],
         71: [70, 59],
         72: [60, 84],
         73: [61, 74],
         74: [73, 86],
         75: [76, 63],
         76: [75, 88],
         77: [78, 89],
         78: [77,66, 90],
         79: [80, 91],
         80: [79, 92],
         81: [82],
         82: [70,81, 94],
         83: [95, 84],
         84: [72, 83],
         85: [86, 97],
         86: [74,85,87],
         87: [86],
         88: [100, 76],
         89: [77, 101],
         90: [78],
         91: [79, 103],
         92: [80, 93],
         93: [92,105],
         94: [82,106],
         95: [83, 96,107],
         96: [95,108],
         97: [85, 98],
         98: [97, 110],
         99: [111, 100],
         100: [99, 88],
         101: [89, 102],
         102: [101, 114],
         103: [91, 104],
         104: [103, 116],
         105: [93, 106,117],
         106: [105, 94],
         107: [95],
         108: [96, 120],
         109: [110,121],
         110: [98,109, 122],
         111: [99, 123],
         112: [113, 124],
         113: [112],
         114: [115, 102],
         115: [116, 114],
         116: [115, 104],
         117: [105, 129],
         118: [119],
         119: [118, 120,131],
         120: [108,119],
         121: [109, 133],
         122: [110],
         123: [111, 135],
         124: [125, 112],
         125: [124, 126],
         126: [125, 138],
         127: [128, 139],
         128: [127, 140],
         129: [117,141],
         130: [131,142],
         131: [130, 119],
         132: [144],
         133: [121, 134],
         134: [133, 135],
         135: [123,134,136],
         136: [135, 137],
         137: [136, 138],
         138: [137, 126],
         139: [127],
         140: [141, 128],
         141: [129, 140],
         142: [130,143],
         143: [142,144],
         144: [132,143],
        }

# Function to find the shortest
# path between two nodes of a graph
def BFS_SP(graph, start, goal, verbose = False):
    explored = []
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is
    # reached
    if start == goal:
        if verbose:
            print("Same Node")
        return(None)
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    if verbose:
                        print("Shortest path = ", *new_path)
                    return(new_path)
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    print("So sorry, but a connecting"\
                "path doesn't exist :(")
    return(None)