import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("*************   A-STAR Pathfinding Algorithm   *************")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Node():
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width # locates the current cell coordinate in x-direction
        self.y = col * width # locates the current cell coordinate in y-direction
        self.color = WHITE # initial color = white, i.e. unused Node
        self.neighbors = [] # empty list for all neighbors of a Node
        self.width = width # seems to be for display
        self.total_rows = total_rows

    # return row-col-position of Node:
    def get_pos(self):
        return self.row, self.col
    def is_closed(self):
        return self.color == RED
    def is_open(self):
        return self.color == GREEN
    def is_barrier(self):
        return self.color == BLACK
    def is_start(self):
        return self.color == ORANGE
    def is_end(self):
        return self.color == TURQUOISE
    def reset(self):
        self.color = WHITE
    def make_start(self):
        self.color = ORANGE
    def make_closed(self):
        self.color = RED
    def make_open(self):
        self.color = GREEN
    def make_barrier(self):
        self.color = BLACK
    def make_end(self):
        self.color = TURQUOISE
    def make_path(self):
        self.color = PURPLE
    def __lt__(self, other):
        return False

    # Use method applied to WINDOW in order to draw Node:
    def draw(self, win):
        # Draw a rectangle of certain color at coordinates x and y with the defined widths and heights:
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])
            
    

# Compute H cost value - needs to be always LESS than the actual cost which would be need to go from the current Node to
# the end Node:
def ComputeHCost(CurrentNode, EndNode):
    # F = G + H
    # F = complete estimated cost for given point
    # G : exact cost from start Node to this Node
    # h : estimated cost from this Node to end Node - requires some estimation heuristic
    x1, y1 = CurrentNode
    x2, y2 = EndNode
    return abs(x1-x2) + abs(y1-y2) # is less costly than the euclidean distance

def reconstruct_path(DictNodesCameFrom, CurrentNode, draw):
    while CurrentNode in DictNodesCameFrom:
        CurrentNode = DictNodesCameFrom[CurrentNode]
        CurrentNode.make_path()
        draw()

def AStartAlgorithm(draw, grid, StartNode, EndNode):
    NodeCount = 0
    OpenSet = PriorityQueue() # Why do we use a priority queue for the open set?
    OpenSet.put((0, NodeCount, StartNode)) # Insert Tuple
    came_from = {}

    g_score = {Node: float("inf") for row in grid for Node in row}
    g_score[StartNode] = 0

    f_score = {Node: float("inf") for row in grid for Node in row}
    f_score[StartNode] = ComputeHCost(StartNode.get_pos(), EndNode.get_pos())

    open_set_hash = {StartNode}

    while not OpenSet.empty(): # why do we use this? The open set should never be empty. Even at the beginning it should contain at least the start Node, right?
        # Check if quitting the game was requested, if so quit the game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Take 1 Node from open set:
        CurrentNode = OpenSet.get()[2] # why get 2??? what does this command do at all???
        # Remove the Node from the
        open_set_hash.remove(CurrentNode) # Remove current Node from open list.

        # End program algorithm if current Node is the end Node:
        if CurrentNode == EndNode:
            reconstruct_path(came_from, CurrentNode, draw) # why is this allowed to do with draw? it's a function which is handed over, but needs "win" as input, which we do not give.
            EndNode.make_end()
            return True # the function will just end and return true, this is because we finish the game.

        # Loop through all neighbors:
        for neighbor in CurrentNode.neighbors:
            # in case necessary, we want to update the neighbors with a new g-score:
            temp_g_score = g_score[CurrentNode] + 1

            # All updates only apply to Nodes which have a LOWER g-score than what we suggest as update
            # Otherwise we know that a) either the new g-score is not the one of the optimal path
            # b) the neighbor in any case has been already processed
            if temp_g_score < g_score[neighbor]:
                # The neighbors need to "remember" from which Node the way led to them:
                came_from[neighbor] = CurrentNode
                # Updating g-score:
                g_score[neighbor] = temp_g_score
                # Updating f-score:
                f_score[neighbor] = temp_g_score + ComputeHCost(neighbor.get_pos(), EndNode.get_pos())
                #Add neighbor to open set:
                if neighbor not in open_set_hash:
                    NodeCount += 1
                    OpenSet.put((f_score[neighbor], NodeCount, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        # Why make closed? that actually only changes the color, right?
        if CurrentNode != StartNode:
            CurrentNode.make_closed()


def make_grid(rows, width):

    gap = width // rows
    grid = [[Node(i, j, gap, rows) for j in range(rows)] for i in range(rows)]

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for Node in row:
            Node.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


# Function for taking action on a RIGHT mouse click - goal: reset Node.
def click_right_mouse(StartNode, EndNode, ROWS, width, grid):

    # Function for evaluation position in window:
    pos = pygame.mouse.get_pos()
    # Position in window and row/col count gives info in which cell the clicking event happened:
    row, col = get_clicked_pos(pos, ROWS, width)
    # Access current Node from grid:
    CurrentNode = grid[row][col]
    # Right mouse button means we want to reset:
    CurrentNode.reset()
    if CurrentNode == StartNode:
        StartNode = None
    elif CurrentNode == EndNode:
        EndNode = None

    return StartNode, EndNode

# Function for taking action on a LEFT mouse click - goal: set start or end Node or make_barrier
def click_left_mouse(StartNode, EndNode, ROWS, width, grid):

    # Function for evaluation position in window:
    pos = pygame.mouse.get_pos()
    # Position in window and row/col count gives info in which cell the clicking event happened:
    row, col = get_clicked_pos(pos, ROWS, width)
    # Access current Node from grid:
    CurrentNode = grid[row][col]

    # Algorithm for StartNode and EndNode selection and drawing walls:
    if not StartNode and CurrentNode != EndNode:  # Start
        StartNode = CurrentNode
        StartNode.make_start()
    elif not EndNode and CurrentNode != StartNode:  # End
        EndNode = CurrentNode
        EndNode.make_end()
    elif CurrentNode != EndNode and CurrentNode != StartNode:  # Draw Walls
        CurrentNode.make_barrier()

    return StartNode, EndNode


# Function is called in case of a keyboard interaction:
def press_key_on_keyboard(event, StartNode, EndNode, grid, win, ROWS, width):

    # Space button starts game in case start and end Node have been set:
    if event.key == pygame.K_SPACE and StartNode and EndNode:
        # Update iteratively all Nodes in all rows regarding their neighbors:
        for row in grid:
            for CurrentNode in row:
                CurrentNode.update_neighbors(grid)

        # Call algorithm for evaluating grid:
        AStartAlgorithm(lambda: draw(win, grid, ROWS, width), grid, StartNode, EndNode)

    # Resetting game:
    if event.key == pygame.K_c:
        StartNode = None
        EndNode = None
        grid = make_grid(ROWS, width)

    return grid, StartNode, EndNode


def main(win, width):
    # Number of rows and columns of the grid:
    ROWS = 50
    # Generate the cell grid: (the grid is a list containing a number of rows * rows Nodes.
    grid = make_grid(ROWS, width)

    # Initialize start and end point:
    StartNode = None
    EndNode = None

    run = True
    while run:

        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            # In case of closing the window:
            if event.type == pygame.QUIT:
                run = False

            # Pressing LEFT mouse button:
            if pygame.mouse.get_pressed()[0]:
                StartNode, EndNode = click_left_mouse(StartNode, EndNode, ROWS, width, grid)

            # Pressing RIGHT mouse button:
            elif pygame.mouse.get_pressed()[2]:
                StartNode, EndNode = click_right_mouse(StartNode, EndNode, ROWS, width, grid)

            # Check pressed keyboard buttons for space button:
            # This means NO CLICKING is evaluated anymore until the algorithm found the solution:
            # AFTER that, the game can be quit or resetted again:
            elif event.type == pygame.KEYDOWN:
                grid, StartNode, EndNode = press_key_on_keyboard(event, StartNode, EndNode, grid, win, ROWS, width)

    pygame.quit()

main(WIN, WIDTH)