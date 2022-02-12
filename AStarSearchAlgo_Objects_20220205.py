import pygame
import math
from queue import PriorityQueue

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Node():
    def __init__(self, row, col, width, total_row_count):
        self.row = row
        self.col = col
        self.x = row * width # locates the current cell coordinate in x-direction
        self.y = col * width # locates the current cell coordinate in y-direction
        self.color = WHITE # initial color = WHITE, i.e. unused Node
        self.neighbors = [] # empty list for all neighbors of a Node
        self.width = width # seems to be for display
        self.total_row_count = total_row_count

    # return row-col-clicked_positionition of Node:
    def _get_clicked_position(self):
        return self.row, self.col
    def _is_barrier(self):
        return self.color == BLACK
    def _reset(self):
        self.color = WHITE
    def _make_start(self):
        self.color = ORANGE
    def _make_closed(self):
        self.color = RED
    def _make_open(self):
        self.color = GREEN
    def _make_barrier(self):
        self.color = BLACK
    def _make_end(self):
        self.color = TURQUOISE
    def _make_path(self):
        self.color = PURPLE
    def __lt__(self, other):
        return False

    # Use method applied to WINDOW in order to draw Node:
    def _draw(self, win):
        # Draw a rectangle of certain color at coordinates x and y with the defined widths and heights:
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def _update_neighbors(self, GRID):
        self.neighbors = []
        if self.row < self.total_row_count - 1 and not GRID[self.row + 1][self.col]._is_barrier():  # DOWN
            self.neighbors.append(GRID[self.row + 1][self.col])

        if self.row > 0 and not GRID[self.row - 1][self.col]._is_barrier():  # UP
            self.neighbors.append(GRID[self.row - 1][self.col])

        if self.col < self.total_row_count - 1 and not GRID[self.row][self.col + 1]._is_barrier():  # RIGHT
            self.neighbors.append(GRID[self.row][self.col + 1])

        if self.col > 0 and not GRID[self.row][self.col - 1]._is_barrier():  # LEFT
            self.neighbors.append(GRID[self.row][self.col - 1])

ROW_COUNT = 50
WIDTH_WINDOW = 800
GRID = []
width_cell = WIDTH_WINDOW // ROW_COUNT
GRID = [[Node(i, j, width_cell, ROW_COUNT) for j in range(ROW_COUNT)] for i in range(ROW_COUNT)]
del width_cell

class Game():
    # Number of ROW_COUNT and columns of the GRID:

    pygame_window = pygame.display.set_mode((WIDTH_WINDOW, WIDTH_WINDOW))

    def __init__(self):
        # Generate the cell GRID: (the GRID is a list containing a number of ROW_COUNT * ROW_COUNT Nodes.
        # Initialize start and end point:
        self.start_node = None
        self.end_node = None

    def _run_astar_algorithm(self, draw, GRID):

        node_count = 0
        open_set = PriorityQueue()  # Why do we use a priority queue for the open set?
        open_set.put((0, node_count, self.start_node))  # Insert Tuple
        came_from = {}

        gcost = {Node: float("inf") for row in GRID for Node in row}
        gcost[self.start_node] = 0

        fcost = {Node: float("inf") for row in GRID for Node in row}
        fcost[self.start_node] = self._compute_hcost(self.start_node._get_clicked_position(), self.end_node._get_clicked_position())

        # hashing of nodes:
        open_set_hash = {self.start_node}

        while not open_set.empty():  # why do we use this? The open set should never be empty. Even at the beginning it should contain at least the start Node, right?
            # Check if quitting the game was requested, if so quit the game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # Take 1 Node from open set:
            current_node = open_set.get()[2]  # why get 2??? what does this command do at all???
            # Remove the Node from the hash dict:
            open_set_hash.remove(current_node)  # Remove current Node from open list.

            # End program algorithm if current Node is the end Node:
            if current_node == self.end_node:
                self._reconstruct_path(came_from, current_node,
                                 draw)  # why is this allowed to do with draw? it's a function which is handed over, but needs "win" as input, which we do not give.
                self.end_node._make_end()
                return True  # the function will just end and return true, this is because we finish the game.

            # Loop through all neighbors:
            for neighbor in current_node.neighbors:
                # in case necessary, we want to update the neighbors with a new g-score:
                temp_gscore = gcost[current_node] + 1

                # All updates only apply to Nodes which have a LOWER g-score than what we suggest as update
                # Otherwise we know that a) either the new g-score is not the one of the optimal path
                # b) the neighbor in any case has been already processed
                if temp_gscore < gcost[neighbor]:
                    # The neighbors need to "remember" from which Node the way led to them:
                    came_from[neighbor] = current_node
                    # Updating g-score:
                    gcost[neighbor] = temp_gscore
                    # Updating f-score:
                    fcost[neighbor] = temp_gscore + self._compute_hcost(neighbor._get_clicked_position(),
                                                                  self.end_node._get_clicked_position())
                    # Add neighbor to open set:
                    if neighbor not in open_set_hash:
                        node_count += 1
                        open_set.put((fcost[neighbor], node_count, neighbor))
                        # Add neighbor to hash list:
                        open_set_hash.add(neighbor)
                        neighbor._make_open()

            draw()

            # Close node:
            if current_node != self.start_node:
                current_node._make_closed()

    # Compute H cost value - needs to be always LESS than the actual cost which would be need to go from the current Node to
    # the end Node:
    def _compute_hcost(self, current_node_position, end_node_position):
        # F = G + H
        # F = complete estimated cost for given point
        # G : exact cost from start Node to this Node
        # h : estimated cost from this Node to end Node - requires some estimation heuristic
        x1, y1 = current_node_position
        x2, y2 = end_node_position
        return abs(x1 - x2) + abs(y1 - y2)  # is less costly than the euclidean distance

    def _reconstruct_path(self, dict_nodes_came_from, current_node, draw):
        while current_node in dict_nodes_came_from:
            current_node = dict_nodes_came_from[current_node]
            current_node._make_path()
            draw()

    def _draw_grid(self, win, ROW_COUNT, width):
        gap = width // ROW_COUNT
        for col in range(ROW_COUNT):
            pygame.draw.line(win, GREY, (0, col * gap), (width, col * gap))
            for row in range(ROW_COUNT):
                pygame.draw.line(win, GREY, (row * gap, 0), (row * gap, width))

    def _draw(self):
        self.pygame_window.fill(WHITE)

        for row in GRID:
            for Node in row:
                Node._draw(self.pygame_window)

        self._draw_grid(self.pygame_window, ROW_COUNT, WIDTH_WINDOW)
        pygame.display.update()

    def _getRowsAndCols(self, clicked_position):
        cell_width = WIDTH_WINDOW // ROW_COUNT
        y, x = clicked_position

        row = y // cell_width
        col = x // cell_width

        return row, col

    # Function for taking action on a RIGHT mouse click - goal: _reset Node.
    def _click_right_mouse(self):

        # Function for evaluation clicked_positionition in window:
        clicked_position = pygame.mouse.get_pos()
        # clicked_positionition in window and row/col count gives info in which cell the clicking event happened:
        row, col = self._getRowsAndCols(clicked_position)
        # Access current Node from GRID:
        current_node = GRID[row][col]
        # Right mouse button means we want to _reset:
        current_node._reset()
        if current_node == self.start_node:
            self.start_node = None
        elif current_node == self.end_node:
            self.end_node = None

        return self.start_node, self.end_node

    # Function for taking action on a LEFT mouse click - goal: set start or end Node or _make_barrier
    def _click_left_mouse(self):

        # Function for evaluation clicked_positionition in window:
        clicked_position = pygame.mouse.get_pos()
        # clicked_positionition in window and row/col count gives info in which cell the clicking event happened:
        row, col = self._getRowsAndCols(clicked_position)
        # Access current Node from GRID:
        current_node = GRID[row][col]

        # Algorithm for start_node and end_node selection and drawing walls:
        if not self.start_node and current_node != self.end_node:  # Start
            self.start_node = current_node
            self.start_node._make_start()
        elif not self.end_node and current_node != self.start_node:  # End
            self.end_node = current_node
            self.end_node._make_end()
        elif current_node != self.end_node and current_node != self.start_node:  # Draw Walls
            current_node._make_barrier()

        return self.start_node, self.end_node

    # Function is called in case of a keyboard interaction:
    def _press_key_on_keyboard(self, event, GRID):

        # Space button starts game in case start and end Node have been set:
        # if not both end and start node are set nothing will happen when you press the space bar
        if event.key == pygame.K_SPACE and self.start_node and self.end_node:
            # Update iteratively all Nodes in all ROW_COUNT regarding their neighbors:
            for row in GRID:
                for current_node in row:
                    current_node._update_neighbors(GRID)

            # Call algorithm for evaluating GRID:
            self._run_astar_algorithm(
                lambda: self._draw(), GRID)

        # _resetting game:
        if event.key == pygame.K_c:
            self.start_node = None
            self.end_node = None
            GRID = self._make_grid(ROW_COUNT, WIDTH_WINDOW)

        return self.start_node, self.end_node

def main():


    pygame.display.set_caption("*************   A-STAR Pathfinding Algorithm   *************")

    game_instance = Game()

    run = True
    while run:

        game_instance._draw()
        for event in pygame.event.get():
            # In case of closing the window:
            if event.type == pygame.QUIT:
                run = False

            # Pressing LEFT mouse button:
            if pygame.mouse.get_pressed()[0]:
                game_instance._click_left_mouse()

            # Pressing RIGHT mouse button:
            elif pygame.mouse.get_pressed()[2]:
                game_instance._click_right_mouse()

            # Check pressed keyboard buttons for space button:
            # This means NO CLICKING is evaluated anymore until the algorithm found the solution:
            # AFTER that, the game can be quit or __resetted again:
            elif event.type == pygame.KEYDOWN:
                game_instance._press_key_on_keyboard(event, GRID)

    pygame.quit()


main()