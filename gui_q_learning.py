# Q learning source code

import pygame
import numpy as np
import random
from time import sleep


def convert(x, y):
    return (x * row_length) + y


def detect_click(pos):
    return pos[0] // cell_size, pos[1] // cell_size


class cellCreate:

    def __init__(self, x, y, l, b):
        self.x = x
        self.y = y
        self.l = l
        self.b = b

    def create(self, color):
        pygame.draw.rect(screen, color, (self.x, self.y, self.l, self.b))


row_length, column_length = 20, 20
cell_size = 40
q_table = np.zeros((row_length, column_length, 4))

# print(q_table)
grid = np.zeros((row_length, column_length))

grid.fill(-1)
# print(grid)

pygame.init()
screen = pygame.display.set_mode((row_length * cell_size, column_length * cell_size))

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

temp = []
for i in range(0, row_length):
    for j in range(0, column_length):
        obj = cellCreate(i * cell_size, j * cell_size, cell_size - 1, cell_size - 1)
        obj.create(white)
        temp.append(obj)

running = True
walls = []
start = (0, 0)
target = (0, 0)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            got = pygame.mouse.get_pos()
            print(got)
            click = (detect_click(got))
            walls.append((click[0], click[1]))
            temp[convert(click[0], click[1])].create(black)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            got = pygame.mouse.get_pos()
            print(got)
            click = detect_click(got)
            start = (click[0], click[1])
            temp[convert(click[0], click[1])].create(green)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            got = pygame.mouse.get_pos()
            print(got)
            click = detect_click(got)
            target = (click[0], click[1])
            temp[convert(click[0], click[1])].create(red)
            running = False
    pygame.display.update()

print(walls)
print(start)
print(target)

for item in walls:
    grid[item[0]][item[1]] = -100

start_location_X, start_location_Y = start[0], start[1]

target_location_X, target_location_Y = target[0], target[1]
grid[target_location_X][target_location_Y] = 100

print(grid)


def is_terminal_state(current_row_index, current_column_index):
    if grid[current_row_index, current_column_index] == -1:
        return False
    else:
        return True


def get_start_location():
    start_row = random.randint(0, row_length - 1)
    start_column = random.randint(0, column_length - 1)
    while is_terminal_state(start_row, start_column):
        start_row = random.randint(0, row_length - 1)
        start_column = random.randint(0, column_length - 1)
    return start_row, start_column


def get_next_action(current_row, current_column, take_max=False):
    if random.random() < epsilon or take_max:
        return np.argmax(q_table[current_row, current_column])
    else:
        return random.randint(0, 3)


def next_location(current_row, current_column, action):
    if action == 0 and current_row > 0:
        current_row -= 1
    if action == 1 and current_column < column_length - 1:
        current_column += 1
    if action == 2 and current_row < row_length - 1:
        current_row += 1
    if action == 3 and current_column > 0:
        current_column -= 1
    return current_row, current_column


def get_shortest_path(start_row, start_column):
    counter = 0
    if is_terminal_state(start_row, start_column):
        return []
    else:
        curr_row, curr_column = start_row, start_column
        path = [(curr_row, curr_column)]

        while not is_terminal_state(curr_row, curr_column):
            counter += 1
            if counter == row_length*column_length:
                return []
            take_action = get_next_action(curr_row, curr_column, True)
            next_state = next_location(curr_row, curr_column, take_action)
            curr_row, curr_column = next_state
            path.append(next_state)
        return path


# NOW WE START TRAINING OUR MODEL :
epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9
for episode in range(10000):
    row, column = get_start_location()

    while not is_terminal_state(row, column):
        action_index = get_next_action(row, column)
        row_old, column_old = row, column

        row, column = next_location(row_old, column_old, action_index)
        reward = grid[row, column]

        old_q_value = q_table[row_old, column_old, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_table[row, column])) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_table[row_old, column_old, action_index] = new_q_value

path_followed = get_shortest_path(start_location_X, start_location_Y)
print(path_followed)

running = True

if len(path_followed) == 0:
    print("No Possible Path Exists")

else:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        for cell in path_followed:
            temp[convert(cell[0], cell[1])].create(green)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break
            sleep(0.5)

pygame.quit()
print("----------------------------------------------------")
