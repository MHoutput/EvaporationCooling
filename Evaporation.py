# Evaporation cooling applet
# By Matthew Houtput (matthew.houtput@uantwerpen.be)
# Original idea and implementation: Physics-2000, JILA (University of Colorado, Boulder)

# Requires the NumPy and Pygame packages to be installed on your system

import os
import sys

import pygame
import pygame.locals

import numpy as np
import math
from random import uniform

# These two lines are necessary to package the .py file into an executable using PyInstaller,
# but can be ignored if the script is simply run as Python code
# If the code runs as an executable ('frozen'), change directory to the temporary MEIPASS folder
# where all external files are stored
# On Windows, the MEIPASS folder is located at C:\Users\xxxxxx\AppData\Local\Temp\_MEIxxxxxx
if getattr(sys, 'frozen', False):
    # noinspection PyProtectedMember
    os.chdir(sys._MEIPASS)

# Initialize some useful constants
PLAY_WIDTH = 496
PLAY_HEIGHT = 528
LEFT_BORDER = 48
RIGHT_BORDER = 96
TOP_BORDER = 48
BOTTOM_BORDER = 96
WINDOW_WIDTH = LEFT_BORDER + PLAY_WIDTH + RIGHT_BORDER
WINDOW_HEIGHT = TOP_BORDER + PLAY_HEIGHT + BOTTOM_BORDER
FPS = 30

GRAVITY = 10 / FPS
GRAVITY_VEC = np.array([0, GRAVITY])  # If you really want you can change the direction of gravity with this
ATOM_RADIUS = 12
ATOM_STARTING_NUMBER = 49
PARABOLA_CURVATURE = 0.01
PARABOLA_XY = (LEFT_BORDER + PLAY_WIDTH / 2, WINDOW_HEIGHT - BOTTOM_BORDER - 32)

WHITE = (255, 255, 255)
LIGHT_GRAY = (159, 159, 159)
GRAY = (127, 127, 127)
DARK_GRAY = (79, 79, 79)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


# ===== MAIN FUNCTION =====
def main():
    # Set up pygame and the display
    pygame.init()
    fpsclock = pygame.time.Clock()
    displaysurf = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Evaporation cooling')

    # Initialize fonts
    font_subscript = pygame.font.SysFont('verdana', 11)
    font_small = pygame.font.SysFont('verdana', 14)
    font_normal = pygame.font.SysFont('verdana', 18)
    font_large = pygame.font.SysFont('verdana', 24)
    font_huge = pygame.font.SysFont('verdana', 40)

    # Initialize some mouse variables
    mouse_xy = (0, 0)
    mouse_is_down = False

    # Create the slider bar, the restart button, and the pause button
    s_para_height = Slider(displaysurf,
                           (LEFT_BORDER + 192, WINDOW_HEIGHT - BOTTOM_BORDER + 35,
                            WINDOW_WIDTH - RIGHT_BORDER - 32 - (LEFT_BORDER + 192), 8),
                           (25., 450.), 300., 10, 'Trap height', font_normal, LIGHT_GRAY)
    b_restart = ImageButton(displaysurf, (LEFT_BORDER + 96, WINDOW_HEIGHT - BOTTOM_BORDER + 16, 64, 64),
                            'images/Restart_idle.png', 'images/Restart_hover.png')
    b_paused = ToggleButton(displaysurf, (LEFT_BORDER + 16, WINDOW_HEIGHT - BOTTOM_BORDER + 16, 64, 64),
                            'images/Play_idle.png', 'images/Pause_idle.png',
                            'images/Play_hover.png', 'images/Pause_hover.png')

    # Initialize the game:
    parabola_height, game_paused, flag_restart, atoms, initial_critical_temperature, maximum_temperature, \
        atom_collision_times = setup(s_para_height, b_paused)

    # Define the critical temperature as a function of the number of atoms
    def critical_temperature(n):
        return initial_critical_temperature * math.pow(n / (ATOM_STARTING_NUMBER + 1e-4), 2 / 3)

    # Main game loop:
    while True:
        # Get the user input from the mouse
        mouse_is_clicked = False
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT \
                    or (event.type == pygame.locals.KEYUP and event.key == pygame.locals.K_ESCAPE):
                pygame.quit()
                sys.exit()  # Close the applet
            elif event.type == pygame.locals.MOUSEMOTION:
                mouse_xy = event.pos
            elif event.type == pygame.locals.MOUSEBUTTONDOWN and event.button == 1:
                mouse_xy = event.pos
                mouse_is_down = True
                mouse_is_clicked = True
            elif event.type == pygame.locals.MOUSEBUTTONUP and event.button == 1:
                mouse_xy = event.pos
                mouse_is_clicked = False
                mouse_is_down = False
        mouse_state = (mouse_xy, mouse_is_clicked, mouse_is_down)

        # Calculate the temperature
        current_temperature = get_temperature(atoms, PARABOLA_XY)
        current_critical_temperature = critical_temperature(len(atoms))
        if current_temperature < current_critical_temperature:
            player_won = True
        else:
            player_won = False

        # Draw everything: background, parabola, atoms, margins, ...
        pygame.draw.rect(displaysurf, LIGHT_GRAY, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))  # Draw the gray background
        for atom in atoms:
            atom.draw(displaysurf)
        draw_parabola(PARABOLA_CURVATURE, PARABOLA_XY, parabola_height, displaysurf)
        draw_borders(displaysurf)
        draw_thermometer(current_temperature, current_critical_temperature, maximum_temperature, displaysurf,
                         font_small, font_subscript)
        draw_text(displaysurf, atoms, font_large, font_huge, player_won)

        # Control the buttons and the slider bar
        flag_restart = b_restart.control(mouse_state)
        game_paused = b_paused.control(mouse_state)
        parabola_height = s_para_height.control(mouse_state)

        if flag_restart:
            # Re-setup the game if the restart button is clicked
            parabola_height, game_paused, flag_restart, atoms, initial_critical_temperature, maximum_temperature, \
                atom_collision_times = setup(s_para_height, b_paused)

        if not game_paused:
            # Move all the atoms one step
            move_atoms(atoms, atom_collision_times, parabola_height, player_won)
            atom_collision_times = remove_outside_atoms(atoms, atom_collision_times)

        # Update the screen and wait until the next step:
        pygame.display.update()
        fpsclock.tick(FPS)


# ===== ATOM CLASS =====
class Atom:
    # An object that represents one atom
    # Most of the code that controls the simulation (movement, collisions, ...) is in other functions. This is basically
    # just an object that is drawn.
    # position and velocity are stored as 2x1 NumPy arrays, but the input can also be in the form of a tuple or a list

    def __init__(self, position=np.array([0, 0]), velocity=np.array([0, 0]),
                 radius=ATOM_RADIUS, mass=1, image_path='images/Sphere.png'):
        self.position = np.asarray(position)  # Convert input to an array
        self.velocity = np.asarray(velocity)  # Convert input to an array
        self.radius = radius
        self.mass = mass
        self.image_path = image_path
        self.image = pygame.transform.scale(pygame.image.load(self.image_path), (2 * self.radius, 2 * self.radius))

    def draw(self, surface):
        # The atom is drawn with its xy-position in the center
        surface.blit(self.image, tuple(self.position - self.radius))

    def set_radius(self, radius):
        # Set the radius of the atom
        self.radius = radius
        self.image = pygame.transform.scale(pygame.image.load(self.image_path), (2 * self.radius, 2 * self.radius))


# ===== ATOM MOVEMENT AND COLLISIONS =====
def move_atoms(atoms, atom_collision_times, parabola_height, player_won=False):
    # This overarching function moves all atoms to their new position after one time step, including collisions
    # The idea is that we freely propagate all atoms until there is a collision, then perform the collision, freely
    # propagate again, ... and so on, until the elapsed time is one time step
    if not atoms:
        pass  # If there are no atoms, do nothing
    else:
        time_remaining = 1.
        # We recalculate para_collision_times every time step because it is approximate
        para_collision_times = np.array([collision_time_parabola(atom, parabola_height) for atom in atoms])
        if player_won:
            damping = 0.2
        else:
            damping = 0.
        while time_remaining > 1e-4:
            para_min_index = np.argmin(para_collision_times)
            time_to_para_collision = para_collision_times[para_min_index]  # Time to parabola collision
            if player_won:
                atom_min_index = 0
                time_to_atom_collision = 1000000
            else:
                atom_min_index = np.argmin(atom_collision_times)
                time_to_atom_collision = atom_collision_times.flat[atom_min_index]  # Time to next atom-atom collision
            time_to_propagate = 0.99 * min(time_to_para_collision, time_to_atom_collision, time_remaining)
            # To avoid atom overlap, we only propagate for 99% of the time to the collision
            propagate_time(atoms, time_to_propagate)
            # To prevent atoms getting stuck in an infinite collision loop, we perform an "emergency" collision whenever
            # a collision happens immediately after another one
            emergency_collision = time_to_propagate < 1e-3

            if time_remaining < time_to_para_collision and time_remaining < time_to_atom_collision:
                # There are no more collisions in this step
                indices_to_update = ()
            else:
                if time_to_para_collision < time_to_atom_collision:
                    # An atom will collide with the parabola
                    collide_parabola(atoms[para_min_index], damping, emergency_collision)
                    indices_to_update = (para_min_index,)
                else:
                    # Two atoms will collide with each other
                    atom1_index = atom_min_index // len(atoms)
                    atom2_index = atom_min_index % len(atoms)
                    collide_atoms(atoms[atom1_index], atoms[atom2_index])
                    indices_to_update = (atom1_index, atom2_index)

            time_remaining -= time_to_propagate
            # Recalculate the collision times of all atoms that have collided
            update_collision_times(atoms, time_to_propagate, indices_to_update, para_collision_times,
                                   atom_collision_times, parabola_height)


def update_collision_times(atoms, propagation_time, indices_to_update,
                           para_collision_times, atom_collision_times, parabola_height):
    # This function calculates the new collision times after one or more collisions have occurred
    para_collision_times -= propagation_time
    atom_collision_times -= propagation_time
    for atom_index in indices_to_update:
        atom_to_collide = atoms[atom_index]
        para_collision_times[atom_index] = collision_time_parabola(atom_to_collide, parabola_height)
        atom1_collision_times = np.array([collision_time_atoms(atom_to_collide, atom) for atom in atoms])
        atom_collision_times[atom_index, :] = atom1_collision_times
        atom_collision_times[:, atom_index] = atom1_collision_times


def propagate_time(atoms, time):
    # This function freely propagates all the atoms over a time t, without collisions
    for atom in atoms:
        atom.position += atom.velocity * time + 0.5 * GRAVITY_VEC * time ** 2
        atom.velocity += GRAVITY_VEC * time


def collide_atoms(atom1, atom2):
    # Calculate the new velocities of two atoms that collide with each other, and update them
    if atom1 is atom2:
        # An atom cannot collide with itself
        pass
    else:
        relative_velocity = atom1.velocity - atom2.velocity
        mass1 = atom1.mass
        mass2 = atom2.mass
        n_vec = -relative_velocity / np.linalg.norm(relative_velocity)
        momentum_exchange = 2 * mass1 * mass2 / (mass1 + mass2) * (relative_velocity @ n_vec) * n_vec
        atom1.velocity -= momentum_exchange / mass1
        atom2.velocity += momentum_exchange / mass2


def collision_time_atoms(atom1, atom2):
    # This function returns the time it will take before atom 1 and atom 2 will collide
    # The result is exact and does not depend on the strength of gravity
    pos1 = atom1.position
    pos2 = atom2.position
    vel1 = atom1.velocity
    vel2 = atom2.velocity
    dd = (atom1.radius + atom2.radius) ** 2  # Squared sum of the atom radii
    xx = (pos1 - pos2) @ (pos1 - pos2)  # Squared distance between the atoms
    vv = (vel1 - vel2) @ (vel1 - vel2)  # Squared relative velocity of the atoms
    xv = (pos1 - pos2) @ (vel1 - vel2)
    if xv ** 2 - vv * (xx - dd) > 0 and atom1 is not atom2:
        # The atoms will cross paths, so there can be a collision
        collision_time = (-xv - np.sqrt(abs(xv ** 2 - vv * (xx - dd)))) / vv
        if collision_time < 0:
            # This collision would take place in the past, so we ignore it
            collision_time = 1000000.
    else:
        # There is no collision, so we return some large placeholder value
        collision_time = 1000000.
    return collision_time


def collide_parabola(atom, damping=0., emergency=False, parabola_top=PARABOLA_XY,
                     parabola_curvature=PARABOLA_CURVATURE):
    # Calculate the new velocity of an atom that collides with the parabola, and update it
    # "damping" makes the atoms lose a fraction of their velocity when they bounce
    pos = atom.position
    vel = atom.velocity
    if np.sqrt(vel @ vel) < 1:
        damping = 0.  # We want the damping to stop if the velocity of the atoms is low enough
    collision_point = para_collision_point(pos, parabola_top, parabola_curvature)
    slope = -2 * parabola_curvature * (collision_point[0] - parabola_top[0])
    slope_vec = np.array([slope, -1])
    if emergency:
        # "Emergency" collision: Bounce away from the parabola
        atom.velocity = (1 - damping) * np.sqrt((vel @ vel) / (slope_vec @ slope_vec)) * slope_vec
    else:
        # "Normal" collision: Reflect around the tangent line of the parabola
        transformation_matrix = np.array([[1 - slope ** 2, 2 * slope], [2 * slope, -1 + slope ** 2]]) / (
                slope_vec @ slope_vec)
        atom.velocity = (1 - damping) * transformation_matrix @ vel


def collision_time_parabola(atom, parabola_height,
                            gravity=GRAVITY_VEC, parabola_top=PARABOLA_XY, parabola_curvature=PARABOLA_CURVATURE):
    # Calculate the time before the atom will collide with the parabola
    # The result is an approximation that gets better if the atom is closer to the parabola
    pos = atom.position - parabola_top  # Position relative to the top of the parabola
    vel = atom.velocity
    r = atom.radius
    collision_time = 1000000.  # Some large placeholder value
    if -parabola_curvature * pos[0] ** 2 > pos[1] > -parabola_height - r:
        # The atom must be inside the parabola, but not too high
        # Instead of calculating an exact collision time, we calculate the time until the atom collides with one of two
        # approximating lines: the line from point 1 to point 2, and the line from point 1 to point 3
        # As the atom gets closer, this approximation becomes exact
        point1 = np.array([pos[0], -parabola_curvature * pos[0] ** 2])  # Point on the parabola directly below the atom
        point2 = np.array(
            [-np.sqrt(abs(pos[1] / parabola_curvature)), pos[1]])  # Point on the parabola left of the atom
        point3 = np.array(
            [np.sqrt(abs(pos[1] / parabola_curvature)), pos[1]])  # Point on the parabola right of the atom
        for vec in (point1 - point2, point3 - point1):
            nqp = vec / np.sqrt(vec @ vec)
            if np.cross(vel, nqp) < 0:  # "True" if the velocity of the atom points towards the line
                # b = vec[1] / vec[0]
                # c = point1[1] - b*point1[0]
                # s = np.sqrt(1 + b**2)
                # disc = ((vel[1] - b*vel[0])**2 - 2*gravity[1]*(pos[1] - b*pos[0] - c + r*s))
                a = np.cross(0.5 * gravity, nqp)
                b = np.cross(vel, nqp)
                c = np.cross(pos - point1, nqp)
                disc = b * b - 4 * a * (c - r)
                if disc >= 0:
                    # Formula for the time until the atom collides with the line
                    # collision_time = min(collision_time, -2*(pos[1]-b*pos[0]-c+r * s)/(vel[1]-b*vel[0]+np.sqrt(disc)))
                    collision_time = min(collision_time, (-b - np.sqrt(disc)) / (2 * a))
                else:
                    # There is no collision, ignore it
                    collision_time = 1000000.

    return max(collision_time, 0.)


def para_collision_point(point_xy, parabola_top=PARABOLA_XY, parabola_curvature=PARABOLA_CURVATURE):
    # This function calculates the intersection point between the parabola (with given top and curvature) and the
    # perpendicular line from the given point to that parabola
    # This point will be the point around which the atom will reflect, on a collision
    a = parabola_curvature
    x1 = point_xy[0] - parabola_top[0]
    y1 = point_xy[1] - parabola_top[1]
    intersect_x = x1
    for i in range(5):  # Basic Newton-Raphson method
        intersect_x = (x1 + 4 * a ** 2 * intersect_x ** 3) / (1 + 2 * a * y1 + 6 * a ** 2 * intersect_x ** 2)
    intersect_y = -a * intersect_x ** 2
    return np.array([intersect_x, intersect_y]) + parabola_top


# ===== MISCELLANEOUS FUNCTIONS =====
def setup(slider_para_height, button_paused):
    # Initialisation of the different variables
    parabola_height = slider_para_height.set_slider_value(400.)
    game_paused = True
    button_paused.set_state(game_paused)
    flag_restart = False
    atoms = create_atoms(ATOM_STARTING_NUMBER)
    initial_critical_temperature = 0.75 * get_temperature(atoms, PARABOLA_XY)
    maximum_temperature = 1.2 * get_temperature(atoms, PARABOLA_XY) + 1e-4
    atom_collision_times = np.array([[collision_time_atoms(atom1, atom2) for atom1 in atoms] for atom2 in atoms])

    return parabola_height, game_paused, flag_restart, atoms, initial_critical_temperature, maximum_temperature, \
        atom_collision_times


def create_atoms(amount, parabola_top=PARABOLA_XY, parabola_curvature=PARABOLA_CURVATURE, random_radius=8,
                 vel_base=100 / FPS, vel_step=10 / FPS):
    # This function creates amount atoms, semi-randomly distributed inside the parabola
    # Their velocities are also chosen randomly, with higher atoms having higher velocity on average

    a = parabola_curvature
    r = ATOM_RADIUS + random_radius
    atom_positions = np.empty((amount, 2))  # Position of the atoms, relative to the top of the parabola
    atom_velocities = np.empty((amount, 2))  # Velocities of the atoms
    y_min = -1
    count = 0
    row = 0
    while count < amount:
        dx = 2 * np.sqrt(abs(-y_min / a))
        if row == 0:
            row_count = 1
            xmin = -1
            xmax = 1
        else:
            row_count = int(np.floor(dx / (2 * r)))
            xmin = -0.5 * dx + r
            xmax = 0.5 * dx - r
        for i in range(row_count):
            if count + i >= amount:
                break
            else:
                atom_positions[count + i, :] = np.array([xmin + i * (xmax - xmin)/(row_count - 1 + 1e-10), y_min - r])\
                                               + np.random.rand(2) * 2 * random_radius - random_radius
                vel_size = uniform(vel_base + row * vel_step, vel_base + (row + 1) * vel_step)
                vel_angle = uniform(0, 2 * math.pi)
                atom_velocities[count + i, :] = vel_size * np.array([math.cos(vel_angle), math.sin(vel_angle)])
        count += row_count
        row += 1
        y_min -= 2 * r

    atom_positions += np.tile(parabola_top, (amount, 1))
    atoms = [Atom(position, velocity) for position, velocity in zip(atom_positions, atom_velocities)]
    return atoms


def remove_outside_atoms(atoms, atom_collision_times):
    # This function removes any atom that goes outside the screen
    removed_atom = False
    for atom in atoms:
        if not (0 < atom.position[0] < WINDOW_WIDTH and atom.position[1] < WINDOW_HEIGHT):
            removed_atom = True
            atoms.remove(atom)
    if removed_atom:
        atom_collision_times = np.array([[collision_time_atoms(atom1, atom2) for atom1 in atoms] for atom2 in atoms])
    return atom_collision_times


def get_temperature(atoms, parabola_top):
    def getEnergy(atom): return 0.5 * atom.mass * (atom.velocity @ atom.velocity) \
                                - atom.mass * GRAVITY_VEC @ (atom.position - parabola_top)

    total_energy = sum(map(getEnergy, atoms))
    return total_energy / (len(atoms) + 1e-4)


# ===== DRAW FUNCTIONS =====
def draw_parabola(curvature, top_xy, height, surface):
    # Draw a parabola to the given surface
    para_half_width = math.ceil(math.sqrt(height / curvature))
    x_top = top_xy[0]
    y_top = top_xy[1]
    x_values = np.arange(x_top - para_half_width, x_top + para_half_width + 1)
    y_values = y_top - curvature * (x_values - x_top) ** 2
    pixel_array = np.array([x_values, y_values])
    for i in range(len(x_values) - 2):
        pygame.draw.line(surface, BLACK, tuple(pixel_array[:, i]), tuple(pixel_array[:, i + 2]), 6)


def draw_borders(surface):
    # Draw the dark gray borders on the edges of the screen
    pygame.draw.rect(surface, DARK_GRAY, (0, 0, LEFT_BORDER, WINDOW_HEIGHT))
    pygame.draw.rect(surface, DARK_GRAY, (0, 0, WINDOW_WIDTH, TOP_BORDER))
    pygame.draw.rect(surface, DARK_GRAY, (WINDOW_WIDTH - RIGHT_BORDER, 0, RIGHT_BORDER, WINDOW_HEIGHT))
    pygame.draw.rect(surface, DARK_GRAY, (0, WINDOW_HEIGHT - BOTTOM_BORDER, WINDOW_WIDTH, BOTTOM_BORDER))
    pygame.draw.rect(surface, BLACK, (LEFT_BORDER, TOP_BORDER, PLAY_WIDTH, PLAY_HEIGHT), 1)


def draw_thermometer(temperature, crit_temperature, maximum_temperature, surface, small_font, subscript_font):
    # This function draws the thermometer on the screen

    # The fixed part of the thermometer:
    radius = 32
    half_radius = int(radius / 2)
    quarter_radius = int(radius / 4)
    y_min = WINDOW_HEIGHT - BOTTOM_BORDER - radius
    y_max = TOP_BORDER + radius + half_radius
    x_mid = WINDOW_WIDTH - int(RIGHT_BORDER / 2)
    pygame.draw.circle(surface, WHITE, (x_mid, y_min), radius)
    pygame.draw.circle(surface, WHITE, (x_mid, y_max - half_radius), radius)
    pygame.draw.circle(surface, BLACK, (x_mid, y_min), radius, 1)
    pygame.draw.circle(surface, BLACK, (x_mid, y_max - half_radius), radius, 1)
    pygame.draw.rect(surface, WHITE, (x_mid - radius, y_max - half_radius, 2 * radius, y_min - y_max + half_radius))
    pygame.draw.line(surface, BLACK, (x_mid - radius, y_min), (x_mid - radius, y_max - half_radius))
    pygame.draw.line(surface, BLACK, (x_mid + radius, y_min), (x_mid + radius, y_max - half_radius))
    pygame.draw.circle(surface, DARK_GRAY, (x_mid, y_max - 3 * quarter_radius), quarter_radius)
    pygame.draw.circle(surface, BLACK, (x_mid, y_max - 3 * quarter_radius), quarter_radius, 1)
    pygame.draw.rect(surface, BLACK, (x_mid - quarter_radius, y_max, half_radius, y_min - y_max), 1)
    pygame.draw.circle(surface, RED, (x_mid, y_min), half_radius)
    pygame.draw.circle(surface, BLACK, (x_mid, y_min), half_radius, 1)

    # The "mercury" inside the thermometer indicating the temperature, plus a line for the critical temperature:
    rectangle_height = (y_min - y_max - radius / 2) * temperature / maximum_temperature
    crit_line_height = (y_min - y_max - radius / 2) * crit_temperature / maximum_temperature
    pygame.draw.rect(surface, RED, (x_mid - quarter_radius + 1, y_min - rectangle_height,
                                    half_radius - 2, rectangle_height))
    pygame.draw.line(surface, BLACK, (x_mid - radius / 4, y_min - crit_line_height),
                     (x_mid + radius / 4 - 1, y_min - crit_line_height))

    # The label for the critical temperature:
    temperature_text = small_font.render('T', True, BLACK)
    temperature_text_xy = (x_mid + radius / 4 + 2, y_min - crit_line_height - small_font.size('T')[1] / 2)
    subscript_text = subscript_font.render('c', True, BLACK)
    subscript_text_xy = (temperature_text_xy[0] + 7, temperature_text_xy[1] + 5)
    surface.blit(temperature_text, temperature_text_xy)
    surface.blit(subscript_text, subscript_text_xy)


def draw_text(surface, atoms, title_font, score_font, player_won=False):
    # Draw all the necessary text

    # Title
    title_string = 'Evaporation cooling'
    title_text = title_font.render(title_string, True, LIGHT_GRAY)
    title_text_size = title_font.size(title_string)
    title_text_xy = ((WINDOW_WIDTH - title_text_size[0]) / 2,
                     (TOP_BORDER - title_text_size[1]) / 2)
    surface.blit(title_text, title_text_xy)

    # Score, after the player wins the game:
    if player_won:
        score_string = 'Score: ' + str(len(atoms))
    else:
        score_string = ''
    score_text = score_font.render(score_string, True, DARK_GRAY)
    score_text_size = score_font.size(score_string)
    score_text_xy = (LEFT_BORDER + (PLAY_WIDTH - score_text_size[0]) / 2,
                     TOP_BORDER + (PLAY_HEIGHT - score_text_size[1]) / 2)
    surface.blit(score_text, score_text_xy)


# ===== BUTTONS AND SLIDERS ===== #
class Button:  # Bare-bones button, we are likely not going to make any of these

    def __init__(self, surface, bounding_rectangle):
        self.surface = surface
        self.bounding_rectangle = bounding_rectangle
        self.x = bounding_rectangle[0]
        self.y = bounding_rectangle[1]
        self.width = bounding_rectangle[2]
        self.height = bounding_rectangle[3]

    def action(self):  # This function is what happens when the button is clicked
        return True

    def idle(self):  # This function is what happens while the button is not clicked
        return False

    def is_active(self, mouse_state):
        # A function that determines whether the effect of the button should activate
        # mouse_state is a tuple of the form (mouse_xy, mouse_is_clicked, mouse_is_down)
        # Current implementation: Activates only on the exact frame the button is clicked
        if self.check_mouse(mouse_state[0]) and mouse_state[1]:
            return True
        else:
            return False

    def check_mouse(self, mouse_xy):
        # Checks if the mouse is inside the button
        # Current implementation: Rectangular hitbox
        mouse_x = mouse_xy[0]
        mouse_y = mouse_xy[1]
        if self.x <= mouse_x <= self.x + self.width and self.y <= mouse_y <= self.y + self.height:
            mouse_inside = True
        else:
            mouse_inside = False
        return mouse_inside

    def draw(self, mouse_state):
        pygame.draw.rect(self.surface, BLACK, self.bounding_rectangle)

    def control(self, mouse_state):
        # This function can be called in the main game loop to handle the entire button
        self.draw(mouse_state)
        if self.is_active(mouse_state):
            return self.action()
        else:
            return self.idle()


class ImageButton(Button):
    # This is a button with an image. There are three images: one for the idle state, one when the mouse hovers over,
    # and one when the button is clicked.
    # Action inputs: None
    # Action outputs: is_pressed

    def __init__(self, surface, bounding_rectangle, idle_image_path, hover_image_path=None, down_image_path=None):
        if hover_image_path is None:
            hover_image_path = idle_image_path
        if down_image_path is None:
            down_image_path = hover_image_path
        super().__init__(surface, bounding_rectangle)
        self.idle_image_path = idle_image_path
        self.hover_image_path = hover_image_path
        self.down_image_path = down_image_path
        self.idle_image = pygame.transform.scale(pygame.image.load(idle_image_path), (self.width, self.height))
        self.hover_image = pygame.transform.scale(pygame.image.load(hover_image_path), (self.width, self.height))
        self.down_image = pygame.transform.scale(pygame.image.load(down_image_path), (self.width, self.height))

    def draw(self, mouse_state):
        mouse_xy = mouse_state[0]
        mouse_is_clicked = mouse_state[1]
        if self.check_mouse(mouse_xy):
            if mouse_is_clicked:
                image = self.down_image
            else:
                image = self.hover_image
        else:
            image = self.idle_image

        self.surface.blit(image, (self.x, self.y))


class ToggleButton(ImageButton):
    # This is a button with two states: "on" and "off". The button can be clicked to change the state.
    # Action inputs: None
    # Action outputs: is_on

    def __init__(self, surface, bounding_rectangle, idle_on_image_path, idle_off_image_path=None,
                 hover_on_image_path=None, hover_off_image_path=None, down_on_image_path=None, down_off_image_path=None,
                 is_on=True):
        if idle_off_image_path is None:
            idle_off_image_path = idle_on_image_path
        if hover_on_image_path is None:
            hover_on_image_path = idle_on_image_path
        if hover_off_image_path is None:
            down_off_image_path = idle_off_image_path
        if down_on_image_path is None:
            down_on_image_path = hover_on_image_path
        if down_off_image_path is None:
            down_off_image_path = hover_off_image_path
        super().__init__(surface, bounding_rectangle, idle_on_image_path, hover_on_image_path, down_on_image_path)
        self.idle_on_image = pygame.transform.scale(pygame.image.load(idle_on_image_path), (self.width, self.height))
        self.hover_on_image = pygame.transform.scale(pygame.image.load(hover_on_image_path), (self.width, self.height))
        self.down_on_image = pygame.transform.scale(pygame.image.load(down_on_image_path), (self.width, self.height))
        self.idle_off_image = pygame.transform.scale(pygame.image.load(idle_off_image_path), (self.width, self.height))
        self.hover_off_image = pygame.transform.scale(pygame.image.load(hover_off_image_path),
                                                      (self.width, self.height))
        self.down_off_image = pygame.transform.scale(pygame.image.load(down_off_image_path), (self.width, self.height))
        self.is_on = is_on
        self.update_images()

    def update_images(self):
        if self.is_on:
            self.idle_image = self.idle_on_image
            self.hover_image = self.hover_on_image
            self.down_image = self.down_on_image
        else:
            self.idle_image = self.idle_off_image
            self.hover_image = self.hover_off_image
            self.down_image = self.down_off_image

    def set_state(self, state):
        self.is_on = state
        self.update_images()

    def action(self):
        self.set_state(not self.is_on)
        return self.is_on

    def idle(self):
        return self.is_on


class Slider:  # A slider bar

    def __init__(self, surface, bounding_rectangle, minmax=(0., 1.), starting_value=None, slider_radius=8,
                 text_string='', text_font=None, text_color=(0, 0, 0)):
        self.surface = surface
        self.bounding_rectangle = bounding_rectangle
        self.x = bounding_rectangle[0]
        self.y = bounding_rectangle[1]
        self.width = bounding_rectangle[2]
        self.height = bounding_rectangle[3]
        self.minmax = minmax
        self.min_value = minmax[0]
        self.max_value = minmax[1]
        if starting_value is None:
            self.activation = 0.5
        else:
            self.activation = min(max((starting_value - self.min_value) / (self.max_value - self.min_value), 0), 1)
        self.slider_radius = slider_radius
        self.sliding = False
        self.text_string = text_string
        if text_font is None:
            self.text_font = pygame.font.SysFont('arial', 11)
        else:
            self.text_font = text_font
        self.text_color = text_color

    def get_slider_activation(self, slider_x):
        return min(max((slider_x - self.x) / self.width, 0), 1)

    def get_slider_xy(self, activation=None):
        if activation is None:
            activation = self.activation
        return self.x + activation * self.width, self.y + 0.5 * self.height

    def get_slider_value(self, activation=None):
        if activation is None:
            activation = self.activation
        return (1 - activation) * self.min_value + activation * self.max_value

    def set_slider_value(self, value):
        self.activation = min(max((value - self.min_value) / (self.max_value - self.min_value), 0), 1)
        return self.get_slider_value()

    def is_sliding(self, mouse_state):
        # A function that determines whether the slider is sliding or not
        # mouse_state is a tuple of the form (mouse_xy, mouse_is_clicked, mouse_is_down)
        if mouse_state[2]:
            if mouse_state[1] and self.check_mouse(mouse_state[0]):
                self.sliding = True
        else:
            self.sliding = False
        return self.sliding

    def check_mouse(self, mouse_xy):
        # Checks if the mouse is on the slider
        # The hitbox is a rectangle, as wide as the slider bar and as high as the slider button
        mouse_x = mouse_xy[0]
        mouse_y = mouse_xy[1]
        if self.x <= mouse_x <= self.x + self.width \
                and self.y - self.slider_radius <= mouse_y <= self.y + self.slider_radius:
            mouse_inside = True
        else:
            mouse_inside = False
        return mouse_inside

    def draw(self):
        pygame.draw.rect(self.surface, WHITE, self.bounding_rectangle)
        pygame.draw.rect(self.surface, BLACK, self.bounding_rectangle, 1)
        pygame.draw.circle(self.surface, GRAY, self.get_slider_xy(), self.slider_radius)
        pygame.draw.circle(self.surface, BLACK, self.get_slider_xy(), self.slider_radius, 1)

        # Draw the text
        text_surf = self.text_font.render(self.text_string, True, self.text_color)
        text_size = self.text_font.size(self.text_string)
        text_xy = (self.x + (self.width - text_size[0]) / 2,
                   self.y + self.height / 2 + self.slider_radius)
        self.surface.blit(text_surf, text_xy)

    def control(self, mouse_state):
        # This function can be called in the main game loop to handle the entire slider
        mouse_xy = mouse_state[0]
        self.draw()
        if self.is_sliding(mouse_state):
            self.activation = self.get_slider_activation(mouse_xy[0])
        return self.get_slider_value()


if __name__ == '__main__':
    main()
