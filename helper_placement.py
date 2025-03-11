import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import os

# -------------------------------
# Configuration and Dimensions
# -------------------------------
furniture_dims = {
    'bed': (2.0, 1.5),       # (length, width)
    'dresser': (1.0, 0.5),
    'nightstand': (0.5, 0.5),
    'table': (1.2, 0.8),
    'desk': (1.0, 0.5)       # new furniture piece
}
pillar_dims = (0.8, 0.8)     # (width, depth) of the pillar
margin = 0.2                 # fixed margin from walls
table_offset = 2             # offset to move table closer to door
desk_offset = 0.3            # offset for desk placement
dresser_door_offset = 0.7    # offset for dresser placement

# -------------------------------
# Helper Functions for Placement
# -------------------------------
def place_furniture_fixed(room_length, room_width, furniture, wall, margin=margin):
    """
    Places furniture flush against a given wall.
    """
    length, width = furniture_dims[furniture]
    if wall == 'left':
        x = margin + width/2
        y = room_width/2
    elif wall == 'right':
        x = room_length - margin - width/2
        y = room_width/2
    elif wall == 'top':
        y = room_width - margin - width/2
        x = room_length/2
    else:  # bottom
        y = margin + width/2
        x = room_length/2
    return (round(x,2), round(y,2))

def place_nightstand_left(bed_center, bed_wall, margin=margin):
    """
    Places the nightstand on the physical left side of the bed.
    For vertical beds (if bed is flush on left/right wall), we assume the bed's long axis is vertical so
    the left side is in the negative y direction.
    For horizontal beds (if flush on top/bottom), the left side is in the negative x direction.
    """
    bed_length, bed_width = furniture_dims['bed']
    ns_length, ns_width = furniture_dims['nightstand']
    x, y = bed_center
    if bed_wall in ['left', 'right']:
        offset = (bed_length/2) + (ns_length/2) + margin
        x_new = x
        y_new = y - offset
    else:
        offset = (bed_width/2) + (ns_width/2) + margin
        x_new = x - offset
        y_new = y
    return (round(x_new,2), round(y_new,2))

def place_nightstand_adjacent(bed_center, margin=margin):
    """
    Always places the nightstand to the left of the bed.
    This ignores the bed's wall and shifts the nightstand horizontally left relative to the bed center.
    """
    bed_length, bed_width = furniture_dims['bed']
    ns_length, ns_width = furniture_dims['nightstand']
    x, y = bed_center
    # Shift left: subtract half the bed's width plus margin plus half the nightstand's width.
    x_new = x - (bed_width / 2 + ns_width / 2 + margin)
    return (round(x_new, 2), round(y, 2))

def place_table_near_door(room_length, room_width, door_wall, margin=margin, offset=table_offset):
    """
    Places the table closer to the door.
    """
    table_length, table_width = furniture_dims['table']
    if door_wall == 'left':
        x = margin + table_width/2 + offset
        y = room_width/2
    elif door_wall == 'right':
        x = room_length - margin - table_width/2 - offset
        y = room_width/2
    elif door_wall == 'top':
        y = room_width - margin - table_width/2 - offset
        x = room_length/2
    else:  # bottom
        y = margin + table_width/2 + offset
        x = room_length/2
    return (round(x,2), round(y,2))


#helper functions for stage-2
def place_dresser_right_of_door(room_length, room_width, door_wall, door_pos,
                                margin=0.2, offset=0.7):
    """
    Places the dresser on the occupant's 'right side' from the perspective
    of someone standing at the door and facing inward.

    - bottom wall => occupant faces up => occupant's right => x > door_pos
    - top wall => occupant faces down => occupant's right => x < door_pos
    - left wall => occupant faces right => occupant's right => y < door_pos
    - right wall => occupant faces left => occupant's right => y > door_pos

    The dresser is flush with the door's wall and offset in the perpendicular axis.
    Clamps the position if it goes out of bounds.
    """
    dresser_length, dresser_width = furniture_dims['dresser']

    if door_wall == 'bottom':
        # Flush with bottom wall
        # occupant faces up => occupant's right => x > door_pos
        y = margin + dresser_width / 2
        x = door_pos + (dresser_length / 2) + offset
        # clamp if out of bounds
        if x + (dresser_length / 2) > room_length - margin:
            x = room_length - margin - (dresser_length / 2)

    elif door_wall == 'top':
        # Flush with top wall
        # occupant faces down => occupant's right => x < door_pos
        y = room_width - margin - (dresser_width / 2)
        x = door_pos - (dresser_length / 2) - offset
        # clamp
        if x - (dresser_length / 2) < margin:
            x = margin + (dresser_length / 2)

    elif door_wall == 'left':
        # Flush with left wall
        # occupant faces right => occupant's right => y < door_pos
        x = margin + (dresser_width / 2)
        y = door_pos - (dresser_length / 2) - offset
        # clamp
        if y - (dresser_length / 2) < margin:
            y = margin + (dresser_length / 2)

    else:  # door_wall == 'right'
        # Flush with right wall
        # occupant faces left => occupant's right => y > door_pos
        x = room_length - margin - (dresser_width / 2)
        y = door_pos + (dresser_length / 2) + offset
        # clamp
        if y + (dresser_length / 2) > room_width - margin:
            y = room_width - margin - (dresser_length / 2)

    return (round(x, 2), round(y, 2))



def place_desk_near_window(room_length, room_width, window_wall, margin=margin, offset=desk_offset):
    """
    Places the desk flush against the unobstructed window wall.
    """
    desk_length, desk_width = furniture_dims['desk']
    if window_wall == 'top':
        x = room_length/2
        y = room_width - margin - desk_width/2 - offset
    elif window_wall == 'bottom':
        x = room_length/2
        y = margin + desk_width/2 + offset
    elif window_wall == 'left':
        x = margin + desk_width/2 + offset
        y = room_width/2
    else:  # right
        x = room_length - margin - desk_width/2 - offset
        y = room_width/2
    return (round(x,2), round(y,2))

def place_pillar_at_window(room_length, room_width, window_wall, margin=margin):
    """
    Places a pillar flush against the specified window wall.
    """
    pillar_width, pillar_depth = pillar_dims
    if window_wall == 'top':
        x = room_length/2
        y = room_width - margin - pillar_depth/2
    elif window_wall == 'bottom':
        x = room_length/2
        y = margin + pillar_depth/2
    elif window_wall == 'left':
        x = margin + pillar_width/2
        y = room_width/2
    else:  # right
        x = room_length - margin - pillar_width/2
        y = room_width/2
    return (round(x,2), round(y,2))