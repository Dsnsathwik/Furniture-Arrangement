import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helper_placement import (
    place_furniture_fixed, 
    place_nightstand_adjacent, 
    place_table_near_door, 
    place_pillar_at_window, 
    place_nightstand_left, 
    place_desk_near_window, 
    place_dresser_right_of_door
)
import os

# -------------------------------
# ASSUMPTIONS:
# The following helper functions are assumed to be defined (or you can paste them above):
#   - place_furniture_fixed(room_length, room_width, furniture, wall, margin)
#   - place_nightstand_adjacent(bed_center, margin)
#   - place_table_near_door(room_length, room_width, door_wall, margin, offset)
#   - place_pillar_at_window(room_length, room_width, window_wall, margin)
#   - place_desk_near_window(room_length, room_width, window_wall, margin, offset)
#   - place_dresser_right_of_door(room_length, room_width, door_wall, door_pos, margin=margin, offset=dresser_door_offset)
#
# Also, global variables:
margin = 0.2
table_offset = 2
desk_offset = 0.3
dresser_door_offset = 0.7
pillar_dims = (0.8, 0.8)
furniture_dims = {
    'bed': (2.0, 1.5),
    'dresser': (1.0, 0.5),
    'nightstand': (0.5, 0.5),
    'table': (1.2, 0.8),
    'desk': (1.0, 0.5)
}

# -------------------------------
# GENERATOR FUNCTIONS
# -------------------------------
def generate_sample_stage1(room_length, room_width, door_wall):
    """
    Generates a Stage 1 sample.
    In Stage 1, only one window is used. For one window, we choose the window on the 'right' side of the door.
    Mapping for one window (example):
      - If door is 'bottom': window on 'right'
      - If door is 'top': window on 'left'
      - If door is 'left': window on 'top'
      - If door is 'right': window on 'bottom'
    """
    door_exist = 1
    # For door position, assume center of the wall:
    if door_wall in ['left', 'right']:
        door_pos = round(room_width/2, 2)
    else:
        door_pos = round(room_length/2, 2)
    door_wall_left   = 1 if door_wall == 'left' else 0
    door_wall_right  = 1 if door_wall == 'right' else 0
    door_wall_top    = 1 if door_wall == 'top' else 0
    door_wall_bottom = 1 if door_wall == 'bottom' else 0

    # For one window, choose the window based on door:
    if door_wall == 'bottom':
        window_wall = 'right'
    elif door_wall == 'top':
        window_wall = 'left'
    elif door_wall == 'left':
        window_wall = 'top'
    else:  # door_wall == 'right'
        window_wall = 'bottom'
    window_exist = 1
    # For window position, if wall is top/bottom, use room_length/2; else room_width/2.
    if window_wall in ['top','bottom']:
        window_pos = round(room_length/2, 2)
    else:
        window_pos = round(room_width/2, 2)
    window_wall_left   = 1 if window_wall == 'left' else 0
    window_wall_right  = 1 if window_wall == 'right' else 0
    window_wall_top    = 1 if window_wall == 'top' else 0
    window_wall_bottom = 1 if window_wall == 'bottom' else 0

    # Stage 1: second window, pillar, blocked windows, and desk are not used:
    window2_wall_left = window2_wall_right = window2_wall_top = window2_wall_bottom = 0
    window2_pos = 0
    pillar_x = pillar_y = 0
    blocked_window_left = blocked_window_right = blocked_window_top = blocked_window_bottom = 0
    desk_x = desk_y = 0

    # Furniture:
    # Bed: flush opposite door.
    if door_wall == 'left':
        bed_wall = 'right'
    elif door_wall == 'right':
        bed_wall = 'left'
    elif door_wall == 'top':
        bed_wall = 'bottom'
    else:
        bed_wall = 'top'
    bed_center = place_furniture_fixed(room_length, room_width, 'bed', bed_wall, margin)
    # Dresser: if door in left/right, dresser on top; else on left.
    if door_wall in ['left','right']:
        dresser_wall = 'top'
    else:
        dresser_wall = 'left'
    dresser_center = place_furniture_fixed(room_length, room_width, 'dresser', dresser_wall, margin)
    # Nightstand: always to the left of bed.
    nightstand_center = place_nightstand_adjacent(bed_center, margin)
    # Table: near door.
    table_center = place_table_near_door(room_length, room_width, door_wall, margin, table_offset)
    
    features = {
        'stage': 1,
        'room_length': room_length,
        'room_width': room_width,
        'door_exist': door_exist,
        'door_wall_left': door_wall_left,
        'door_wall_right': door_wall_right,
        'door_wall_top': door_wall_top,
        'door_wall_bottom': door_wall_bottom,
        'door_pos': door_pos,
        'window_exist': window_exist,
        'window1_wall_left': window_wall_left,
        'window1_wall_right': window_wall_right,
        'window1_wall_top': window_wall_top,
        'window1_wall_bottom': window_wall_bottom,
        'window2_wall_left': window2_wall_left,
        'window2_wall_right': window2_wall_right,
        'window2_wall_top': window2_wall_top,
        'window2_wall_bottom': window2_wall_bottom,
        'window1_pos': window_pos,
        'window2_pos': window2_pos,
        'pillar_x': pillar_x,
        'pillar_y': pillar_y,
        'blocked_window_left': blocked_window_left,
        'blocked_window_right': blocked_window_right,
        'blocked_window_top': blocked_window_top,
        'blocked_window_bottom': blocked_window_bottom,
        'bed_wall': bed_wall,
        'dresser_wall': dresser_wall,
        'door_wall': door_wall,
        'window_wall': window_wall,
        'chosen_window_for_desk': window_wall
    }
    outputs = {
        'bed_x': bed_center[0],
        'bed_y': bed_center[1],
        'dresser_x': dresser_center[0],
        'dresser_y': dresser_center[1],
        'nightstand_x': nightstand_center[0],
        'nightstand_y': nightstand_center[1],
        'table_x': table_center[0],
        'table_y': table_center[1],
        'desk_x': desk_x,
        'desk_y': desk_y
    }
    return {**features, **outputs}

def generate_sample_stage2(room_length, room_width, door_wall, blocked_window_choice):
    """
    Generates a Stage 2 sample.
    In Stage 2, two windows are used. The mapping is as follows:
      - If door is in ['left', 'right']: window1 = 'top', window2 = 'bottom'
      - Otherwise: window1 = 'left', window2 = 'right'
    The user chooses which window is blocked by the pillar (blocked_window_choice = "Window 1" or "Window 2").
    """
    door_exist = 1
    if door_wall in ['left', 'right']:
        door_pos = round(room_width/2, 2)
    else:
        door_pos = round(room_length/2, 2)
    door_wall_left   = 1 if door_wall=='left' else 0
    door_wall_right  = 1 if door_wall=='right' else 0
    door_wall_top    = 1 if door_wall=='top' else 0
    door_wall_bottom = 1 if door_wall=='bottom' else 0
    
    # Two windows: if door in left/right, then window1 = top, window2 = bottom; else window1 = left, window2 = right.
    if door_wall in ['left','right']:
        window1_wall = 'top'
        window2_wall = 'bottom'
    else:
        window1_wall = 'left'
        window2_wall = 'right'
    window_exist = 1
    # For window positions: if on top/bottom, use room_length/2; if on left/right, use room_width/2.
    if window1_wall in ['top','bottom']:
        window1_pos = round(room_length/2, 2)
    else:
        window1_pos = round(room_width/2, 2)
    if window2_wall in ['top','bottom']:
        window2_pos = round(room_length/2, 2)
    else:
        window2_pos = round(room_width/2, 2)
    window1_wall_left   = 1 if window1_wall=='left' else 0
    window1_wall_right  = 1 if window1_wall=='right' else 0
    window1_wall_top    = 1 if window1_wall=='top' else 0
    window1_wall_bottom = 1 if window1_wall=='bottom' else 0
    window2_wall_left   = 1 if window2_wall=='left' else 0
    window2_wall_right  = 1 if window2_wall=='right' else 0
    window2_wall_top    = 1 if window2_wall=='top' else 0
    window2_wall_bottom = 1 if window2_wall=='bottom' else 0
    
    # Pillar: block one of the windows based on user choice.
    if blocked_window_choice == "Window 1":
        blocked_window = window1_wall
    else:
        blocked_window = window2_wall
    pillar_center = place_pillar_at_window(room_length, room_width, blocked_window, margin)
    
    # Desk: near the unobstructed window.
    chosen_window = window2_wall if blocked_window == window1_wall else window1_wall
    desk_center = place_desk_near_window(room_length, room_width, chosen_window, margin, offset=desk_offset)
    
    # Dresser: placed on the interior side of the door.
    dresser_center = place_dresser_right_of_door(room_length, room_width, door_wall, door_pos)
    dresser_wall = door_wall
    
    # Bed: flush opposite the door.
    if door_wall == 'left':
        bed_wall = 'right'
    elif door_wall == 'right':
        bed_wall = 'left'
    elif door_wall == 'top':
        bed_wall = 'bottom'
    else:
        bed_wall = 'top'
    bed_center = place_furniture_fixed(room_length, room_width, 'bed', bed_wall, margin)
    
    # Nightstand: on physical left side of bed.
    nightstand_center = place_nightstand_left(bed_center, bed_wall, margin)
    
    # Table: near door.
    table_center = place_table_near_door(room_length, room_width, door_wall, margin, table_offset)
    
    features = {
        'stage': 2,
        'room_length': room_length,
        'room_width': room_width,
        'door_exist': door_exist,
        'door_wall_left': door_wall_left,
        'door_wall_right': door_wall_right,
        'door_wall_top': door_wall_top,
        'door_wall_bottom': door_wall_bottom,
        'door_pos': door_pos,
        'window_exist': window_exist,
        'window1_wall_left': window1_wall_left,
        'window1_wall_right': window1_wall_right,
        'window1_wall_top': window1_wall_top,
        'window1_wall_bottom': window1_wall_bottom,
        'window2_wall_left': window2_wall_left,
        'window2_wall_right': window2_wall_right,
        'window2_wall_top': window2_wall_top,
        'window2_wall_bottom': window2_wall_bottom,
        'window1_pos': window1_pos,
        'window2_pos': window2_pos,
        'pillar_x': pillar_center[0],
        'pillar_y': pillar_center[1],
        'blocked_window_left': 1 if blocked_window=='left' else 0,
        'blocked_window_right': 1 if blocked_window=='right' else 0,
        'blocked_window_top': 1 if blocked_window=='top' else 0,
        'blocked_window_bottom': 1 if blocked_window=='bottom' else 0,
        'bed_wall': bed_wall,
        'dresser_wall': dresser_wall,  # set to the interior side (for uniformity)
        'door_wall': door_wall,
        'window_wall': chosen_window,
        'chosen_window_for_desk': chosen_window
    }
    outputs = {
        'bed_x': bed_center[0],
        'bed_y': bed_center[1],
        'dresser_x': dresser_center[0],
        'dresser_y': dresser_center[1],
        'nightstand_x': nightstand_center[0],
        'nightstand_y': nightstand_center[1],
        'table_x': table_center[0],
        'table_y': table_center[1],
        'desk_x': desk_center[0],
        'desk_y': desk_center[1]
    }
    return {**features, **outputs}
