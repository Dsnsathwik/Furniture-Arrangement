import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import os

# Global parameters:
furniture_dims = {
    'bed': (2.0, 1.5), 
    'dresser': (1.0, 0.5),
    'nightstand': (0.5, 0.5),
    'table': (1.2, 0.8),
    'desk': (1.0, 0.5)  
}
pillar_dims = (0.8, 0.8)     
margin = 0.2                 
table_offset = 2             
desk_offset = 0.3            
dresser_door_offset = 0.7    

# -------------------------------
# Helper Functions
# -------------------------------
def place_furniture_fixed(room_length, room_width, furniture, wall, margin=margin):
    
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
    
    bed_length, bed_width = furniture_dims['bed']
    ns_length, ns_width = furniture_dims['nightstand']
    x, y = bed_center
    
    x_new = x - (bed_width / 2 + ns_width / 2 + margin)
    return (round(x_new, 2), round(y, 2))

def place_table_near_door(room_length, room_width, door_wall, margin=margin, offset=table_offset):
    
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
   
    dresser_length, dresser_width = furniture_dims['dresser']

    if door_wall == 'bottom':
        y = margin + dresser_width / 2
        x = door_pos + (dresser_length / 2) + offset
        
        if x + (dresser_length / 2) > room_length - margin:
            x = room_length - margin - (dresser_length / 2)

    elif door_wall == 'top':
        y = room_width - margin - (dresser_width / 2)
        x = door_pos - (dresser_length / 2) - offset
        
        if x - (dresser_length / 2) < margin:
            x = margin + (dresser_length / 2)

    elif door_wall == 'left':
        x = margin + (dresser_width / 2)
        y = door_pos - (dresser_length / 2) - offset
        
        if y - (dresser_length / 2) < margin:
            y = margin + (dresser_length / 2)

    else:  # door_wall == 'right'
        x = room_length - margin - (dresser_width / 2)
        y = door_pos + (dresser_length / 2) + offset
        
        if y + (dresser_length / 2) > room_width - margin:
            y = room_width - margin - (dresser_length / 2)

    return (round(x, 2), round(y, 2))



def place_desk_near_window(room_length, room_width, window_wall, margin=margin, offset=desk_offset):
    
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