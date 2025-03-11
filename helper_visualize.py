import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#global variables
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



def visualize_layout(sample, name, save_path='stage1_visuals/'):
    """
    Visualizes the room layout
    """
    room_length = sample['room_length']
    room_width = sample['room_width']
    fig, ax = plt.subplots(figsize=(8, 8))
    
    room_rect = patches.Rectangle((0, 0), room_length, room_width, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(room_rect)
    
    thickness = 0.1
    
    if sample['door_exist'] == 1:
        door_wall = sample['door_wall']
        door_pos = sample['door_pos']
        if door_wall == 'left':
            door_rect = patches.Rectangle((-thickness, door_pos - 0.8/2), thickness, 0.8, color='brown', alpha=0.7)
        elif door_wall == 'right':
            door_rect = patches.Rectangle((room_length, door_pos - 0.8/2), thickness, 0.8, color='brown', alpha=0.7)
        elif door_wall == 'top':
            door_rect = patches.Rectangle((door_pos - 0.8/2, room_width), 0.8, thickness, color='brown', alpha=0.7)
        elif door_wall == 'bottom':
            door_rect = patches.Rectangle((door_pos - 0.8/2, -thickness), 0.8, thickness, color='brown', alpha=0.7)
        ax.add_patch(door_rect)
    
    if sample['window_exist'] == 1:
        window_wall = sample['window_wall']
        if sample.get('stage', 2) == 1:
            window_pos = sample.get('window1_pos', room_length/2)
        else:
            window_pos = sample.get('window_pos', room_length/2)
        if window_wall == 'left':
            window_rect = patches.Rectangle((-thickness, window_pos - 1.0/2), thickness, 1.0, color='blue', alpha=0.7)
        elif window_wall == 'right':
            window_rect = patches.Rectangle((room_length, window_pos - 1.0/2), thickness, 1.0, color='blue', alpha=0.7)
        elif window_wall == 'top':
            window_rect = patches.Rectangle((window_pos - 1.0/2, room_width), 1.0, thickness, color='blue', alpha=0.7)
        elif window_wall == 'bottom':
            window_rect = patches.Rectangle((window_pos - 1.0/2, -thickness), 1.0, thickness, color='blue', alpha=0.7)
        ax.add_patch(window_rect)
    
    def draw_furniture(center, furniture, color='green'):
        length, width = furniture_dims[furniture]
        x, y = center
        bottom_left = (x - width/2, y - length/2)
        rect = patches.Rectangle(bottom_left, width, length, edgecolor=color, facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, furniture, color=color, ha='center', va='center', fontsize=8)
    
    draw_furniture((sample['bed_x'], sample['bed_y']), 'bed', 'green')
    draw_furniture((sample['dresser_x'], sample['dresser_y']), 'dresser', 'purple')
    draw_furniture((sample['nightstand_x'], sample['nightstand_y']), 'nightstand', 'orange')
    draw_furniture((sample['table_x'], sample['table_y']), 'table', 'red')
    
    ax.set_xlim(-1, room_length + 1)
    ax.set_ylim(-1, room_width + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Predicted Room Layout")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}{name}.png")
    plt.close()


# -------------------------------
# Visualization Function for Stage 2
# -------------------------------
def visualize_layout_stage2(sample, name, save_path='stage2_visuals/'):
    """
    Visualizes the Stage 2 room layout
    """
    room_length = sample['room_length']
    room_width  = sample['room_width']
    fig, ax = plt.subplots(figsize=(8,8))
    
    room_rect = patches.Rectangle((0,0), room_length, room_width, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(room_rect)
    
    thickness = 0.1
    
    if sample['door_exist'] == 1:
        door_wall = sample['door_wall']
        door_pos = sample['door_pos']
        if door_wall == 'left':
            door_rect = patches.Rectangle((-thickness, door_pos - 0.8/2), thickness, 0.8, color='brown', alpha=0.7)
        elif door_wall == 'right':
            door_rect = patches.Rectangle((room_length, door_pos - 0.8/2), thickness, 0.8, color='brown', alpha=0.7)
        elif door_wall == 'top':
            door_rect = patches.Rectangle((door_pos - 0.8/2, room_width), 0.8, thickness, color='brown', alpha=0.7)
        else:
            door_rect = patches.Rectangle((door_pos - 0.8/2, -thickness), 0.8, thickness, color='brown', alpha=0.7)
        ax.add_patch(door_rect)
    
    def draw_window(center, wall):
        if wall in ['top','bottom']:
            w = 1.0; h = thickness
        else:
            w = thickness; h = 1.0
        bottom_left = (center[0]-w/2, center[1]-h/2)
        rect = patches.Rectangle(bottom_left, w, h, color='blue', alpha=0.7)
        ax.add_patch(rect)
    
    if sample['window1_wall_top']:
        w1_center = (room_length/2, room_width - margin)
        w1_wall = 'top'
    elif sample['window1_wall_bottom']:
        w1_center = (room_length/2, margin)
        w1_wall = 'bottom'
    elif sample['window1_wall_left']:
        w1_center = (margin, room_width/2)
        w1_wall = 'left'
    else:
        w1_center = (room_length - margin, room_width/2)
        w1_wall = 'right'
    draw_window(w1_center, w1_wall)
    
    if sample['window2_wall_top']:
        w2_center = (room_length/2, room_width - margin)
        w2_wall = 'top'
    elif sample['window2_wall_bottom']:
        w2_center = (room_length/2, margin)
        w2_wall = 'bottom'
    elif sample['window2_wall_left']:
        w2_center = (margin, room_width/2)
        w2_wall = 'left'
    else:
        w2_center = (room_length - margin, room_width/2)
        w2_wall = 'right'
    draw_window(w2_center, w2_wall)
    
    pillar_center = (sample['pillar_x'], sample['pillar_y'])
    p_w, p_h = pillar_dims
    p_bl = (pillar_center[0]-p_w/2, pillar_center[1]-p_h/2)
    pillar_rect = patches.Rectangle(p_bl, p_w, p_h, color='grey', alpha=0.8)
    ax.add_patch(pillar_rect)
    
    dims = furniture_dims.copy()
    def draw_furniture(center, furniture, color):
        length, width = dims[furniture]
        cx, cy = center
        bottom_left = (cx - width/2, cy - length/2)
        rect = patches.Rectangle(bottom_left, width, length, edgecolor=color, facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(cx, cy, furniture, color=color, ha='center', va='center', fontsize=8)
    
    draw_furniture((sample['bed_x'], sample['bed_y']), 'bed', 'green')
    draw_furniture((sample['dresser_x'], sample['dresser_y']), 'dresser', 'purple')
    draw_furniture((sample['nightstand_x'], sample['nightstand_y']), 'nightstand', 'orange')
    draw_furniture((sample['table_x'], sample['table_y']), 'table', 'red')
    draw_furniture((sample['desk_x'], sample['desk_y']), 'desk', 'magenta')
    
    ax.set_xlim(-1, room_length+1)
    ax.set_ylim(-1, room_width+1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Predicted Room Layout")
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}{name}.png")
    plt.close()