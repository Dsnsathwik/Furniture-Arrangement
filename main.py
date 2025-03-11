import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import joblib
from helper_predict import predict_optimal_placements
from helper_visualize import visualize_layout, visualize_layout_stage2
from helper_placement import place_pillar_at_window

# -------------------------------
# Global Parameters and Assumptions
# -------------------------------
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
# Streamlit Interface for Prediction
# -------------------------------
st.title("Room Layout Prediction Interface")

st.sidebar.header("Obstacle Parameters (Input)")

# Room dimensions:
room_length = st.sidebar.number_input("Room Length (m)", min_value=6.0, max_value=7.0, value=6.5, step=0.1)
room_width  = st.sidebar.number_input("Room Width (m)", min_value=6.0, max_value=7.0, value=6.5, step=0.1)

# Door:
door_exist = 1  # Always assume door exists
door_wall = st.sidebar.selectbox("Select Door Wall", options=['left', 'right', 'top', 'bottom'], index=0)
if door_wall in ['left', 'right']:
    door_pos = room_width / 2
else:
    door_pos = room_length / 2

# Number of windows (this determines the stage)
num_windows = st.sidebar.selectbox("Number of Windows", options=["1", "2"], index=0)
if num_windows == "1":
    stage = 1
    # For Stage 1, rule: if door is bottom -> window on right; top->left; left->top; right->bottom.
    if door_wall == 'bottom':
        window_wall = 'right'
    elif door_wall == 'top':
        window_wall = 'left'
    elif door_wall == 'left':
        window_wall = 'top'
    else:  # door_wall == 'right'
        window_wall = 'bottom'
    window_exist = 1
    if window_wall in ['top', 'bottom']:
        window1_pos = room_length / 2
    else:
        window1_pos = room_width / 2
    window2_pos = 0
    window1_wall_left   = 1 if window_wall == 'left' else 0
    window1_wall_right  = 1 if window_wall == 'right' else 0
    window1_wall_top    = 1 if window_wall == 'top' else 0
    window1_wall_bottom = 1 if window_wall == 'bottom' else 0
    # For Stage 1, second window fields are 0.
    window2_wall_left = window2_wall_right = window2_wall_top = window2_wall_bottom = 0
    # Pillar and blocked windows set to 0.
    pillar_x = pillar_y = 0
    blocked_window_left = blocked_window_right = blocked_window_top = blocked_window_bottom = 0
else:
    stage = 2
    window_exist = 1
    # For Stage 2, rule: if door is in left/right then window1 = top, window2 = bottom; else window1 = left, window2 = right.
    if door_wall in ['left', 'right']:
        window1_wall = 'top'
        window2_wall = 'bottom'
    else:
        window1_wall = 'left'
        window2_wall = 'right'
    if window1_wall in ['top', 'bottom']:
        window1_pos = room_length / 2
    else:
        window1_pos = room_width / 2
    if window2_wall in ['top', 'bottom']:
        window2_pos = room_length / 2
    else:
        window2_pos = room_width / 2
    window1_wall_left   = 1 if window1_wall=='left' else 0
    window1_wall_right  = 1 if window1_wall=='right' else 0
    window1_wall_top    = 1 if window1_wall=='top' else 0
    window1_wall_bottom = 1 if window1_wall=='bottom' else 0
    window2_wall_left   = 1 if window2_wall=='left' else 0
    window2_wall_right  = 1 if window2_wall=='right' else 0
    window2_wall_top    = 1 if window2_wall=='top' else 0
    window2_wall_bottom = 1 if window2_wall=='bottom' else 0

    # Pillar: since pillar is an obstacle, place it based on user's choice.
    blocked_window_choice = st.sidebar.selectbox("Which window is blocked by a pillar?", options=["Window 1", "Window 2"])
    if blocked_window_choice == "Window 1":
        blocked_wall = window1_wall
    else:
        blocked_wall = window2_wall
    # Now, compute the pillar position using our helper function.
    pillar_x, pillar_y = place_pillar_at_window(room_length, room_width, blocked_wall, margin)
    blocked_window_left   = 1 if blocked_wall=='left' else 0
    blocked_window_right  = 1 if blocked_wall=='right' else 0
    blocked_window_top    = 1 if blocked_wall=='top' else 0
    blocked_window_bottom = 1 if blocked_wall=='bottom' else 0


# -------------------------------
# Build the Input Feature Vector (Numeric Features)
# -------------------------------
training_input = {
    # "stage": stage,
    "room_length": room_length,
    "room_width": room_width,
    "door_exist": door_exist,
    "door_wall_left": 1 if door_wall=="left" else 0,
    "door_wall_right": 1 if door_wall=="right" else 0,
    "door_wall_top": 1 if door_wall=="top" else 0,
    "door_wall_bottom": 1 if door_wall=="bottom" else 0,
    "door_pos": door_pos,
    "window_exist": window_exist,
    "window1_wall_left": window1_wall_left,
    "window1_wall_right": window1_wall_right,
    "window1_wall_top": window1_wall_top,
    "window1_wall_bottom": window1_wall_bottom,
    "window2_wall_left": window2_wall_left if stage==2 else 0,
    "window2_wall_right": window2_wall_right if stage==2 else 0,
    "window2_wall_top": window2_wall_top if stage==2 else 0,
    "window2_wall_bottom": window2_wall_bottom if stage==2 else 0,
    "window1_pos": window1_pos,
    "window2_pos": window2_pos if stage==2 else 0,
    "pillar_x": pillar_x,
    "pillar_y": pillar_y,
    "blocked_window_left": blocked_window_left,
    "blocked_window_right": blocked_window_right,
    "blocked_window_top": blocked_window_top,
    "blocked_window_bottom": blocked_window_bottom
}

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("Predict Layout"):
    # This function from helper_predict should take the training_input and additional parameters (door_wall, window_wall, etc.)
    # and return a full sample dictionary (with predicted furniture positions) with all keys.
    # Note: We need to provide window_wall parameters for visualization.
    if stage == 1:
        # For Stage 1, our window_wall is the single window's wall.
        sample_full, pred_outputs = predict_optimal_placements(training_input, stage, door_wall, window_wall, window1_wall=None, window2_wall=None, blocked_window_choice=None)
    else:
        sample_full, pred_outputs = predict_optimal_placements(training_input, stage, door_wall, window_wall=None, window1_wall=window1_wall, window2_wall=window2_wall, blocked_window_choice=blocked_window_choice)
    
    st.write("Predicted Sample:")
    st.json(pred_outputs)
    
    # Save predicted sample to CSV.
    sample_df = pd.DataFrame([sample_full])
    sample_df.to_csv("predicted_sample.csv", index=False)
    
    # Visualize the predicted layout.
    if stage == 1:
        st.markdown("""
        **Explanation:**
        
        - The room is modeled with a single window.
        - The door and window act as fixed obstacles, ensuring furniture is not placed in front of them.
        - The bed is placed flush opposite the door, keeping the entry clear.
        - The dresser is positioned to remain fully accessible, and the nightstand is always to the left of the bed.
        - These placements guarantee that each piece serves its function without interference.
        - The model learns these patterns from the synthetic data.
        """)
        visualize_layout(sample_full, name="predicted_sample", save_path="visualizations_stage1/")
        st.image("visualizations_stage1/predicted_sample.png", caption="Predicted Room Layout")
    else:
        st.markdown("""
        **Explanation:**
        
        - The room features two windows on adjacent walls (excluding the door wall).
        - The door and windows are immovable obstacles; furniture is never placed directly in front of them.
        - The user selects which window is blocked by a pillar. To maximize natural light and fresh air, the desk is placed in front of the unobstructed window.
        - The layout is further optimized so the bed isn’t placed in front of the dresser, ensuring functional accessibility.
        - The model learns these patterns from the synthetic data.
        """)
        visualize_layout_stage2(sample_full, name="predicted_sample", save_path="visualizations_stage2/")
        st.image("visualizations_stage2/predicted_sample.png", caption="Predicted Room Layout")
