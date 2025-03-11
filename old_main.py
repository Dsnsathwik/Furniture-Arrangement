import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from helper_data import generate_sample_stage1, generate_sample_stage2
from helper_visualize import visualize_layout, visualize_layout_stage2

# Assume helper functions and global variables (margin, table_offset, etc.) are defined above:
#   - place_furniture_fixed, place_nightstand_adjacent, place_table_near_door, 
#   - place_pillar_at_window, place_desk_near_window, place_dresser_right_of_door
#   - generate_sample_stage1, generate_sample_stage2 (as defined above)
#   - visualize_layout (for Stage 1), visualize_layout_stage2 (for Stage 2)
# Also assume your trained model is loaded (if prediction is desired).

st.title("Room Layout Generator & Visualizer")

# Sidebar inputs
st.sidebar.header("Room Parameters")
room_length = st.sidebar.number_input("Room Length (m)", min_value=6.0, max_value=7.0, value=6.5, step=0.1)
room_width  = st.sidebar.number_input("Room Width (m)", min_value=6.0, max_value=7.0, value=6.5, step=0.1)
door_wall   = st.sidebar.selectbox("Select Door Wall", options=['left', 'right', 'top', 'bottom'])
num_windows = st.sidebar.selectbox("Number of Windows", options=["1", "2"])

blocked_window_choice = None
if num_windows == "2":
    blocked_window_choice = st.sidebar.selectbox("Which window is blocked by the pillar?", options=["Window 1", "Window 2"])

# Button to generate sample
if st.sidebar.button("Generate Sample"):
    if num_windows == "1":
        sample = generate_sample_stage1(room_length, room_width, door_wall)
    else:
        sample = generate_sample_stage2(room_length, room_width, door_wall, blocked_window_choice)
    
    st.write("Generated Sample Parameters:")
    st.json(sample)
    
    # Visualize the generated layout.
    # We capture the matplotlib figure from the visualization function.
    # Here, we assume the visualization functions return a matplotlib figure.
    # If they only save to file, you can modify them to also return the figure.
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    
    # We re-use the visualization code inline here (or call your function if modified to return fig)
    # For Stage 1:
    if sample['stage'] == 1:
        # Call your visualize_layout function that draws on current ax:
        # Example: visualize_layout(sample, ax)
        # (You may need to modify your function to accept an axis and return the figure.)
        # For simplicity, I'll include inline code:
        room_rect = patches.Rectangle((0, 0), sample['room_length'], sample['room_width'],
                                      fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(room_rect)
        # Draw door, window, furniture, etc.
        # (For brevity, assume we call a function)
        visualize_layout(sample, name="temp_stage1", save_path="temp/")  # This saves file; you might want to also show it.
        st.write("Stage 1 layout saved in temp folder.")
    else:
        visualize_layout_stage2(sample, name="temp_stage2", save_path="temp/")
        st.write("Stage 2 layout saved in temp folder.")
    
    st.image(f"temp/{'temp_stage1.png' if sample['stage']==1 else 'temp_stage2.png'}", caption="Generated Layout")
    
    # Optionally, you can then use this sample for prediction using your model.
    # (Extract numeric input features from the sample dictionary and run the model.)
