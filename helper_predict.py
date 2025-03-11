import torch
from model import MLP
import joblib
import pandas as pd

# Global parameters:
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

def predict_optimal_placements(obstacles_dict, stage, door_wall, window_wall, window1_wall, window2_wall, blocked_window_choice):


    # -------------------------------
    # Load saved scalers and model
    # -------------------------------
    scaler_X = joblib.load("scaler_X.joblib")
    scaler_y = joblib.load("scaler_y.joblib")

    input_features = [
        # "stage",
        "room_length", "room_width",
        # "door_exist",
        "door_wall_left", "door_wall_right", "door_wall_top", "door_wall_bottom",
        "door_pos",
        # "window_exist",
        "window1_wall_left", "window1_wall_right", "window1_wall_top", "window1_wall_bottom",
        "window2_wall_left", "window2_wall_right", "window2_wall_top", "window2_wall_bottom",
        "window1_pos", "window2_pos",
        "pillar_x", "pillar_y",
        "blocked_window_left", "blocked_window_right", "blocked_window_top", "blocked_window_bottom"
    ]
    output_features = [
        "bed_x", "bed_y",
        "dresser_x", "dresser_y",
        "nightstand_x", "nightstand_y",
        "table_x", "table_y",
        "desk_x", "desk_y"
    ]
    input_dim = len(input_features)
    output_dim = len(output_features)

    model = MLP(input_dim, output_dim)
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()

    input_df = pd.DataFrame([obstacles_dict])
    # Ensure the order matches the training list.
    ordered_inputs = input_df[input_features].values

    X_input_scaled = scaler_X.transform(ordered_inputs)
    x_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(x_tensor).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)[0]

    pred_outputs = dict(zip(output_features, pred))

    sample_full = obstacles_dict.copy()
    sample_full.update(pred_outputs)
    sample_full["door_wall"] = door_wall
    if stage == 1:
        sample_full["window_wall"] = window_wall
    else:
        sample_full["window_wall"] = window1_wall if blocked_window_choice=="Window 2" else window2_wall


    return sample_full, pred_outputs

