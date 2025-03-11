# Room Layout Prediction Application

This application is an interactive interface for predicting room layouts based on obstacle parameters using a pre-trained model. It uses Streamlit to provide a user-friendly UI where you can specify room dimensions, door and window configurations, and (in Stage 2) choose which window is blocked by a pillar. Based on these inputs, the model predicts optimal placements for furniture (bed, dresser, nightstand, table, and desk), and the application visualizes the resulting layout.

## Features

- **Interactive UI:**  
  Users can select room dimensions, choose a door wall, and specify the number of windows.
  
- **Stage-based Layouts:**  
  - **Stage 1:** A single-window configuration.  
  - **Stage 2:** A two-window configuration with an option to choose which window is blocked by a pillar.
  
- **Prediction & Visualization:**  
  The model predicts furniture placements from the obstacle parameters, and the app visualizes the resulting layout.
  
- **Consistent & Deterministic:**  
  The app uses obstacle data only for prediction, ensuring that furniture placements are based on learned patterns from synthetic data while preserving functionality (e.g., the bed is never placed in front of the dresser).

## Prerequisites

- Python 3.7 or higher
- Required Python packages (see `requirements.txt`):
  - streamlit
  - pandas
  - numpy
  - torch (PyTorch)
  - matplotlib
  - joblib
  - tqdm

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `app.py`: Main Streamlit application file.
- `helper_predict.py`: Contains the prediction function `predict_optimal_placements()`.
- `helper_visualize.py`: Contains visualization functions (`visualize_layout` for Stage 1 and `visualize_layout_stage2` for Stage 2).
- `helper_placement.py`: Contains helper functions for placement (e.g., `place_pillar_at_window()`, `place_furniture_fixed()`, etc.).
- `scaler_X.joblib`, `scaler_y.joblib`: Pre-fitted scalers used for input and output normalization.
- `best_model_stage2.pth`: The saved model checkpoint.
- Generated files:  
  - `predicted_sample.csv`: The sample with predicted furniture placements.
  - Visualization images saved under `visualizations_stage1/` and `visualizations_stage2/`.

## How It Works

1. **User Input:**  
   - The user sets the room dimensions and door configuration.
   - The user selects the number of windows; if two windows are selected (Stage 2), the UI also allows choosing which window is blocked by a pillar.
   - These obstacle parameters form the numeric input vector (in the same order as used during training).

2. **Prediction:**  
   - The input vector is normalized using the pre-fitted scaler.
   - The pre-trained model (an MLP) predicts the furniture positions (bed, dresser, nightstand, table, and desk).
   - The predictions are inverse-transformed back to the original scale.

3. **Visualization:**  
   - The predicted furniture coordinates are merged with the obstacle parameters.
   - The app then calls the appropriate visualization function (Stage 1 or Stage 2) to generate an image of the room layout.
   - The visualization shows obstacles (door, window, pillar) and furniture, ensuring that each item is placed functionally (e.g., the bed is not blocking the dresser, and in Stage 2 the desk is placed in front of the unobstructed window).

## Running the Application

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Ensure All Required Files Are Present:**  
   Make sure that `scaler_X.joblib`, `scaler_y.joblib`, and `best_model_stage2.pth` are in the project directory, along with the helper modules (`helper_predict.py`, `helper_visualize.py`, `helper_placement.py`).

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App:**

   ```bash
   streamlit run main.py
   ```

5. **Use the Interface:**  
   - Set the room dimensions, door wall, and number of windows from the sidebar.
   - For a two-window configuration, select which window is blocked by a pillar.
   - Click the **"Predict Layout"** button.
   - The app will display a description of how the model places furniture based on the obstacle configuration and then show the predicted room layout image.

## How Furniture Placement is Determined

- **Obstacles as Constraints:**  
  The door and windows are treated as fixed obstacles. Furniture is never placed directly in front of these obstacles to keep entryways clear and ensure proper natural light.

- **Functional Layout:**  
  - The bed is always placed flush opposite the door to maintain a clear pathway.  
  - The dresser is positioned so that it is easily accessible (and never blocked by the bed).  
  - The nightstand is always placed to the left of the bed, ensuring balanced access.  
  - The table is placed closer to the door, maintaining convenience and space.  
  - In Stage 2, if one window is blocked by a pillar, the desk is placed in front of the unobstructed window to maximize light and airflow.

- **Model Learning:**  
  The model is trained on synthetic data generated with these rules. It learns to map obstacle parameters to optimal furniture placements, ensuring that the resulting layout is both functional and aesthetically pleasing.

## Future Enhancements

- Improve the visualization interactivity (e.g., dynamic zoom or clickable layout elements).
- Incorporate more realistic (noisy) data to simulate real-world scenarios.
- Allow users to adjust additional parameters, such as furniture sizes or positions, in real time.

## License

This project is licensed under the MIT License.
