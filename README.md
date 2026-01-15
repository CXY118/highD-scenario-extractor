# highD-scenario-extractor
The HighD Scenario Extractor extracts traffic scenarios from the HighD dataset. It processes vehicle trajectories to provide initial states and spatial relationships (position, speed, distance). Output data is compatible with HighwayEnv, supporting autonomous driving research and testing.
## Prerequisites
Python 3.8+; Ensure 'pip' or 'conda' is available; <br>
pip install pandas numpy or conda install pandas numpy
## Usage Sequence
### Cut-in
Run the scripts in the following order:<br>
'cut_in_trajectory_extractor.py' (Extracts vehicle trajectories performing cut-in maneuvers from driving datasets)<br>
'cut_in_trajectory_cleaner.py' (The filter clips the trajectory according to the speed range (Optional))<br>
'victim_trajectory_extractor.py' (Extracts trajectories of victim vehicles (vehicles being cut in on))<br>
'initial_cut_in_state.py' (Extract the initial state information of the vehicle at the start of a cut-in)<br>
'initial_victim_state.py' (Extract the initial state information of the victim at the start of a cut-in)<br>
'cut_in_scenario_initializer.py' (Initializes cut-in scenarios for highway-env)<br>
Cut-in scenario diagram<br>
<img width="1069" height="270" alt="切入场景示意图" src="https://github.com/user-attachments/assets/6bb3a993-aa82-4135-ae12-4fcde4c83e16" />

### Lane-change
Run the scripts in the following order:<br>
'lane_change_trajectory_extractor.py' (Extracts lane change scenarios from driving datasets)<br>
'lane_change_trajectory_cleaner.py' (Filters lane_change trajectories by velocity range (Optional))<br>
'surrounding_vehicles_trajectories_extractor.py' (Extracts trajectories of surrounding vehicles in lane change scenarios)<br>
'lane_change_scenario_initializer.py' (Extract the initial state information of each vehicle at the beginning of lane change)<br>
'lane_change_initial_information_integrator.py' (HighwayEnv vehicle initial state alignment)<br>
Lane-change scenario diagram<br>
<img width="1069" height="270" alt="换道场景示意图" src="https://github.com/user-attachments/assets/693e34a8-0f99-45e2-a21e-05141f3c7705" />
