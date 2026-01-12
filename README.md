# highD-scenario-extractor
The HighD Scenario Extractor extracts traffic scenarios from the HighD dataset. It processes vehicle trajectories to provide initial states and spatial relationships (position, speed, distance). Output data is compatible with HighwayEnv, supporting autonomous driving research and testing.
# Prerequisites
Python 3.8+; Ensure 'pip' or 'conda' is available; 
pip install pandas numpy or conda install pandas numpy
# Usage Sequence
Run the scripts in the following order:<br>
'cut_in_trajectory_extractor.py' (Extracts vehicle trajectories performing cut-in maneuvers from driving datasets)<br>
'cut_in_trajectory_cleaner.py' (The filter clips the trajectory according to the speed range (Optional))<br>
'victim_trajectory_extractor.py' (Extracts trajectories of victim vehicles (vehicles being cut in on))<br>
'initial_cut_in_state.py' (Extract the initial state information of the vehicle at the start of a cut-in)<br>
'initial_victim_state.py' (Extract the initial state information of the victim at the start of a cut-in)<br>
'cut_in_scenario_initializer.py' (Initializes cut-in scenarios for highway-env)
