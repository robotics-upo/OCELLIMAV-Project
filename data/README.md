
Processed data (synthetic and real) recorded to train the CNNBiGRU network. In simulation we considered two main scenarios: inside and outside a building, changing illumination conditions and number of light sources. About real data, we recorded data in indoor and outdoor scenarios aswell: an office, a building hall, under trees, a porch, etc., modifying light conditions.

![Datasets](../.github/datasets.png)

# Synthetic data.

131 datasets, organized in `simulation_data` folder as follow (in order):

- 35 sets in outdoor scenario with fixed light source (the Sun) direction.
- 11 sets in outdoor scenario changing light source direction.
- 30 sets in indoor scenario with 11 simultaneous light sources.
- 20 sets in indoor scenario with 6 light sources. 
- 35 sets in indoor scenario with windows (a porch), changing the external light direction.

# Real data.

24 datasets, in `real_data` folder as follow:

- 11 sets in purely outdoor scenario.
- 1 set in a porch.
- 8 sets in indoor scenario.
- 4 sets in indoor scenario only with pure translations.






In `processed_data` folder you will find two folders with datasets processed in order to train the network, aswell as information about data processing. 
