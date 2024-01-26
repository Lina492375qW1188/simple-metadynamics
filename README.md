# simple-metadynamics

This is a demonstration of a simple 1-dimensional metadynamics. It is built with some basic numpy to make the code cleaner and matplotlib for visualization.

## Structure

`DATA/` with the trajectories. 

`output/` with the output `*.gif`, which can be used in presentation.

`md.ipynb` shows the normal md result without metadynamics. This is for comparing later with metadynamics results. 

`metad_sigma.ipynb` is an example of demonstrating the performance of metadynamics under various Gaussian width. 

`swim.py` contains the main md and metadynamics code. 

