<h1><center>CS 7642 Spring 2021 Project 2: Solving the Lunar Lander Environment with DDQN</center></h1>
<h2><center>Evan Jones</center></h2>
<h3><center>March 22, 2021</center></h3>

---
## Repository Directory Structure:

CSV_Data_Files: In this folder I have the various graph data and logged training data for each model. These data files contain records of episodic rewards, average 100 previous rewards, epsilon etc. For the entire work. Organized by hyper parameter when possible. Essiential files are provided directly whereas any data not directly graphed but stored resides in zip files.

tensorflow_scrap: all of the painstakingly slow code leftover from the tensorflow implementation pitfall. migrated to pytorch due to computation issues.

Tensorboard_Logs: this folder contains three zips files containing the rather extensive tensorboard log and event files generated and evaluated during training for each model.

python_Scripts: this folder should contain everything you need to run the code. The primary run file is titled DDQN.py referernces should be intact given that all dependencies are within that directory. This folder also included implementations for a sum tree part of an incomlete priority queue attempt, helper functions for tensorboard and clearing out dead PID's still recognized through jupyter as well as a standard replay buffer and epsilon scheduler. 

trained_model: contains the pickled pytorch models of all of the reports trained agents.

The last files on the root directory are the jupyter notebook i used to work on the models, the intial project breif, and report. 


Saved_Graphs: Holds each of the jpgs utilized within the report as well as the ipynb file used to render them. Must be run from the same directory as the data files to work properly with directory references.



