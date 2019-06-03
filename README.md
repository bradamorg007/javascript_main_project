# javascript_main_project: Foundation Branch - THIS SHOULD CHANGED ALL CHANGES SHOULD OCCUR IN TEST BRANCH
My machine learning main project using javascript instead

The Foundation code represents a working topologically static neuroevolution project.

I have exponential Fitness;

Agents distance from the center of the gap on the y axis and their total distance traveled on the x axis both
constribute to fitness. using gap centering significantly inproves performance.

Softmax is now normalised so that the input values to this function dont explode the output. Can now use many layered nn.
Added Speed up and draw buttons,so the simulation can run at fast speeds and performance can be increased by not drawing the frames. 

# THINGS TO ADD FROM HERE

Need To be able to preDefine levels, so they have start and end points its not procedural.

Need to be able to define what kinds of absticles the agent will face. First i want the agent to be exposed to only a 
very limited amount of obsticles then as the levels change the difficulty increases. 

Program needs a block buffer, that preloads a level into the game, then the draw function reads the blocks into the
simulation as needed. they are not all loaded once. 

Add More Buttons and Text to th HTML page so that i can get more feedback.
Create an end of session stats function that generates plots and graphs and general statistics of model performance
for the given session, with an option to to save the graphs. 

Allow the Neural networks the ability to change there lift amount. nn takes lift amount in as input then outputs an appropriate lift amount after. This will allow them to adapt to when gaps get very small, they can reduce there jump amount.

Implement simulation where the agents control their x axis direction aswell. The camera will slide along when the one agent 
reaches the edge of the game space. 

Implement a simulation where agents be given their own environment spaces to train in. display could show a grid of many minature games all occuring. or the program can allow the user to select which agent environment to view, this might be better
because only one environment has to be drawn then which speed up performance. 

# Long Goals

Experiment with simulation environments that best capture the problem i am trying to tackle.

Implement neat to this problem as a good starting point for practise with neat. 
Implement novely search into this for practise. 

Implement autoencoder aspect of the project

# GPU Acceleration

Areas to apply GPU to:
  Functios need to avoid if statements and be independent. Indepenednt means that each elment can be computed on its own and     isnt depenendent on the results of other elements
  Matrix Operations
  Calculating the fitness function for each agent
  Calculating avg  distance from gap for each agent

# Notes General Ideas 

In my model when an agent needs to simulate an internal thought process. That testing can occur of screen in a seperate 
simulation instance. it doesnt have to be drawn and the autoencoder does not need to be applied
