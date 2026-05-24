# Dynamic_Programming
Drone Rescue MDP with Dynamic Programming

# Problem Statement:
Design and implement a reinforcement learning agent using dynamic programming - Value Iteration or 
Policy Iteration to compute an optimal policy for an autonomous rescue drone operating in a disaster zone. 
The drone must learn how to rescue stranded civilians while managing battery power, avoiding dangerous 
zones, and reaching charging stations when necessary. The problem must be modelled as a finite Markov 
Decision Process (MDP). The register number of the first student in a group (alphabetically sorted) will 
determine the environment configuration. The student will: 
• Implement a custom Drone Rescue environment.  
• Use dynamic programming to compute the optimal value function and policy.  
• Analyse how state design and reward design affect convergence and rescue efficiency.

# Scenario 
A rescue drone is deployed in a disaster-hit city after an earthquake. The city is represented as a grid 
world. The drone must: 
• rescue stranded civilians,  
• avoid dangerous areas,  
• manage battery usage,  
• and reach charging stations before battery depletion.  
The environment contains: 
• Safe zones  
• Danger zones  
• Charging stations  
• Rescue targets  
• Wind zones that disturb movement  
The drone begins with limited battery power and must maximize total rescue reward.
