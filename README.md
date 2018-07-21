# Monte Carlo Particle Filter for Localization

 Particle Filter Algorithm is a nonparametric implementation of
 the Bayes Filter to approximate state, for example of a robot moving in a maze.
 
 The idea is to represent the posterior belief by a finite number of random variables (particles).
  The algorithm is repeatedly resampling those particles based on likelihood derived from a measurement model.

This project is a examination project for SS18 [Monte Carlo Methods in Artificial Intelligence and Machine Learning](https://www.ki.tu-berlin.de/menue/lehre/sommersemster_2018/) course taught by Prof. Dr. Manfred Opper and Theo Galy-Fajou, at TU Berlin.


## Setup
```
pip install -r requirements.txt
```

## Usage
```
> python particle-filter.py

Options
    - scene :  [scene-1, scene-2, scene-1-kidnapping, scene-2-kidnapping, scene-8.12]
    - no_particles : number of particles, for example 100.
    - total_frames :  total time step to run the simulation, it's useful for debugging.
    - show_particles : a boolean option whether to show the particles.
    - no_random_particles: number of random particles introduced to the system, required for kidnapping scenes.
    - save : a boolean option whether to see the simulartor live or save the result as a video.
    - frame_interval : time interval for each frame, default is 50s.
```

### Example
```
python particle-filter.py --scene scene-1 --no-particles 100 --save
```

![](https://i.imgur.com/RxP35Wa.png)
![](https://i.imgur.com/smYPY13.gif)


## Members
Anders Dahl Hjort, Luis Dreisbach, and Pattarawat Chormai


