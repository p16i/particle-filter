#!/usr/bin/env bash

python main2.py --scene scene-1-kidnapping --no_particles 1000 --save
python main2.py --scene scene-1-kidnapping --no_particles 750 --save --no_random_particles 250
python main2.py --scene scene-2-kidnapping --no_particles 1000 --save
python main2.py --scene scene-2-kidnapping --no_particles 750 --save --no_random_particles 250
