#!/usr/bin/env bash

python main2.py --scene scene-1  --no_particles 10 --save
python main2.py --scene scene-1  --no_particles 100 --save
python main2.py --scene scene-1  --no_particles 1000 --save

python main2.py --scene scene-2  --no_particles 10 --save
python main2.py --scene scene-2 --no_particles 100 --save
python main2.py --scene scene-2 --no_particles 1000 --save
