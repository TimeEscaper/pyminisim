# PyMiniSim
2D simulator for mobile robot and pedestrians simulation.

---

### Current status
Current status is *version zero proof-of-concept*. Hope that we will continue active development and maintenance of the project.

### Use cases
* High-level applications and algorithms that require NPC behavious simulation much more than accurate physics simulation (e.g. planning, social navigation)
* Different educational purposes

### Roadmap
* Documentation
* Support of the maps and obstacles (now only plain environments without obstacles are supported)
* More sensors simulation (e.g. range sensor)
* Sensors and motion noise modeling support
* More pedestrians behaviour models (e.g. Extended Social Force Model, Inverse Reinforcement  Learning based approaches)
* Keyboard control for the robot

---

## Architecture overview

Note: current diagram display not only current architecture, but also concepts for the future updates (see roadmap).

<img src="./.img/pyminisim_architecture.png" alt="PyMiniSim architecture">

---

## Examples

See [examples/](./examples) directory. Available examples:
* [examples/basic.py](examples/basic.py): Basic example of simulation with Headed Social Force model and simple pedestrian detector
