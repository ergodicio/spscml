# Dynamical Low-Rank Project for the Structure-Preserving Scientific Computing and Machine Learning summer school

**People**:
- Jingwei Hu, project organizer (hujw@uw.edu)
- Jack Coughlin, project lead ([johnbcoughlin.com], jack.coughlin@simulation.science)
- Howard Cheng, subject matter expert (yhcheng8@uw.edu)

## Getting set up

This repository uses `uv` to manage python dependencies. Install it from here: https://github.com/astral-sh/uv


## TODO list

@ Howard--let's do all the work in this repository, including writing up the math.

- [ ] Intro presentation
- [ ] Write up of equation sets for reference
- [ ] (in-progress) implement fulltensor 1D1V code
- [ ] Implement straightforward 1D1V version of DLR
- [ ] weak Landau damping test case for validation
- [ ] Langmuir sheath demo run scripts for each code
- [ ] Add Fokker-Planck collisions to fulltensor and DLR codes
- [ ] Differentiable fluid code harness
- [ ] Integrate sim runner code with MLFlow for experiment tracking and data collection
- [ ] Test out neural network flux BC training with different objective functions


## Tentative summer school project list

- [ ] Implement differentiable DLR version of vlasov sheath model
- [ ] Get to end-to-end run through WDM
- [ ] Perform training of sheath model surrogate
- [ ] Perform end-to-end optimization via surrogate
