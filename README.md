# Dynamical Low-Rank Project for the Structure-Preserving Scientific Computing and Machine Learning summer school

**People**:
- Jingwei Hu, project organizer (hujw@uw.edu)
- Jack Coughlin, project lead ([johnbcoughlin.com], jack.coughlin@simulation.science)
- Howard Cheng, subject matter expert (yhcheng8@uw.edu)

## Getting set up

This repository uses `uv` to manage python dependencies. Install it from here: https://github.com/astral-sh/uv

Run `uv sync` to synchronize dependencies.

We'll run commands in this repository with `uv run`. This command wrapper avoids the need for managing virtualenvs.
Use it like this:
```
uv run python
```

## Hackathon task list

- [ ] Implement one of both of the Vlasov solvers for the sheath problem
    - [ ] Full-tensor Vlasov
    - [ ] Projector-splitting DLR
        - [ ] Works for the weak Landau damping test case
- [ ] Check that your solver gives correct gradients for the (voltage -> current density) 
      mapping by comparing to finite difference estimates
- [ ] Build the `tanh_sheath` tesseract:
    ```
    uv run tesseract build tesseracts/sheaths/tanh_sheath
    ```
    and test it out
    ```
    uv run tesseract run tanh_sheath apply @tesseracts/sheaths/example_inputs/apply.json
    ```
- [ ] Build the Tesseract that wraps your Vlasov solver
    ```
    uv run tesseract build tesseracts/sheaths/...
    ```
    and test it out:
    ```
    uv run tesseract run vlasov_sheath apply @tesseracts/sheaths/example_inputs/apply.json
    ```
- [ ] Implement the whole-device model ODE solver's implicit Euler step function
- [ ] Test out the whole-device model forward pass:
    ```
    uv run scripts/run_wdm.py
    uv run scripts/run_wdm.py --image vlasov_sheath
    ```
- [ ] Deploy your Vlasov and WDM tesseracts to the cloud by pushing to a github branch

### Vlasov sheath solvers

The repository contains partial code for two Vlasov solvers that can be applied to the plasma sheath problem:
- `fulltensor_vlasov/solver.py`: A full-f Vlasov solver based on a slope-limited finite volume scheme. You'll have to add
    - The E*df/dv term and a Poisson solve call for the electric field
    - The BGK collision term
- `straightforward_dlra/solver.py`: A projector-splitting dynamical low-rank solver. You'll have to add
    - The E*df/dv term and Poisson solve calls for the electric field at each substep
    - The BGK collision term and flux source term
    - The absorbing wall boundary condition handling.
Both files contain `# HACKATHON` comments indicating work to be done to complete the solver.

The scripts `sheath_fulltensor_vlasov.py` and `sheath_dlr_vlasov.py` contain harness code to set up and solve
the sheath problem using the respective solver. For the DLR code, it's suggested to make sure you're on the 
right track by checking against the `weak_landau_damping.py` script.

