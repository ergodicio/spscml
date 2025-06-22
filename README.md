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


## Glossary

- **Adiabat**: In fluid dynamics, smooth solutions of the ideal fluid equations satisfy a constant entropy relation: `d/dt(T/n^gamma) = 0`, where `gamma = 5/3` is most commonly used in three-dimensional simulations of plasma. An **adiabat** refers to a path through density/temperature space which respects this constant entropy relation. A change is **adiabatic** if it follows an adiabat. State changes can fail to be adiabatic if they are accompanied by non-ideal terms like resistivity or radiative cooling.
- **Adjoint**: In an automatic differentiation setting, refers to the method used to differentiate through a time-dependent simulation.
- **Bremsstrahlung radiation**: A phenomenon occurring in hot plasmas where electrons release radiation as they collide with other particles. This is a primary energy loss mechanism in hot plasmas.
- **Current and current density**: The total current is denoted by `I`. The current _density_ is denoted by `j`. Our plasma simulations produce `j`, and we have to multiply by the cross-sectional area of the pinch to obtain `I`.
- **Debye length**: A fundamental distance in plasmas. This is the distance that a particle travels during one period of the **plasma frequency**. In the "unit normalization", the Debye length is 1.
- **Density**: The number density of particles. Denoted in units of particles per cubic meter.
- **Electron-volt**: The most common measure of **temperature** in plasma physics. One electron-volt is equal to the energy gained by an electron as it falls through a potential difference of one volt. 
- **Inductance**: The tendency of a circuit to resist changes in the current flow. The phenomenon of inductance is related to the magnetic field produced by a current flow: current "wants" to flow through a wire surrounded by a magnetic field. Inductance is a property of the wire geometry.
- **Mean free path**: The average distance that a particle will travel before being turned 90 degrees due to collisions. In our 1D simulations, it's equal to the **thermal velocity** divided by the collision frequency: `lambda_mfp = v_t / nu`.
- **omega_p**: The symbol for the **plasma frequency**.
- **Plasma frequency**: A fundamental frequency in plasmas. This is the frequency of the electrostatic oscillations of particles. The electrons oscillate extremely fast; the ions slightly less so. In a "unit normalization" Vlasov simulation where the charge, temperature and density are all 1, the plasma frequency is also 1.
- **Resistance**: The tendency of a current-carrying medium to resist current flow. Appears in either the total-current form of Ohm's law, `V = IR`, or the volumetric version, `E = eta*j`.
- **Sheath**: The phenomenon of a persistent electrostatic potential at plasma-wall boundaries. The sheath is caused by high electron mobility relative to the ions.
- **Spitzer resistivity**: A well-known estimate of the resistivity of a plasma at a given temperature.
- **Temperature**: The average energy of particles in a plasma.
- **Thermal velocity**: The average speed of particles in a population with a given temperature. Equal to `sqrt(T / m)`, where `m` is the mass. Electrons are much faster than ions: `v_te >> v_ti`.
