import sys
sys.path.append("src")
sys.path.append("tesseracts")

import os

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import wdm.tesseract_api as tesseract_api
import sheaths.vlasov.tesseract_api as vlasov_sheath_tesseract_api
import jpu
import equinox as eqx
import mlflow
from spscml.whole_device_model.local_wrapper import apply

jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()

# Parse command line arguments
def parse_args():
    args = sys.argv[1:]  # Skip the script name

    # Default values
    Vc0 = 40*1e3  # 40kV
    T = 20.0      # 20 eV
    Vp = 500.0    # 500 V
    tesseract = "vlasov_sheath"  # Default tesseract image

    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == '--Vc0' and i + 1 < len(args):
            Vc0 = float(args[i + 1])
            i += 2
        elif args[i] == '--T' and i + 1 < len(args):
            T = float(args[i + 1])
            i += 2
        elif args[i] == '--Vp' and i + 1 < len(args):
            Vp = float(args[i + 1])
            i += 2
        elif args[i] == '--image' and i + 1 < len(args):
            image = args[i + 1]
            i += 2
        elif args[i] == '--help' or args[i] == '-h':
            print("Usage: python run_wdm.py [--Vc0 VALUE] [--T VALUE] [--Vp VALUE] [--image NAME]")
            print("  --Vc0 VALUE   Total capacitor voltage in volts (default: 40000)")
            print("  --T VALUE     Initial temperature in eV (default: 20.0)")
            print("  --Vp VALUE    Initial plasma voltage in volts (default: 500.0)")
            print("  --image NAME  Tesseract image name (default: vlasov_sheath)")
            print("  --help, -h    Show this help message")
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}")
            print("Use --help for usage information")
            sys.exit(1)

    return Vc0, T, Vp, tesseract

Vc0, T_input, Vp_input, image_name = parse_args()

print(f"Running WDM simulation with:")
print(f"  Vc0 = {Vc0} V")
print(f"  T = {T_input} eV")
print(f"  Vp = {Vp_input} V")
print(f"  Tesseract image = {image_name}")

R = 1.5e-3
L = 2.0e-7
C = 222*1e-6

Lz = 0.5
Lp = -.4e-7
Lp_prime = Lp / Lz
L_tot = L - Lp

# Use the initial plasma from Fig 7 of Shumlak et al. (2012) as an example
n0 = 6e22 * ureg.m**-3

Z = 1.0


with Tesseract.from_tesseract_api(vlasov_sheath_tesseract_api) as sheath_tx:
    Vp0 = Vp_input * ureg.volts

    T0 = T_input * ureg.eV
    j = sheath_tx.apply(dict(
        n=n0.magnitude, T=T0.magnitude, Vp=Vp0.magnitude, Lz=0.5
        ))["j"] * (ureg.A / ureg.m**2)
    N = ((8*jnp.pi * (1 + Z) * T0 * n0**2) / (ureg.mu0 * j**2)).to(ureg.m**-1)
    print("N = ", N)

    Ip = (j * N / n0).to(ureg.A)
    print("Ip = ", Ip)

    a0 = ((N / n0 / jnp.pi)**0.5).to(ureg.m)

    result = apply(dict(
        Vc0=Vc0,
        Ip0=Ip.magnitude,
        a0=a0.magnitude,
        N=N.magnitude,
        Lp_prime=Lp_prime,
        Lz=Lz,
        R=R, L=L, C=C,
        dt=5e-8,
        t_end=2e-5,
        mlflow_parent_run_id=None
    ), sheath_tx)
