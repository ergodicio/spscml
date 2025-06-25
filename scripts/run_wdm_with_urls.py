import sys
sys.path.append("src")
sys.path.append("tesseracts")

import os

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import jpu

jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()

# Parse command line arguments
def parse_args():
    args = sys.argv[1:]  # Skip the script name
    
    # Default values
    Vc0 = 40*1e3  # 40kV
    T = 20.0      # 20 eV
    Vp = 500.0    # 500 V
    sheath_tesseract_url = None
    wdm_tesseract_url = None
    
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
        elif args[i] == '--sheath-tesseract-url' and i + 1 < len(args):
            sheath_tesseract_url = args[i + 1]
            i += 2
        elif args[i] == '--wdm-tesseract-url' and i + 1 < len(args):
            wdm_tesseract_url = args[i + 1]
            i += 2
        elif args[i] == '--help' or args[i] == '-h':
            print("Usage: python run_wdm_with_urls.py [--Vc0 VALUE] [--T VALUE] [--Vp VALUE] [--sheath-tesseract-url URL] [--wdm-tesseract-url URL]")
            print("  --Vc0 VALUE                Total capacitor voltage in volts (default: 40000)")
            print("  --T VALUE                  Initial temperature in eV (default: 20.0)")
            print("  --Vp VALUE                 Initial plasma voltage in volts (default: 500.0)")
            print("  --sheath-tesseract-url URL URL for the sheath tesseract (required)")
            print("  --wdm-tesseract-url URL    URL for the WDM tesseract (required)")
            print("  --help, -h                 Show this help message")
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}")
            print("Use --help for usage information")
            sys.exit(1)
            
    # Check required arguments
    if sheath_tesseract_url is None:
        print("Error: --sheath-tesseract-url is required")
        print("Use --help for usage information")
        sys.exit(1)
        
    if wdm_tesseract_url is None:
        print("Error: --wdm-tesseract-url is required")
        print("Use --help for usage information")
        sys.exit(1)
            
    return Vc0, T, Vp, sheath_tesseract_url, wdm_tesseract_url

Vc0, T_input, Vp_input, sheath_tesseract_url, wdm_tesseract_url = parse_args()

print(f"Running WDM simulation with:")
print(f"  Vc0 = {Vc0} V")
print(f"  T = {T_input} eV") 
print(f"  Vp = {Vp_input} V")
print(f"  Sheath tesseract URL = {sheath_tesseract_url}")
print(f"  WDM tesseract URL = {wdm_tesseract_url}")

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

with Tesseract.from_url(sheath_tesseract_url) as sheath_tx:
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

    with Tesseract.from_url(wdm_tesseract_url) as tx:
        result = tx.apply(dict(
            Vc0=Vc0,
            Ip0=Ip.magnitude,
            a0=a0.magnitude,
            N=N.magnitude,
            Lp_prime=Lp_prime,
            Lz=Lz,
            R=R, L=L, C=C,
            sheath_tesseract_url=sheath_tx._client.url,
            dt=5e-8,
            t_end=2e-5,
        ))
