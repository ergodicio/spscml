from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

import matplotlib.pyplot as plt

with tx as Tesseract.from_image("rlc"):
    result = apply_tesseract(ts, {
        "Vc0": 2.5e4,
        "C": 200.0,
        "L": 245.0,
        "R": 1.4,
        "sheath_tesseract": "tanh_sheath",
    })

    plt.plot(result["sol_ts"], result["sol_Ip"])

