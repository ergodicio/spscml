import sys
sys.path.append("src")

import matplotlib.pyplot as plt

from spscml.fulltensor_vlasov.sheath_model import calculate_plasma_current


results = calculate_plasma_current(1500, 20, 6e22, 0.5)


fig, axes = plt.subplots(4, 1, figsize=(10, 8))
axes[0].imshow(results['fe'].T, origin='lower')
axes[0].set_aspect("auto")
axes[0].set_title("$f_e$")
axes[1].imshow(results['fi'].T, origin='lower')
axes[1].set_aspect("auto")
axes[1].set_title("$f_i$")
axes[2].plot(results['ji'].T, label='ji')
axes[2].plot(-results['je'].T, label='-je')
axes[2].plot(results['E'], label='E')
axes[2].legend()
axes[3].plot(results['ne'], label='ne')
axes[3].plot(results['ni'], label='ni')
axes[3].legend()

plt.show()
