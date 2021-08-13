# import matplotlib.pyplot as plt
# import numpy as np
# import mplcursors
#
# data = np.outer(range(10), range(1, 5))
#
# fig, ax = plt.subplots()
# lines = ax.plot(range(3), range(3), "o")
# labels = ["a", "b", "c"]
# cursor = mplcursors.cursor()
# fig.canvas.mpl_connect(
#     "motion_notify_event",
#     lambda event: fig.canvas.toolbar.set_message(""))
# cursor = mplcursors.cursor(hover=True)
# cursor.connect(
#     "add",
#     lambda sel: fig.canvas.toolbar.set_message(
#         sel.annotation.get_text().replace("\n", "; ")))
#
# mplcursors.cursor() # or just mplcursors.cursor()
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import mplcursors

data = np.outer(range(10), range(1, 5))

fig, ax = plt.subplots()
ax.set_title("Multiple non-draggable annotations")
ax.plot(data)

mplcursors.cursor(multiple=True).connect(
    "add", lambda sel: sel.annotation.draggable(True))

plt.show()