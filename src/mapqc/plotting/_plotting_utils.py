import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def translate_values_to_colors_rgba(
    point_color_values,
    point_ref_q_values,
    point_filtering_values,
    vmin,
    vmax,
    cmap_name,
) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array(
        [
            _value_to_color(point, r_or_q, f, cmap, norm)
            for point, r_or_q, f in zip(
                point_color_values,
                point_ref_q_values,
                point_filtering_values,
                strict=False,
            )
        ]
    )
    return colors


def _value_to_color(value, r_or_q, filtering, cmap, norm):
    if r_or_q == "r":
        return (0.5, 0.5, 0.5, 1.0)  # Grey for reference
    elif isinstance(value, float) and not np.isnan(value):
        return cmap(norm(value))
    elif filtering == "not sampled":
        return mcolors.to_rgba("black")
    else:
        return mcolors.to_rgba("darkolivegreen")
