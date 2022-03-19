import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

a = [
    [1.000, 0.551, 0.533, 0.627, 0.464, 0.220, 0.324, 0.302],
    [0.551, 1.000, 0.569, 0.630, 0.534, 0.245, 0.360, 0.334],
    [0.533, 0.569, 1.000, 0.642, 0.429, 0.227, 0.353, 0.372],
    [0.627, 0.630, 0.642, 1.000, 0.583, 0.295, 0.377, 0.385],
    [0.464, 0.534, 0.429, 0.583, 1.000, 0.204, 0.257, 0.316],
    [0.220, 0.245, 0.227, 0.295, 0.204, 1.000, 0.137, 0.138],
    [0.324, 0.360, 0.353, 0.377, 0.257, 0.137, 1.000, 0.220],
    [0.302, 0.334, 0.372, 0.385, 0.316, 0.138, 0.220, 1.000],
]

sns.set(font_scale=1.4)

langs = "DE,ES,FR,NL,RO,PT,IT,RU".split(",")


def make_data_frame(data):
    df = pd.DataFrame(data, columns=langs)
    df.index = langs
    return df


def draw_overall(weight, name, m, layer=None):
    fig = plt.figure(figsize=(14, 6))

    weight = make_data_frame(weight)
    plot = sns.heatmap(weight, cmap='GnBu', vmax=m)
    plot.xaxis.set_label_position('top')
    if layer is not None:
        plot.text(2.7, 6.5, f'Layer {layer}', )
    plt.savefig(name)
    # plt.show()
    plot.clear()


draw_overall(a, 'figure-3.pdf', m=0.7)
