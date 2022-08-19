import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

first_weight = """
0.21474454	0.14865324	0.14111608	0.13819733	0.14426127	0.183027
0.11492428	0.29872444	0.113333374	0.12730005	0.13789952	0.17781821
0.13849735	0.14772971	0.21698636	0.13806593	0.14363718	0.18508312
0.12746489	0.15561955	0.121785544	0.23726475	0.14425661	0.1836083
0.11272178	0.14606062	0.110179946	0.12715179	0.27735785	0.19652782
0.10857911	0.14568451	0.09943957	0.12249494	0.15184595	0.3359554
"""

middle_weight = """
0.26477218	0.1335136	0.15751733	0.13426848	0.14587525	0.16405389
0.115978375	0.3618456	0.11550732	0.11745408	0.13474053	0.15447386
0.15934065	0.13454704	0.2641083	0.13572292	0.13990092	0.16638067
0.13869548	0.13459742	0.13310026	0.28866422	0.14293364	0.16200894
0.110697016	0.13192795	0.1053854	0.11650741	0.35354406	0.18193805
0.09663497	0.11066316	0.095169134	0.09818834	0.14076902	0.45857507
"""

last_weight = """
0.2820835	0.1311431	0.15354006	0.12671289	0.14484522	0.16167545
0.10827513	0.38334787	0.10916026	0.11242162	0.13484009	0.15195501
0.14911291	0.1312989	0.28561625	0.12751706	0.14012507	0.16632935
0.13134408	0.13290608	0.12831026	0.30429775	0.14147492	0.16166699
0.10451011	0.12811787	0.100800306	0.11037472	0.37666553	0.17953022
0.0942795	0.11014229	0.09348829	0.09690772	0.14210545	0.4630763
"""


sns.set(font_scale=1.4)
first_weight = [[float(w) - 0.04 for w in weight.split()] for weight in first_weight.strip().split("\n")]
middle_weight = [[float(w) - 0.04 for w in weight.split()] for weight in middle_weight.strip().split("\n")]
last_weight = [[float(w) - 0.04 for w in weight.split()] for weight in last_weight.strip().split("\n")]


def make_data_frame(data):
    df = pd.DataFrame(data, columns=['ES', 'NL', 'PT', 'RO', 'RU', 'ZH'])
    df.index = ['ES', 'NL', 'PT', 'RO', 'RU', 'ZH']
    return df


def draw_overall(weight, name, m, layer=None):
    weight = make_data_frame(weight)
    plot = sns.heatmap(weight, cmap='GnBu', vmax=m)
    plot.set_xlabel('Auxiliary Languages', labelpad=5)
    plot.set_ylabel('Language of Interests')
    plot.xaxis.set_ticks_position('top')
    plot.xaxis.set_label_position('top')
    if layer is not None:
        plot.text(1.5, 6.5, f'{layer}')
    plt.savefig(name)
    plot.clear()


draw_overall(first_weight, 'data/figure-first-weight.pdf', m=0.15, layer="First 25% Tokens")
draw_overall(middle_weight, 'data/figure-middle-weight.pdf', m=0.15, layer="First 25% Tokens")
draw_overall(last_weight, 'data/figure-last-weight.pdf', m=0.15, layer="First 25% Tokens")
