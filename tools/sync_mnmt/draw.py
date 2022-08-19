import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


csfont = {'fontname':'Songti SC'}
sns.set(font_scale=1.0)


overall_weight = """
0.6907328	0.05515064	0.069206096	0.057479333	0.061313286	0.06612482
0.046656977	0.671384	0.10787294	0.05090123	0.058818083	0.064360514
0.064329006	0.0536613	0.64251236	0.056297828	0.12045065	0.06273528
0.05460502	0.1171081	0.05407991	0.6519835	0.061410908	0.06080836
0.042946182	0.047655467	0.043243907	0.11180153	0.6909826	0.063368805
0.037834104	0.04184243	0.037965707	0.041070186	0.05280098	0.7884907
"""

layer_weight = ["""
0.99194795	0.000807865	0.002558346	0.000418425	0.004135369	0.000150961
0.001723692	0.98862064	0.006252248	0.001448852	0.00159582	0.000361924
0.002606744	0.001035094	0.98891306	0.000719905	0.006400962	0.000326187
0.001797993	0.003232909	0.001062864	0.9919208	0.00177294	0.000232847
0.001155898	0.000835682	0.002174834	0.007026142	0.98870623	0.000122179
2.12E-05	3.01E-05	3.15E-05	1.26E-05	3.36E-05	0.9998944
""", """	
0.989004	0.00221765	0.003655971	0.001754782	0.002094598	0.001271288
0.000384946	0.99620533	0.000720559	0.000478081	0.001578934	0.000633611
0.001519263	0.001085949	0.99372286	0.00127757	0.001802472	0.000589793
0.008913408	0.004620257	0.01995818	0.9606305	0.001845885	0.004033747
0.006284732	0.003272695	0.01627441	0.000550447	0.9710767	0.00254151
0.003246949	0.005048389	0.01445771	0.000550127	0.000721982	0.9759755
""", """	
0.882671	0.012664254	0.049315736	0.022522992	0.017865203	0.014960673
0.009703749	0.92529666	0.011493268	0.021449646	0.018328035	0.013727811
0.026746843	0.020474494	0.89996505	0.02097645	0.01905812	0.012779245
0.012971388	0.007475301	0.016128402	0.9263636	0.02613241	0.010929679
0.009657225	0.007046394	0.014222743	0.016956497	0.94308007	0.009037741
0.01426111	0.009036552	0.012771968	0.013626756	0.016597478	0.93370754
""", """	
0.68244565	0.061458252	0.064146526	0.051858447	0.07102333	0.06906822
0.043425743	0.7492967	0.033305094	0.040343937	0.062471163	0.071158424
0.06658234	0.05488963	0.7167605	0.048318442	0.05778751	0.055661052
0.045737956	0.045778256	0.037572555	0.763332	0.05913715	0.048442125
0.025088985	0.027654843	0.023291484	0.02782794	0.85973597	0.036400996
0.015411317	0.02257704	0.012937512	0.015289734	0.023451254	0.9103339
""", """	
0.6211358	0.069612734	0.07866667	0.06846905	0.07728538	0.08482848
0.05806372	0.67173845	0.05785907	0.060133815	0.07200636	0.080199905
0.07800008	0.069334246	0.6234884	0.06877748	0.074769184	0.08562815
0.07249195	0.07204087	0.075203635	0.61873597	0.07603174	0.08549649
0.05871913	0.06922535	0.061374873	0.05976639	0.6570329	0.093881816
0.051363733	0.061941717	0.05603129	0.05223783	0.07554532	0.7028813	
""", """				
0.3450017	0.11675349	0.141887	0.13314621	0.12029216	0.14292054
0.10896646	0.40399384	0.09810546	0.12190209	0.12650886	0.14052275
0.13404262	0.10690823	0.3850097	0.13021965	0.10742295	0.13639706
0.12213954	0.11339184	0.11714189	0.38370752	0.12936348	0.1342565
0.10434276	0.111950174	0.09702763	0.11780238	0.4219817	0.14689542
0.09458718	0.09553404	0.08999694	0.11302112	0.12563662	0.48122442
"""]


overall_weight = [[float(w) for w in weight.split()] for weight in overall_weight.strip().split("\n")]
layer_weight = [[[float(w) for w in weight.split()] for weight in layer_w.strip().split("\n")] for layer_w in layer_weight]


def make_data_frame(data):
    df = pd.DataFrame(data, columns=['ES', 'NL', 'PT', 'RO', 'RU', 'ZH'])
    df.index = ['ES', 'NL', 'PT', 'RO', 'RU', 'ZH']
    return df


def draw_overall(weight, name, m, layer=None):
    fig = plt.figure()
    weight = make_data_frame(weight)
    plot = sns.heatmap(weight, cmap='GnBu', vmax=m)
    plot.set_xlabel('主要语言', labelpad=10, fontsize=15, **csfont)
    plot.set_ylabel('辅助语言', labelpad=10, fontsize=15, **csfont)
    plot.xaxis.set_ticks_position('top')
    plot.xaxis.set_label_position('top')
    if layer is not None:
        plot.text(2.7, 6.5, f'Layer {layer}',)
    # plt.show()
    plt.savefig(name)
    plot.clear()


def draw_layer_weight(weights):
    _, axes = plt.subplots(3, 2)
    for i, weight in enumerate(weights):
        weight = make_data_frame(weight)
        plot = sns.heatmap(weight, cmap='GnBu', ax=axes[i//2][i%2], vmax=1)
    plt.show()


draw_overall(overall_weight, 'data/figure-7-overall-weight.pdf', m=1)

for i, weight in enumerate(layer_weight):
    draw_overall(weight, f'data/figure-8-layer{i}-weight.pdf', m=1.0, layer=i)

draw_overall(layer_weight[-1], 'data/figure-9-lang-weight.pdf', m=0.15)