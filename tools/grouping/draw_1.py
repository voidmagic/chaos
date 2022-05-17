import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure


overall_weight = """
0.25656289	0.01762348	-0.02133828	-0.03293806	0.02985221	-0.01447386	0.02803624	0.00314218	0.06537795	-0.01170975	-0.01893502	0.00840724	0.01081872	-0.02679461	-0.04861206	-0.05578834	-0.02581602	-0.01284832	0.01425242	0.02748513	-0.04249704	-0.02627814	-0.00231421	-0.01029164	-0.00055629	0.03679883	0.01870173	0.03701234
0.02492672	0.1477474	-0.0067969	-0.03457785	0.05721653	0.03958946	-0.01293701	0.03545994	0.02594823	0.00790977	0.03915012	-0.0009352	-0.00363421	-0.04672056	-0.00236237	-0.01173347	-0.04423612	0.01255929	-0.02292383	0.01557285	0.04295957	-0.00338304	-0.01062548	-0.02235585	-0.01869726	0.03236324	-0.0006609	-0.02381033
-0.00510544	0.00508851	0.08381766	-0.00252527	-0.03080213	-0.01159865	0.00516593	0.02751267	-0.01890677	0.01291656	0.01667333	0.01128137	0.00380528	0.00842541	-0.01719159	-0.01121074	0.03279305	-0.00350028	0.01110268	0.03890902	0.06541127	-0.01554519	0.04387146	-0.00695503	0.00783259	0.01649034	-0.0072493	0.03425252
-0.02175027	-0.02023566	0.02003568	0.10520482	0.00090349	0.00149506	-0.02295268	0.01856732	-0.00760537	0.02111214	0.04743814	0.00694525	0.03591746	0.01213497	0.00844914	0.04524744	0.05862182	0.00086808	-0.0429377	0.05908108	-0.01256734	0.00189728	-0.00445163	-0.05011916	0.01848847	0.05342561	-0.00850457	0.00983995
0.03363001	-0.00532573	-0.00247431	-0.01626348	0.10146832	0.00441796	0.03668952	0.00834644	0.0326739	-0.03132987	-0.0078848	-0.0561319	0.00610673	-0.03592366	0.01297778	-0.01313633	0.01142311	0.00080222	0.01960939	0.05716598	0.04338396	-0.01111597	0.02015781	0.0160678	-0.00020492	0.02109432	0.00471705	0.00563997
-0.02304327	-0.01013678	0.00758934	-0.02467746	0.00815052	0.13634747	-0.0230844	0.03461969	0.00238866	0.00349015	0.04681802	0.04580694	0.05889183	-0.03741676	-0.08566284	0.02872396	0.02133828	0.08157349	0.084764	0.01375788	-0.00334376	0.02230495	-0.00151503	-0.04061139	0.03495914	0.00812113	0.01473671	0.00988317
0.05663514	-0.02916789	-0.0033617	-0.03984702	0.00314641	-0.04307306	0.15366238	0.01159072	0.05976474	-0.01101267	0.02018124	-0.0054512	-0.03179967	0.01593763	0.02119309	-0.03583544	-0.03953636	-0.02965486	-0.02059168	0.01373619	0.02530378	-0.01466912	-0.02838498	-0.0771656	0.03006923	0.01986432	0.00311017	0.03199661
0.05407882	-0.01414764	-0.0099085	0.01459223	-0.00542438	0.04581457	-0.01965177	0.11829072	0.0071013	-0.02014518	-0.01743227	0.00857335	0.02020335	-0.03750056	-0.02274847	0.02568674	0.02025861	0.02683491	0.05802524	0.03326648	-0.005467	-0.03827679	0.02557015	0.00019389	0.01169246	-0.0144583	-0.0398466	0.01713246
0.06459528	-0.01222456	0.00585496	0.00373495	0.0282315	0.03641361	0.03140664	-0.02823842	0.19026107	-0.01682913	0.02658564	-0.01473749	0.00951302	-0.02349865	0.00523055	-0.01166302	0.00181592	-0.01406229	-0.00437486	0.00201195	-0.00010896	-0.0059827	-0.01312226	-0.00331956	0.05773968	0.00686419	0.02583623	0.01229709
0.00292963	0.01993322	-0.01062506	-0.03410953	0.02098089	0.02506632	-0.0108692	-0.02399534	-0.00801188	0.13565171	0.02704304	-0.00129265	-0.00186568	-0.02519172	0.02319771	0.00307178	0.0040313	0.02400309	0.0392921	0.02977014	0.01119369	0.024885	0.03417563	-0.01950496	0.00292319	0.01290166	0.00456315	0.00918889
0.00164068	-0.00084382	0.0313583	0.00567758	-0.04118818	0.0000608	0.04313272	0.01148093	-0.01726145	-0.01588053	0.15534508	0.0135988	-0.0220145	0.0218811	0.04536349	0.01028496	0.03449064	-0.01704007	0.00736773	0.0158118	0.03851914	-0.00287783	-0.01752883	-0.02832633	0.05602962	0.01566291	-0.01192415	0.0191564
0.0138768	-0.03478932	0.00095391	-0.0316366	-0.01505673	-0.01444829	0.01696467	0.04682821	0.02485734	0.00275367	0.0070672	0.14319384	-0.0024811	-0.01103729	-0.02441812	-0.00229669	0.0414108	0.00837266	0.04090619	0.02735895	-0.0338136	-0.01927072	0.01588708	-0.02422559	0.0051766	0.00713599	0.02566898	0.01061904
-0.00093025	-0.01938486	-0.00078988	0.00859165	-0.01777142	0.06140858	-0.04881752	0.0348652	-0.01059401	0.0095697	-0.0061872	0.03675866	0.09552401	-0.05062526	-0.012483	0.01801795	0.00559616	0.02434903	0.06788075	0.03021926	-0.00710565	0.0077287	0.02241009	-0.10285538	-0.02779347	-0.00281304	0.01570934	-0.00557673
-0.02284384	-0.09353101	0.00541675	-0.02470356	-0.05104321	-0.0329321	0.01742417	-0.04382819	-0.0192892	-0.0225724	0.0286572	-0.01906168	-0.03717703	0.27737212	0.10840905	-0.02547091	-0.01358086	-0.03720409	-0.06736666	0.03225225	-0.01084763	-0.06112516	-0.04041606	0.02525693	0.04135144	-0.03685433	-0.00835204	0.06824511
0.03878528	-0.04454327	-0.02388948	0.00708836	0.00573218	-0.01178628	0.00278574	-0.03775471	-0.01391029	-0.03001255	0.04532236	0.01795822	-0.03780037	0.11800021	0.24249798	0.00882429	-0.03856462	-0.02486902	-0.05296457	0.03373313	-0.00277114	-0.03803676	0.01052332	0.04805815	0.06975144	0.02518815	0.03369075	0.07079315
0.00622249	-0.05249983	-0.01990557	0.02353269	0.00379699	0.01632273	-0.00660443	0.04061717	0.00321358	0.02214515	0.01100385	0.01568186	-0.00729096	-0.01553619	-0.037242	0.17723823	0.03982234	0.02062416	0.01184475	0.00029719	-0.01197541	-0.00029254	0.032125	-0.04622149	0.03616029	0.0031817	0.00658047	0.00280273
0.00828218	-0.04023981	0.01160961	-0.03213423	-0.02811176	0.0221144	-0.02567744	-0.01942307	-0.01624513	0.01311809	0.03565866	0.01537365	0.01844144	-0.02038723	0.00052106	-0.00214982	0.09748656	-0.02221012	-0.01650113	0.05940527	0.02826953	-0.00256258	-0.02091163	-0.06159991	0.05459523	0.05829132	-0.05438566	-0.02564692
0.02940041	0.01726753	-0.03310406	-0.00281453	-0.00350863	0.06067878	0.01000297	0.0160929	0.01375771	-0.0000329	0.0288108	0.03205818	0.0146696	-0.00607324	-0.00139529	0.03144914	0.01269078	0.09553671	0.03541082	0.03612745	0.06019151	0.01450235	-0.02238166	-0.0110988	0.01799947	0.04456925	-0.01517534	-0.0156495
0.01407945	-0.03905326	0.00522685	0.0094465	0.02081132	0.0168134	-0.00767922	0.05022943	-0.02118689	0.01668435	0.00246912	0.02760404	0.02163672	-0.05741137	-0.00679302	-0.01364797	0.00086403	-0.00151438	0.11120063	-0.01828003	-0.01029354	-0.01787674	-0.00333089	-0.02765131	0.01398605	0.00659585	0.02084476	0.00701863
0.03604811	0.05057204	0.0418222	0.01095039	-0.02428019	-0.02884626	-0.01790392	-0.0117743	0.00655782	0.00945055	0.00452143	0.04226905	0.01560116	-0.02001256	0.02353513	-0.0035181	-0.00561452	-0.02217549	-0.00431246	0.09760368	0.0057618	0.00939184	0.00126386	-0.04027444	0.00950319	0.07395315	0.03694957	0.02703953
0.03221619	-0.02275598	0.06555521	-0.01484036	0.02941775	-0.01931667	-0.03132534	0.02192277	0.00758904	0.02715129	0.01642531	0.01654923	-0.01351231	0.00610846	-0.00887579	-0.01802027	0.02845335	-0.00857902	0.03883499	0.01388896	0.13980657	-0.03655738	0.00108558	-0.00503439	0.0037148	0.02904981	-0.00347239	-0.00260568
-0.01363349	0.02575821	0.05995494	0.04170793	0.00074995	0.0263198	-0.02795529	0.02297455	0.01138437	0.06506079	-0.03542888	0.03079289	-0.01125199	-0.05072176	-0.0094288	-0.00009429	-0.01713192	-0.02028888	-0.01381582	0.00632411	0.02021092	0.07955426	-0.01368934	0.00524843	0.01241189	0.02746922	0.01615733	0.0186612
0.00214732	-0.00549287	0.00161827	-0.01185012	-0.0166499	0.00729078	-0.03099817	0.02601045	-0.01467323	0.03644061	-0.03194553	0.02691859	0.06713438	-0.05021459	0.02343231	0.00781578	0.03347385	-0.01519644	0.01376545	0.02155244	0.03341001	0.0189116	0.14383352	-0.043369	-0.00961745	0.02957422	-0.01524478	0.01922339
0.05807292	0.0128364	0.00642949	0.01797175	-0.00063515	-0.02861685	0.00402361	0.02032095	0.00168979	-0.0499832	-0.00573361	-0.04688644	-0.02595514	0.04862881	0.02300942	-0.02705109	-0.01334459	-0.02697319	0.03459144	0.00701493	0.0148626	-0.02049983	-0.03637111	0.3176645	-0.00821888	0.07586223	0.03649199	-0.0223524
0.00182408	0.00638878	0.00212806	0.00854814	-0.02309853	-0.03323448	0.00393504	-0.03799415	-0.02498025	-0.044213	0.02065492	0.02973622	-0.00308174	0.06761718	0.06201154	0.01929498	0.03504205	-0.02534127	-0.05187732	0.03039908	0.05124378	-0.02284855	-0.03090972	-0.04347104	0.25872755	0.02662873	-0.00917542	0.01836377
0.01461256	0.01112086	0.03466815	-0.01365697	0.02759081	-0.00580359	-0.02122742	-0.04828244	0.00271142	-0.00364572	0.01147622	-0.02648026	-0.00204277	-0.03534228	-0.02055001	-0.03002322	-0.00726873	0.00090468	0.00680387	0.03638321	0.00777119	0.01500303	0.00846916	-0.0364275	0.0165292	0.13013798	-0.01640612	0.00361729
0.01631683	-0.0445115	-0.01326185	-0.01359648	-0.02873051	0.02184069	0.0161826	0.00438732	0.01037002	-0.01636058	-0.02492028	0.02302593	-0.00671232	0.01249301	0.03316557	0.01254243	0.00874013	-0.0119822	0.01780272	0.02171242	0.00649655	-0.00799042	-0.01956803	-0.01526183	-0.02141941	0.03023756	0.09758359	0.04749191
0.03269309	0.02619618	-0.00936192	-0.00479186	-0.02327085	-0.03577787	0.02472985	0.0170781	-0.01545858	-0.01412278	-0.00838572	-0.00856304	-0.01644111	0.06943446	0.03574872	0.00267035	0.01305127	-0.03415406	-0.02163279	0.00530863	0.00717396	-0.01396954	-0.02570146	0.00632274	0.01792157	0.01308918	0.01079887	0.17659414
"""

sns.set(font_scale=1)
overall_weight = [[float(w) for w in weight.split()] for weight in overall_weight.strip().split("\n")]

langs = "ar	bg	cs	de	el	es	fa	fr	he	hr	hu	id	it	ja	ko	nl	pl	pt	ro	ru	sk	sr	sv	th	tr	uk	vi	zh".split()
langs1 = "hu zh,ko,ja,th ar,el,fa,he tr,pl,cs,sk bg,hr es,ro,fr,nl,pt de,sr,uk id,vi sv,it,ru".split()
langs1 = [k.split(',') for k in langs1]
langs1 = [item for sublist in langs1 for item in sublist]

indexes = sorted(range(len(langs1)), key=lambda idx: langs1[idx])
langs2 = sorted(langs, key=lambda l: indexes[langs.index(l)])

overall_weight = [sorted(zip(langs, overall_weight_i), key=lambda l: indexes[langs.index(l[0])]) for overall_weight_i in overall_weight]

overall_weight = sorted(zip(langs, overall_weight), key=lambda l: indexes[langs.index(l[0])])
langs = sorted(langs, key=lambda l: indexes[langs.index(l)])


for i in range(len(overall_weight)):
    overall_weight[i] = overall_weight[i][1]
    for j in range(len(overall_weight[i])):
        overall_weight[i][j] = overall_weight[i][j][1]


def make_data_frame(data):
    df = pd.DataFrame(data, columns=langs)
    df.index = langs
    return df


def draw_overall(weight, name, m, n):
    weight = make_data_frame(weight)
    fig, ax = plt.subplots(1, 2, figsize=(20, 6), dpi=20)
    cbar_ax = fig.add_axes([.91, .1, .01, .8])

    plot = sns.heatmap(weight, cmap='GnBu', vmax=m, vmin=n, ax=ax[0], cbar=True, cbar_ax=cbar_ax)
    plot.set_xlabel('Language of Interests', labelpad=8)
    plot.set_ylabel('Auxiliary Languages', labelpad=3)
    plot.xaxis.set_ticks_position('top')
    plot.yaxis.set_ticks_position('left')
    plot.xaxis.set_label_position('top')

    plot = sns.heatmap(weight, cmap='GnBu', vmax=m, vmin=n, ax=ax[1], cbar=False)
    plot.set_xlabel('Language of Interests', labelpad=8)
    plot.set_ylabel('Auxiliary Languages', labelpad=3)
    plot.xaxis.set_ticks_position('top')
    plot.yaxis.set_ticks_position('left')
    plot.xaxis.set_label_position('top')
    plt.show()
    # plt.savefig(name)


flatten = [item for sublist in overall_weight for item in sublist]

draw_overall(overall_weight, 'data/figure-1.pdf', m=max(flatten)/2, n=min(flatten))