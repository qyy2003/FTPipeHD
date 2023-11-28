import matplotlib.pyplot as plt
import numpy as np
import scienceplots

COLOR11= (114/305.0,
        188/305.0,
        213/305.0)
COLOR1= (94/305.0,
        168/305.0,
        193/305.0)
COLOR21= (305/305.0,
        208/305.0,
        111/305.0)
COLOR2= (235/305.0,
        188/305.0,
        91/305.0)
COLOR31= (231/305.0,
        98/305.0,
        84/305.0)
COLOR3= (211/305.0,
        78/305.0,
        64/305.0)

plt.style.use(['science', 'ieee', 'grid', 'std-colors'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
})     
# i = 0
tim=[
    [
# [[[49.31, 96.52, 193.28, 400.32], [63.9, 130.24, 267.63, 530.3], [265.92, 531.07, 1060.91, 2120.79]], [[63.9, 126.56, 302.06, 507.75], [84.98, 169.63, 338.97, 679.47], [337.72, 674.79, 1353.16, 2704.76]]],
# [[[30.8, 53.21, 106.04, 212.22], [33.83, 70.46, 138.46, 285.9], [137.4, 285.07, 557.49, 1112.66]], [[46.43, 89.71, 182.63, 357.89], [59.79, 118.46, 237.58, 478.41], [237.9, 475.36, 955.83, 1905.33]]]
[[[17.00589656829834, 27.139949798583984, 54.120850563049316, 109.30731296539307], [15.90123176574707, 33.84451866149902, 70.48687934875488, 144.0183401107788], [68.82917881011963, 141.9966697692871, 287.6809597015381, 588.5805606842041]], [[12.491106986999512, 24.90401268005371, 48.84319305419922, 97.9151964187622], [16.223573684692383, 32.70447254180908, 64.58537578582764, 129.137921333313], [64.03744220733643, 127.03382968902588, 256.00578784942627, 524.9857664108276]]],

[[[8.3420991897583, 16.537094116210938, 32.6573371887207, 66.56801700592041], [10.03429889678955, 21.314334869384766, 50.95996856689453, 90.54267406463623], [42.272233963012695, 87.59212493896484, 196.8550682067871, 342.09632873535156]], [[26.089787483215332, 51.676082611083984, 102.06146240234375, 205.32395839691162], [34.31408405303955, 68.36462020874023, 159.85107421875, 299.26209449768066], [137.829852104187, 273.71060848236084, 612.9717350006104, 1085.551905632019]]]

    ],
    [
[[[1.8779, 3.6005000000000003, 7.129099999999999, 13.9899], [2.3631, 4.642799999999999, 8.978399999999999, 18.718699800000003], [9.191799999999999, 18.6459001, 37.0965996, 74.1044999]], [[1.7373, 2.6552000000000002, 5.808400000000001, 9.888000000000002], [1.4157, 2.9415999999999998, 6.6708, 13.550400000000002], [6.495700000000001, 12.617799999999999, 26.088799900000005, 53.5392993]]],
[[[2.0306000000000006, 2.257, 4.4201, 9.232699999999998], [1.5076, 2.9483000000000006, 5.816600000000001, 17.9536997], [5.9399999999999995, 11.5918, 22.8772999, 47.29150009999999]], [[0.8492, 1.0843, 2.6455999999999995, 5.6993], [0.7773, 2.0898000000000003, 3.7466999999999997, 7.600299999999999], [3.6184, 7.7458, 14.962800000000001, 32.395500399999996]]]


   ]
]
title=[["a) Nvidia Jetson Nano","b) Nvidia Jetson Orin"],["c) Mi 10 Lite Zoom","d) Redmi K50"]]
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
figure, axis = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        labels = ["Batch Size=1","Batch Size=2","Batch Size=4","Batch Size=8"]
        # labels = ["1",  "8"]
        tim0=tim[i][j][0]
        tim1=tim[i][j][1]


        x = np.arange(len(labels))  # the label locations
        width = 0.5 # the width of the bars
        x = x*2
        ax = axis[i][j]
        rects1 = ax.bar(x - width, tim0[0], width,hatch="/",edgecolor='black', linewidth=0.5, label='BERT-Base T2U',  color=COLOR1)
        rects11 = ax.bar(x - width,tim1[0], width, hatch="\\",edgecolor='black', linewidth=0.5,bottom=tim0[0], label='BERT-Base U2T', color=COLOR11)
        rects2 = ax.bar(x, tim0[1], width,hatch="|",edgecolor='black', linewidth=0.5, label='GPT2-Medium T2U', color=COLOR2)
        rects21 = ax.bar(x,tim1[1],width, hatch="-", edgecolor='black', linewidth=0.5, bottom=tim0[1],  label='GPT2-Medium U2T', color=COLOR21)
        rects3 = ax.bar(x + width, tim0[2], width,hatch="" ,edgecolor='black', linewidth=0.5,label='LLaMA-7B T2U', color=COLOR3)
        rects31 = ax.bar(x + width, tim1[2], width,hatch=".", edgecolor='black', linewidth=0.5,bottom=tim0[2],  label='LLaMA-7B U2T', color=COLOR31)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Total Con. Overhead(ms)", fontsize=30)
        # ax.set_xlabel('Batch Size', labelpad=0.5,fontsize=30)
        # ax.set_title("Adaption Performance Cross-validation", fontsize=30)

        # ax.set_xticks(x, fontsize=20)
        # ax.set_xticklabels(labels, fontsize=20)
        ax.tick_params(axis='y', labelsize=30)
        ax.set_ylim(0, 5000)
        ax.set_xticks(x)
        # ax.set_yticklabels( fontsize=30)
        ax.set_xticklabels(labels, fontsize=30)
        # ax.set_xlabel("Batch Size", fontsize=30)
        ax.legend(fontsize=30, loc='upper left')



        ax.bar_label(rects11,weight="bold", fontsize=20,fmt='%.1f')
        ax.bar_label(rects21,weight="bold", fontsize=20,fmt='%.1f')
        ax.bar_label(rects31,weight="bold", fontsize=20,fmt='%.1f')

        ax.set_title(r'\textbf{%s}'%title[i][j],fontsize=35, y=-0.17)

# fig.tight_layout()
plt.gcf().set_size_inches(30, 15)
figure.savefig("transfer_time1.pdf", dpi=600)
