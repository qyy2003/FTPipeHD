import matplotlib.pyplot as plt
import numpy as np
import scienceplots

COLOR11= (114/255.0,
        188/255.0,
        213/255.0)
COLOR1= (94/255.0,
        168/255.0,
        193/255.0)
COLOR21= (255/255.0,
        208/255.0,
        111/255.0)
COLOR2= (235/255.0,
        188/255.0,
        91/255.0)
COLOR31= (231/255.0,
        98/255.0,
        84/255.0)
COLOR3= (211/255.0,
        78/255.0,
        64/255.0)

plt.style.use(['science', 'ieee', 'grid', 'std-colors'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
})     
# i = 0
tim=[[[[[14.14, 122.29], [17.95, 160.32], [82.96, 652.5]], [[17.9, 145.35], [22.81, 202.19], [95.25, 825.15]]],[[[14.14, 122.29], [17.95, 160.32], [82.96, 652.5]], [[17.9, 145.35], [22.81, 202.19], [95.25, 825.15]]]],[[[[14.14, 122.29], [17.95, 160.32], [82.96, 652.5]], [[17.9, 145.35], [22.81, 202.19], [95.25, 825.15]]],[[[14.14, 122.29], [17.95, 160.32], [82.96, 652.5]], [[17.9, 145.35], [22.81, 202.19], [95.25, 825.15]]]]]
figure, axis = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        labels = ["1","8"]
        tim0=tim[i][j][0]
        tim1=tim[i][j][1]


        x = np.arange(len(labels))  # the label locations
        width = 0.6 # the width of the bars
        x = x*2
        ax = axis[i][j]
        rects1 = ax.bar(x - width, tim0[0], width, label='BERT-Base T2U',  color=COLOR1)
        rects11 = ax.bar(x - width,tim1[0], width, bottom=tim0[0], label='BERT-Base U2T', color=COLOR11)
        rects2 = ax.bar(x, tim0[1], width, label='GPT2-Medium T2U', color=COLOR2)
        rects21 = ax.bar(x,tim1[1],width,   bottom=tim0[1],  label='GPT2-Medium U2T', color=COLOR21)
        rects3 = ax.bar(x + width, tim0[2], width, label='LLaMA-7B T2U', color=COLOR3)
        rects31 = ax.bar(x + width, tim1[2], width, bottom=tim0[2],  label='LLaMA-7B U2T', color=COLOR31)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Total Con. Overhead(ms)", fontsize=25)
        # ax.set_xlabel('Batch Size', labelpad=0.5,fontsize=25)
        # ax.set_title("Adaption Performance Cross-validation", fontsize=25)

        # ax.set_xticks(x, fontsize=20)
        # ax.set_xticklabels(labels, fontsize=20)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        ax.set_ylim(0, 1700)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Batch Size", fontsize=25)
        ax.legend(fontsize=20, loc='upper left')



        ax.bar_label(rects11,weight="bold", fontsize=20,fmt='%.2f')
        ax.bar_label(rects21,weight="bold", fontsize=20,fmt='%.2f')
        ax.bar_label(rects31,weight="bold", fontsize=20,fmt='%.2f')

        ax.title("")

# fig.tight_layout()
plt.gcf().set_size_inches(20, 15)
figure.savefig("transfer_time.pdf", dpi=600)
