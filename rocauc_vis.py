import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Scope(object):
    def __init__(self, ax, ax2, dist_1, dist_2):

        self.ax = ax
        self.ax2 = ax2

        self.FPRs = []
        self.TPRs = []

        self.background = self.ax.twinx()

        self.dist_1 = dist_1
        self.dist_2 = dist_2

        Z = max(max(self.dist_1), max(self.dist_2))

        self.dist_1 /= Z
        self.dist_2 /= Z

        kwargs = {
            'norm_hist': False,
            'kde': True,
            'ax': self.background,
            'bins': np.arange(0, 1, .01),

        }

        sns.distplot(
            self.dist_1,
            **kwargs,
            label='TN',
            kde_kws={'lw': 2, 'ls': '--', 'color': 'orange'},
            hist_kws={'color': 'orange', 'alpha': .2}
        )
        sns.distplot(
            self.dist_2,
            **kwargs,
            label='TP',
            kde_kws={'lw': 2, 'ls': '--', 'color': 'purple'},
            hist_kws = {'color': 'purple', 'alpha': .2}
        )

        self.dist_1_x = self.background.lines[0].get_xdata()
        self.dist_1_y = self.background.lines[0].get_ydata()
        self.dist_2_x = self.background.lines[1].get_xdata()
        self.dist_2_y = self.background.lines[1].get_ydata()

        self.max_y = max(max(self.dist_1_y) or 0, max(self.dist_2_y) or 0)

    def set_axis_style(self):

        self.ax.set_xlabel('Discrimination Threshold')
        self.ax.legend(loc='upper left', ncol=3, fontsize=9)
        self.background.legend(loc='upper right', ncol=2, fontsize=9)
        self.ax.set_xticks(np.arange(0, 1.01, .1))
        self.ax.set_xticklabels(list(map(lambda x: '{:.0%}'.format(x), np.arange(0, 1.1, .1))))
        self.ax.set_ylim(0, self.max_y + 1)

        self.background.set_ylim(self.ax.get_ylim()[0], self.ax.get_ylim()[1])

        for a in [self.ax, self.background]:
            a.spines['top'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.set_yticks([])
            a.set_yticklabels([])
            a.set_xlim(0, 1)

        self.ax2.set_xlim(-0.01, 1)
        self.ax2.set_ylim(-0.01, 1)
        self.ax2.set_title('ROC Curve')
        self.ax2.set_xlabel('FPR')
        self.ax2.set_ylabel('TPR')
        self.ax2.set_xticks(np.arange(0, 1.01, .1))
        self.ax2.set_xticklabels(list(map(lambda x: '{:.0%}'.format(x), np.arange(0, 1.1, .1))))
        self.ax2.set_yticks(np.arange(0, 1.01, .1))
        self.ax2.set_yticklabels(list(map(lambda x: '{:.0%}'.format(x), np.arange(0, 1.1, .1))))
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)

    def update(self, threshold):

        self.ax.clear()

        x1 = self.dist_1_x[self.dist_1_x >= threshold - .01 / 2]
        y1 = self.dist_1_y[self.dist_1_x >= threshold - .01 / 2]
        x2 = self.dist_2_x[self.dist_2_x <= threshold + .01 / 2]
        y2 = self.dist_2_y[self.dist_2_x <= threshold + .01 / 2]

        self.ax.fill_between(x1, 0, y1, alpha=.9, color='red', label='FP')
        self.ax.fill_between(x2, 0, y2, alpha=.9, color='blue', label='FN')

        self.ax.axvline(threshold, ls='--', lw=1, color='k', label='Threshold')

        lines = self.ax.lines[0]

        TN = self.dist_1[self.dist_1 <= threshold]
        FN = self.dist_2[self.dist_2 <= threshold]
        FP = self.dist_1[self.dist_1 >= threshold]
        TP = self.dist_2[self.dist_2 >= threshold]

        FPR = len(FP) / (len(FP) + len(TN))
        TPR = len(TP) / (len(TP) + len(FN))

        self.FPRs.append(FPR)
        self.TPRs.append(TPR)

        self.ax2.plot(self.FPRs, self.TPRs, color='green', lw=1)
        self.ax2.plot([0, 1], [0, 1], ls='--', lw=1, color='k')

        self.set_axis_style()

        if threshold == 1:
            self.ax2.annotate(
                xy=(0.7, 0.1),
                s='AUC = {:.2%}'.format(np.trapz(self.TPRs[::-1], self.FPRs[::-1])),
            )
            self.ax2.fill_between(self.FPRs, 0, self.TPRs, color='green', alpha=.1)

        return lines,


step = .005
thresholds = list(np.arange(0, 1+step, step)) + [1.01] * 100

fig, (axis, axis2) = plt.subplots(1, 2, figsize=(12, 4))

dist_1 = np.random.normal(loc=5.0, scale=1.5, size=100000)
dist_2 = np.random.normal(loc=8.0, scale=1.5, size=100000)

scope = Scope(axis, axis2, dist_1, dist_2)

ani = animation.FuncAnimation(
    fig,
    scope.update,
    frames=thresholds,
    interval=1,
    repeat=False,
)

plt.show()
#
# ani.save('animation.gif', writer='imagemagick', fps=30, dpi=80)
