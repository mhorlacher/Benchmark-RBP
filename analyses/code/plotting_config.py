import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set(style="whitegrid", rc={"grid.linestyle": "--"})
husl = sns.color_palette('husl',32)
sns.palplot(husl)
plt.close()
plt.style.use('seaborn-notebook')

SMALL_SIZE=13
MEDIUM_SIZE=15
BIGGER_SIZE=17

mpl.rc('font',size=SMALL_SIZE)
mpl.rc('axes',titlesize=SMALL_SIZE)
mpl.rc('axes',labelsize=MEDIUM_SIZE)
mpl.rc('xtick',labelsize=SMALL_SIZE)
mpl.rc('ytick',labelsize=SMALL_SIZE)
mpl.rc('legend',fontsize=SMALL_SIZE)
mpl.rc('figure',titlesize=BIGGER_SIZE)


def my_savefig(savefig_file, ext_list=['svg','png','pdf'], bbox_inches='tight'):
    for ext in ext_list:
        plt.savefig(savefig_file.format(EXT=ext))