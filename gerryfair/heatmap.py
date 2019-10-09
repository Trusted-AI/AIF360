import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from gerryfair.reg_oracle_class import *

def calc_disp(A_p, X, y_g, X_sens, g):
    """Return the fp disparity in a group g."""
    X_0 = pd.DataFrame([X_sens.iloc[u, :] for u, s in enumerate(y_g) if s == 0])
    group_0 = g.predict(X_0)
    n = len(y_g)
    g_size_0 = np.sum(group_0) * 1.0 / n
    FP = [A_p[i] for i, c in enumerate(y_g) if c == 0]
    FP = np.mean(FP)
    group_members = g.predict(X_sens)
    fp_g = [A_p[i] for i,c in enumerate(y_g) if group_members[i] == 1 and c == 0]
    if len(fp_g) == 0:
        return 0
    fp_g = np.mean(fp_g)
    return (FP - fp_g) * g_size_0

def heat_map(X, X_prime, y, A, eta, plot_name, vmin=None, vmax=None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    
    columns = [str(c) for c in X_prime.columns]
      
    attribute_1 = np.zeros(int(1/eta))
    attribute_2 = np.zeros(int(1/eta))
    disparity = np.zeros((int(1/eta), int(1/eta)))

    ind = 0.0
    for i in range(int(1/eta)):
        for j in range(int(1/eta)):
            beta = [-1 + 2*eta*i, -1 + 2*eta*j]
            group = LinearThresh(beta)
            
            attribute_1[i] = beta[0]
            attribute_2[j] = beta[1]
            disparity[i,j] = calc_disp(A_p=A, X=X, y_g=y, X_sens=X_prime, g=group)

    X_plot, Y_plot = np.meshgrid(attribute_1, attribute_2)

    ax.set_xlabel(columns[0] + ' coefficient')
    ax.set_ylabel(columns[1] + ' coefficient')
    ax.set_zlabel('gamma disparity')
    ax.plot_surface(X_plot, Y_plot, disparity, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=vmin, vmax=vmax)   
    ax.set_zlim3d([-0.035, 0.035])
    fig.savefig('{}.png'.format(plot_name))
    plt.cla()
    return [np.min(disparity), np.max(disparity)]