import numpy as np
import warnings
from numpy import zeros,ones,linalg
from sklearn.cross_validation import train_test_split
import mosek
import math
import sys
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    pass
    # sys.stdout.write(text)
    # sys.stdout.flush()


def kernel(x1,x2,sigma):
    return math.exp(-1*math.pow(linalg.norm(np.subtract(x1, x2)), 2)/(2*math.pow(sigma,2)))


def getRisk(alpha,x,m,sigma):
    risk = 0
    for i in range(m):
        for j in range(m):
            risk += alpha[i]*alpha[j]*kernel(x[i],x[j],sigma)
    risk *= 1/2
    return risk


def createHyperPlane(x,v,sigma):
    m = len(x)
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
    env = mosek.Env()
    # Attach a printer to the environment
    env.set_Stream (mosek.streamtype.log, streamprinter)
    # Create a task
    task = env.Task()
    task.set_Stream (mosek.streamtype.log, streamprinter)
    # Set up and input bounds and linear coefficients
    bkc = [ mosek.boundkey.fx]
    blc = [ 1.0 ]
    buc = [ 1.0 ]

    bkx = []
    for i in range(m):
        bkx.append(mosek.boundkey.ra)
    bux = (ones((m))) * (1/(v*m))
    blx = (ones((m))) * (-1/(v*m))
    c = (zeros((m)))

    numvar = len(bkx)
    numcon = len(bkc)

    # Append 'numcon' empty constraints.
    # The constraints will initially have no bounds.
    task.appendcons(numcon)

    # Append 'numvar' variables.
    # The variables will initially be fixed at zero (x=0).
    task.appendvars(numvar)

    for j in range(numvar):
        # Set the linear term c_j in the objective.
        task.putcj(j,c[j])
        # Set the bounds on variable j
        # blx[j] <= x_j <= bux[j]
        task.putbound(mosek.accmode.var,j,bkx[j],blx[j],bux[j])

        # Input i, j of A
        task.putaij(0, j, 1)

    for i in range(numcon):
        task.putbound(mosek.accmode.con,i,bkc[i],blc[i],buc[i])

    # Input the objective sense (minimize/maximize)
    task.putobjsense(mosek.objsense.maximize)

    # Set up and input quadratic objective
    for i in range(m):
        for j in range(i+1):
            task.putqobjij(i,j,kernel(x[i],x[j],sigma))

    task.putobjsense(mosek.objsense.minimize)

    # Optimize
    task.optimize()
    # Print a summary containing information
    # about the solution for debugging purposes
    task.solutionsummary(mosek.streamtype.msg)

    prosta = task.getprosta(mosek.soltype.itr)
    solsta = task.getsolsta(mosek.soltype.itr)

    # Output a solution
    xx = zeros(numvar, float)
    task.getxx(mosek.soltype.itr, xx)

    # if solsta == mosek.solsta.optimal or solsta == mosek.solsta.near_optimal:
    #     print("Optimal solution: %s" % xx)
    # elif solsta == mosek.solsta.dual_infeas_cer:
    #     print("Primal or dual infeasibility.\n")
    # elif solsta == mosek.solsta.prim_infeas_cer:
    #     print("Primal or dual infeasibility.\n")
    # elif solsta == mosek.solsta.near_dual_infeas_cer:
    #     print("Primal or dual infeasibility.\n")
    # elif  solsta == mosek.solsta.near_prim_infeas_cer:
    #     print("Primal or dual infeasibility.\n")
    # elif mosek.solsta.unknown:
    #     print("Unknown solution status")
    # else:
    #     print("Other solution status")

    alpha = xx

    return getRisk(alpha,x,m,sigma)

def createPlot(x_0,x_1,x_2,file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_0,x_1,x_2)
    plt.savefig(file_name+'.ps')

def Slab_SVM(file_name, v, sigma):
    plydata = PlyData.read(open(file_name))
    vertexes = plydata['vertex'][:]

    x = np.zeros((len(vertexes),3))
    for i in range(len(vertexes)):
        x[i] = np.array([vertexes[i][0],vertexes[i][1],vertexes[i][2]])

    file_name_points = len(x)

    x_0 = plydata['vertex']['x']
    x_1 = plydata['vertex']['y']
    x_2 = plydata['vertex']['z']
    #createPlot(x_0,x_1,x_2,file_name)

    y = np.zeros(len(x))
    x,ignore_x,y,ignore_y=train_test_split(x,y,train_size=0.01,random_state=8)
    len_x = len(x)

    risk = createHyperPlane(x,v,sigma)

    print file_name,'points:',file_name_points,'subsample:',len_x,'v',v,'sigma',sigma,'risk',risk

args = sys.argv
# print args
if len(args)==4:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Slab_SVM(args[1], float(args[2]), float(args[3]))
#
# drill_file_list = ['drill_1.6mm_0_cyb.ply','drill_1.6mm_210_cyb.ply','drill_1.6mm_30_cyb.ply',
#          'drill_1.6mm_120_cyb.ply','drill_1.6mm_240_cyb.ply','drill_1.6mm_330_cyb.ply',
#          'drill_1.6mm_150_cyb.ply','drill_1.6mm_270_cyb.ply','drill_1.6mm_60_cyb.ply',
#          'drill_1.6mm_180_cyb.ply','drill_1.6mm_300_cyb.ply','drill_1.6mm_90_cyb.ply']
# Slab_SVM(drill_file_list)
#
# bunny_file_list = ['bun000.ply', 'bun090.ply', 'bun270.ply', 'chin.ply', 'top2.ply',
#                    'bun045.ply', 'bun180.ply', 'bun315.ply', 'ear_back.ply', 'top3.ply']
# Slab_SVM(bunny_file_list)
