import numpy as np
import time
import matplotlib.pyplot as plt;

from RobotPlanner_RTAA import RTAAPlanner

plt.ion()
import matplotlib;matplotlib.use("TkAgg")

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import RobotPlanner

def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def load_map(fname):
    mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'), \
                                      'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
    blockIdx = mapdata['type'] == b'block'
    # boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view(('<f4',9))
    # blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view(('<f4',9))
    boundary = np.array(mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b']].tolist())
    blocks = np.array(mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b']].tolist())
    return boundary, blocks


def draw_map(boundary, blocks, start, goal):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hb = draw_block_list(ax,blocks)
    hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
    hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(boundary[0,0],boundary[0,3])
    ax.set_ylim(boundary[0,1],boundary[0,4])
    ax.set_zlim(boundary[0,2],boundary[0,5])
    return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
    f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
    clr = blocks[:,6:]/255
    n = blocks.shape[0]
    d = blocks[:,3:6] - blocks[:,:3]
    vl = np.zeros((8*n,3))
    fl = np.zeros((6*n,4),dtype='int64')
    fcl = np.zeros((6*n,3))
    for k in range(n):
        vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
        fl[k*6:(k+1)*6,:] = f + k*8
        fcl[k*6:(k+1)*6,:] = clr[k,:]

    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
        pc.set_facecolor(fcl)
        h = ax.add_collection3d(pc)
        return h


def runtest(mapfile, start, goal, verbose = True):
    total_distance = 0.0
    # Instantiate a robot planner
    boundary, blocks = load_map(mapfile)
    # RP = RobotPlanner.RobotPlanner(boundary, blocks)
    # RP = AStarPlanner(boundary, blocks)
    RP = RTAAPlanner(boundary, blocks)
    RP.set_goal(goal)

    # Display the environment
    if verbose:
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    # Main loop
    robotpos = np.copy(start)
    numofmoves = 0
    success = True
    while True:

        # Call the robot planner
        t0 = tic()
        newrobotpos = RP.plan(robotpos, goal)
        movetime = max(1, np.ceil((tic()-t0)/2.0))
        print('move time: %d' % movetime)

        # See if the planner was done on time
        if movetime > 1:
            newrobotpos = robotpos-0.5 + np.random.rand(3)


        # Check if the commanded position is valid
        if sum((newrobotpos - robotpos)**2) > 1:
            print('ERROR: the robot cannot move so fast\n')
            success = False

        total_distance += sum((newrobotpos - robotpos)**2)

        if( newrobotpos[0] < boundary[0,0] or newrobotpos[0] > boundary[0,3] or \
                newrobotpos[1] < boundary[0,1] or newrobotpos[1] > boundary[0,4] or \
                newrobotpos[2] < boundary[0,2] or newrobotpos[2] > boundary[0,5] ):
            print('ERROR: out-of-map robot position commanded\n')
            success = False
        for k in range(blocks.shape[0]):
            if( newrobotpos[0] > blocks[k,0] and newrobotpos[0] < blocks[k,3] and \
                    newrobotpos[1] > blocks[k,1] and newrobotpos[1] < blocks[k,4] and \
                    newrobotpos[2] > blocks[k,2] and newrobotpos[2] < blocks[k,5] ):
                print('ERROR: collision_old... BOOM, BAAM, BLAAM!!!\n')
                success = False
                break
        if success is False:
            break

        # Update plot
        if verbose:
            x, y, z = zip(robotpos, newrobotpos)
            ax.plot(x, y, z, c='r')
            hs[0].set_xdata(robotpos[0])
            hs[0].set_ydata(robotpos[1])
            hs[0].set_3d_properties(robotpos[2])
            fig.canvas.flush_events()
            plt.show()

        # Make the move
        robotpos = newrobotpos
        numofmoves += 1

        # Check if the goal is reached
        if sum((robotpos-goal)**2) <= 0.1:
            break

    # plt.savefig(f"./images/RTAA-{mapfile.split('/')[-1].split('.')[0]}-{total_distance}-{numofmoves}.png")
    plt.savefig("./images/RTAA-" + str(mapfile.split('/')[-1].split('.')[0]) + "-" + str(total_distance) + "-" + str(
        numofmoves) + ".png")
    plt.close()

    return success, numofmoves



def test_single_cube():
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 6.0])
    # goal = np.array([-1, -1, -1])
    success, numofmoves = runtest('./maps/single_cube.txt', start, goal, True)
    print('Success: %r'%success)
    print('Number of Moves: %i'%numofmoves)


def test_maze():
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([12.0, 12.0, 5.0])
    success, numofmoves = runtest('./maps/maze.txt', start, goal, True)
    print('Success: %r'%success)
    print('Number of Moves: %i'%numofmoves)

def test_window():
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    success, numofmoves = runtest('./maps/window.txt', start, goal, True)
    print('Success: %r'%success)
    print('Number of Moves: %i'%numofmoves)

def test_tower():
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    success, numofmoves = runtest('./maps/tower.txt', start, goal, True)

def test_flappy_bird():
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    success, numofmoves = runtest('./maps/flappy_bird.txt', start, goal, True)
    print('Success: %r'%success)
    print('Number of Moves: %i'%numofmoves)

def test_room():
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    success, numofmoves = runtest('./maps/room.txt', start, goal, True)
    print('Success: %r'%success)
    print('Number of Moves: %i'%numofmoves)

def test_monza():
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    success, numofmoves = runtest('./maps/monza.txt', start, goal, True)
    print('Success: %r'%success)
    print('Number of Moves: %i'%numofmoves)

if __name__=="__main__":
    # test_single_cube()
    # test_window()
    # test_flappy_bird()
    # test_room()
    # test_tower()
    # test_monza()  # Large step + small step + inf_norm_distance
    test_maze()
  








