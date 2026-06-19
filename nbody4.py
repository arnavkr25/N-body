'''with Barnes-Hut, Runge-Kutta 4th order and energy tracking'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



#constants....
G = 1   # grav const
dt = 0.01   # time step
ep = 0.1    #epsilon to make denominator non-zero
theta = 0.5  #barnes-hut opening angle

energy_hist = []
time_hist = []

#body constructor
class body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.pos = np.array(position, dtype = float)
        self.vel = np.array(velocity, dtype = float)
        self.traj = []

#body creation
bodies = []
np.random.seed(42)

central_mass = 5000
bodies.append(body(central_mass, [0,0], [0,0]))

for _ in range(50):

    r = np.random.uniform(5, 50)
    phi = np.random.uniform(0, 2*np.pi)

    x = r*np.cos(phi)
    y = r*np.sin(phi)

    v = np.sqrt(G * central_mass / r)

    vx = -v*np.sin(phi)
    vy =  v*np.cos(phi)

    bodies.append(body(1, [x,y], [vx,vy]))
          

fig, (ax, ax_E) = plt.subplots(1, 2, figsize = (14,6))

ax_E.set_title("Total Energy")
ax_E.set_xlabel("Time")
ax_E.set_ylabel("Energy")


class QuadNode:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=float)
        self.size = size
        self.mass = 0
        self.com = np.zeros(2)
        self.body_index = None
        self.children = []
    
    def contains(self, point):
        half = self.size / 2
        return ((self.center[0] - half <= point[0] < self.center[0] + half) and (self.center[1] - half <= point[1] < self.center[1] + half))

    def subdivide(self):
        h = self.size / 4
        s = self.size / 2
        offsets = [
            [-h, -h],
            [ h, -h],
            [-h,  h],
            [ h,  h]
        ]

        for off in offsets:
            child_center = self.center + off
            self.children.append(QuadNode(child_center, s))
    
    def insert(self, index, positions, masses):
        pos = positions[index]

        if not self.contains(pos):
            return False
        
        #for empty node
        if (self.body_index is None) and (len(self.children) == 0) and (self.mass == 0):
            self.body_index = index
            self.mass = masses[index]
            self.com = pos.copy()
            return True
        
        #subdivision if needed
        if len(self.children) == 0:
            self.subdivide()

            if self.body_index is not None:
                old = self.body_index
                self.body_index = None

                for child in self.children:
                    if child.insert(old, positions, masses):
                        break
        
        #inserting new body
        for child in self.children:
            if child.insert(index, positions, masses):
                break
        
        total_mass = self.mass + masses[index]  #update com + mass
        self.com = (self.com * self.mass + pos * masses[index]) / total_mass
        self.mass = total_mass

        return True
    
def build_tree(positions):
    positions_np = np.array(positions)

    xmin = np.min(positions_np[:,0])
    xmax = np.max(positions_np[:,0])

    ymin = np.min(positions_np[:,1])
    ymax = np.max(positions_np[:,1])

    size = max(xmax - xmin, ymax - ymin) + 1

    center = [(xmax + xmin)/2, (ymax + ymin)/2]

    root = QuadNode(center, size)

    masses = [b.mass for b in bodies]

    for i in range(len(positions)):
        root.insert(i, positions, masses)
    
    return root

def bh_force(node, i, positions):
    if node.mass == 0:
        return np.zeros(2)
    
    if (node.body_index == i) and (len(node.children) == 0):
        return np.zeros(2)
    
    r = node.com - positions[i]
    dist = np.linalg.norm(r) + ep

    #barnes-hut criterion
    if (len(node.children) == 0) or ((node.size / dist)< theta):
        return G * node.mass * r / dist**3
    
    acc = np.zeros(2)

    for child in node.children:
        acc += bh_force(child, i, positions)
    
    return acc
    
        

    #array for accelerations
def computeacc(positions):
    tree = build_tree(positions)
    accs = []

    for i in range(len(bodies)):
        acc = bh_force(tree, i, positions)
        accs.append(acc)
        
    return accs

#to compute total energy for energy tracking plot
def computeenergy():
    KE = 0
    PE = 0

    #for KE
    for body in bodies:
        vk = np.linalg.norm(body.vel)
        KE += 0.5 * body.mass * vk**2
    
    #for PE
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            r = bodies[j].pos - bodies[i].pos
            dist = np.linalg.norm(r) + ep
            PE -= (G * bodies[i].mass * bodies[j].mass) / dist
    
    total_E = KE + PE
    return total_E

#simulation loop
def update(frame):
    positions = []
    velocities = []

    for _ in bodies:
        positions.append(_.pos.copy())
        velocities.append(_.vel.copy())
    
    #k1
    a1 = computeacc(positions)
    k1_v = a1
    k1_r = velocities

    #k2
    pos2 = []
    vel2 = []
    for i in range(len(bodies)):
        newp2 = positions[i] + (0.5 * dt * k1_r[i])
        pos2.append(newp2)
        newv2 = velocities[i] + (0.5 *dt * k1_v[i])
        vel2.append(newv2)
    
    a2 = computeacc(pos2)
    k2_v = a2
    k2_r = vel2

    #k3
    pos3 = []
    vel3 = []
    for i in range(len(bodies)):
        newp3 = positions[i] + (0.5 * dt * k2_r[i])
        pos3.append(newp3)
        newv3 = velocities[i] + (0.5 *dt * k2_v[i])
        vel3.append(newv3)
    
    a3 = computeacc(pos3)
    k3_v = a3
    k3_r = vel3

    #k4
    pos4 = []
    vel4 = []
    for i in range(len(bodies)):
        newp4 = positions[i] + (dt * k3_r[i])
        pos4.append(newp4)
        newv4 = velocities[i] + (dt * k3_v[i])
        vel4.append(newv4)
    
    a4 = computeacc(pos4)
    k4_v = a4
    k4_r = vel4

    #using k1, k2, k3, k4 for RK4 update
    for i, body in enumerate(bodies):
        body.pos += ((k1_r[i]) + (2 * k2_r[i]) + (2 * k3_r[i]) + (k4_r[i])) * (dt/6)
        body.vel += ((k1_v[i]) + (2 * k2_v[i]) + (2 * k3_v[i]) + (k4_v[i])) * (dt/6)

        body.traj.append(body.pos.copy())
        if (len(body.traj) > 500):
            body.traj.pop(0)

    total_E = computeenergy()
    energy_hist.append(total_E)
    time_hist.append(frame * dt)

    ax.clear()

    for body in bodies:
        #draw body
        ax.scatter(body.pos[0], body.pos[1])

        #draw traj
        # traj = np.array(body.traj)
        # if len(traj) > 1:
        #     ax.plot(traj[:,0], traj[:,1], linewidth=1)
        
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    ax.set_aspect('equal')

    ax_E.clear()
    ax_E.plot(time_hist, energy_hist, linewidth=2)
    ax_E.set_xlim(0, max(10, frame*dt))

    ax_E.relim()
    ax_E.autoscale_view()

    ax_E.set_title("Energy Conservation")
    ax_E.set_xlabel("Time")
    ax_E.set_ylabel("Total Energy")

    ax_E.grid(True, linestyle='--', alpha=0.5)
    
    ax_E.ticklabel_format(useOffset=False, style='plain')

ani = FuncAnimation(fig, update, frames=10000, interval=10)
plt.show()


#very laggy and slow
#needs crazy optimization