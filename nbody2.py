'''with Runge-Kutta 4th order'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#constants....
G = 1   # grav const
dt = 0.01   # time step
ep = 1e-5    #epsilon to make denominator non-zero

#body constructor
class body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.pos = np.array(position, dtype = float)
        self.vel = np.array(velocity, dtype = float)
        self.traj = []

#body creation
sun = body(1000, [0,0], [0,0])

#bodies array
bodies = [sun,
          body(1, [10,0], [0,8]),
          body(1, [-10,0], [0,-8]),
          body(0.5, [0,15], [-7,0])]
          

fig, ax = plt.subplots()

#array for accelerations
def computeacc(positions):
    accs = []
    for i in range(len(bodies)):
        acc = np.zeros(2)
        for j in range(len(bodies)):
            if (i!=j):
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r) + ep
                acc += G * bodies[j].mass * r / dist**3
        accs.append(acc)
    return accs

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

    ax.clear()

    for body in bodies:
        #draw body
        ax.scatter(body.pos[0], body.pos[1])

        #draw traj
        traj = np.array(body.traj)
        if len(traj) > 1:
            ax.plot(traj[:,0], traj[:,1], linewidth=1)
        
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.set_aspect('equal')

ani = FuncAnimation(fig, update, frames=1000, interval=10)
plt.show()

#plotting only
# for body in bodies:
#     traj = np.array(body.traj)
#     plt.plot(traj[:,0], traj[:,1])    #plot x vs y

# plt.scatter(0, 0, color='yellow', label='Sun')
# plt.legend()
# plt.axis('equal')  #ensures 1 unit on x = 1 unit on y
# plt.show()
