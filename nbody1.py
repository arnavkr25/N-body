import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#constants....
G = 1   # grav const
dt = 0.01   # time step

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

#simulation loop
def update(frame):

    forces = []
    for _ in bodies:
        forces.append(np.zeros(2))

    for i in range(len(bodies)):
        for j in range(len(bodies)):
            if (i!=j):
                r = bodies[j].pos - bodies[i].pos
                dist = np.linalg.norm(r) + 1e-5    #dist = (x2 + y2)^1/2 + 1e-5 for adjustment to avoid 0
                force = (G * bodies[i].mass * bodies[j].mass * r) / dist**3
                forces[i] += force
            
    #updating velocities and pos
    for i, body in enumerate(bodies):
        acc = forces[i] / body.mass
        body.pos += body.vel * dt + 0.5 * acc * dt**2
        body.vel += acc * dt
        body.traj.append(body.pos.copy())  #copy() is necessary to it doesnt overwrite prev values
        if (len(body.traj) > 500):         #limit traj length for perfromance
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
