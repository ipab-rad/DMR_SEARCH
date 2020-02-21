import numpy as np

# Make a world to wander in
def make_world():

    N = 21

    world = np.zeros((N,N))
    world[5:10,5:8] = 1
    world[5:10,13:16] = 1
    world[15:16,5:16] = 1

    world[6:7,14:15] = 0
    world[7:8,6:7] = 0
    
    world[np.random.randint(0,N,20),np.random.randint(0,N,20)] = 1

    return world

class Robot:

    # Initialise an agent
    def __init__(self,p_move,p_sense,world):
        self.state = np.zeros(2,)#np.random.randint(0,world.shape[0],2)
        self.home = np.random.randint(0,world.shape[0],2)
        self.p_move = p_move
        self.p_sense = p_sense
        self.world =  world
        self.tracking_error = []
        self.homing_error = []

    # Agent moves are uncertain - p_move chance of responding to command, else remains still
    def move_noisy(self,action):
        if (np.random.rand() < self.p_move):
            self.state = self.state + action

        if self.state[0] < 0:
            self.state[0] = self.world.shape[0]+self.state[0]
        if self.state[0] >= self.world.shape[0]:
            self.state[0] = self.state[0]-self.world.shape[0]
        if self.state[1] < 0:
            self.state[1] = self.world.shape[1]+self.state[1]
        if self.state[1] >= self.world.shape[1]:
            self.state[1] = self.state[1]-self.world.shape[1]
        return self.state

    # Agent can sense environment - but sometimes sensing fails
    def sense(self):
        if (np.random.rand() < self.p_sense):
            return self.world[int(self.state[0]),int(self.state[1])]
        else:
            if self.world[int(self.state[0]),int(self.state[1])] == 0:
                return 1
            else:
                return 0

    # Command to move and sense
    def move_and_sense(self,action):
        self.state = self.move_noisy(action)
        return self.sense()

    # motion model assuming no noise
    def move_perfect(self,s,a):
        s = s+a

        if s[0] < 0:
            s[0] = self.world.shape[0]+s[0]
        if s[0] >= self.world.shape[0]:
            s[0] = s[0]-self.world.shape[0]
        if s[1] < 0:
            s[1] = self.world.shape[1]+s[1]
        if s[1] >= self.world.shape[1]:
            s[1] = s[1]-self.world.shape[1]

        return s.astype(int)

    def visualise_true_position(self):
        x = self.world.copy()
        x[int(self.state[0]),int(self.state[1])] = 0.5
        x[int(self.home[0]),int(self.home[1])] = 0.7

        return x
    
    # Returns error over trajectory (wrapped error)
    def get_mse(self,map_estimate):
        dx = np.abs(self.state[0]-map_estimate[0])
        dy = np.abs(self.state[1]-map_estimate[1])

        if (dx > 0.5*self.world.shape[0]):
            dx = self.world.shape[0] - dx

        if (dy > 0.5*self.world.shape[1]):
            dy = self.world.shape[1] - dy
        
        self.tracking_error.append((dx)**2+(dy)**2)
        return self.tracking_error
    
    # Gets home position
    def get_home_pos(self):
        return self.home
    
    # Gets utility of a state
    def get_utility(self,state):
        goal_pos = self.get_home_pos()
        dx = np.abs(state[0] - goal_pos[0])
        dy = np.abs(state[1] - goal_pos[1])
        if (np.abs(dx) > self.world.shape[0]/2.0):
            dx = self.world.shape[0] - dx
        if (np.abs(dy) > self.world.shape[0]/2.0):
            dy = self.world.shape[0] - dy
    
        return (dx**2 + dy**2)
        

