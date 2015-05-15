Kinematics for arm manipulators - Pick it
=========================================
Computes forward and inverse kinematics of robotic arms including Debra's arms.

## Demo
A visual demo using pygame gets the position of your cursor, sets it as tool position and allows you to see the arm reconstructed through inverse kinematics.
To run it, you need to install pygame (see http://www.pygame.org/wiki/CompileUbuntu), then run:
```sh
python3 debra_arm_inverse_kinematics_visualiser.py
```

You can also test the path planner by running:
```sh
python3 debra_arm_planner_visualiser.py
```

## Usage
This module provides:
- Forward and Inverse kinematics computation for 2dof planar arms (Scara) and 4dof arms we use at CVRA (DebraArm)
- Trajectory generation in joint space that are time-optimal
- Trajectory generation in robot space that move the arms in straight lines
In both planning scenarios, the axis involved are synchronised so they arrive at the same time.

**All following is for a DebraArm object, but Scara objects works similarly**

### Start
To start, you will need to instantiate an object from the arm class you want as in:
```python
arm = DebraArm.DebraArm(
        l1=1.0,
        l2=1.0,
        theta1_constraints = JointMinMaxConstraint(-pi,pi, -2,2, -1,1),
        theta2_constraints = JointMinMaxConstraint(-pi,pi, -2,2, -1,1),
        theta3_constraints = JointMinMaxConstraint(-pi,pi, -2,2, -1,1),
        z_constraints = JointMinMaxConstraint(0,1, -1,1, -1,1),
        q0 = JointSpacePoint(0,0,0,0),
        origin = Vector3D(0,0,0),
        flip_x = FLIP_RIGHT_HAND)
```
`l1` and `l2` are the lengths of the two principal links, constraints on the joints are defined using namedtuples defined in the `Datatypes` module, `q0` is the initial position of the actuators, `origin` is the position of the fixation of the base of the arm in cartesian coordinates and `flip_x` defines if the arm is a right or left arm it therefore defines the default orientation of the elbow.

### Forward / Inverse kinematics
To use the kinematics computation you can set a tool position and get back the joint configuration (by default the elbow is 'behind', so theta2 > 0)
```python
tool = RobotSpacePoint(x = 0.99 * (L1 + L2), y=0, z=0, gripper_hdg=0)
joints = arm.inverse_kinematics(tool)
```
Or set a joint configuration and get back the position of the tool
```python
joints = JointSpacePoint(theta1=0, theta2=0, z=0, theta3=0)
tool = arm.forward_kinematics(joints)
```

### Path planning in joint space
Planning a trajectory for the arm in joint space guarantees a time optimal movement but not a straight line path.
To use this you can simply call
```python
points_th1, points_th2, points_z, points_th3 = \
    arm.get_path(start_pos = RobotSpacePoint(1.0, 0.5, 0, 0),
                 start_vel = RobotSpacePoint(0, 0, 0, 0),
                 target_pos = RobotSpacePoint(0.5, 0.5, 0.2, 0),
                 target_vel = RobotSpacePoint(0, 0, 0, 0),
                 delta_t = 0.01)
```

### Path planning in robot space
Planning a trajectory in robot space guaratees that the tool will move in a straight line but the trajectory is not necessarily time optimal, also it take 10 times more time than planning in joint space.
To use thos you can simply call
```python
points_th1, points_th2, points_z, points_th3 = \
    arm.get_path_xyz(start_pos = RobotSpacePoint(1.0, 0.5, 0, 0),
                     start_vel = RobotSpacePoint(0, 0, 0, 0),
                     target_pos = RobotSpacePoint(0.5, 0.5, 0.2, 0),
                     target_vel = RobotSpacePoint(0, 0, 0, 0),
                     DELTA_T,
                     'joint')
```
