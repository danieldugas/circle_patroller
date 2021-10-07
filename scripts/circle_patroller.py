#!/usr/bin/env python
import numpy as np
import rospy
from pose2d import Pose2D, apply_tf
from geometry_msgs.msg import Twist, Point, PoseStamped
import threading
from visualization_msgs.msg import Marker, MarkerArray

def remove_python2_entries_from_sys_path():
    """ sourcing ros means adding its python2 paths to sys.path. This creates incompatibilities when
    importing tf, so we remove them """
    import sys
    if sys.version[0] == str(3):
        new_path = []
        for p in sys.path:
            if "python2" in p:
                print("REMOVING python2 entry from sys.path: {}".format(p))
                continue
            new_path.append(p)
        sys.path = new_path


remove_python2_entries_from_sys_path()
if True:
    from tf2_ros import TransformException
    import tf

class CirclePatroller(object):
    def __init__(self, args):
        self.args = args
        rospy.init_node('circle_patroller', anonymous=True)
        # consts
        # vars
        self.lock = threading.Lock() # for avoiding race conditions
        self.circle_center = None
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        self.tf_timeout = rospy.Duration(1.)
        # Localization Manager
        self.static_frame = rospy.get_param("~static_frame", "gmap")  # frame where walls don't move (gmap is not great, as the map can change) # noqa
        self.robot_frame = rospy.get_param("~robot_frame", "base_footprint")
        self.max_rot = rospy.get_param("~max_rot", 0.5)
        self.max_vel = rospy.get_param("~max_vel", 0.3)
        # rotation depends on sign: R > 0: trig, R < 0: clockwise (from top)
        self.circle_radius = rospy.get_param("~circle_radius", 1.)
        # Publishers
        self.patrol_pub = rospy.Publisher("/circle_patroller/patrol", MarkerArray, queue_size=1)
        self.traj_pub = rospy.Publisher("/circle_patroller/trajectory", MarkerArray, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # callback
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.set_circle_center, queue_size=1)
        # Timers
        rospy.Timer(rospy.Duration(0.1), self.check_position_callback)
        rospy.Timer(rospy.Duration(1.), self.publish_patrol_callback)
        # let's go.
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Keyboard interrupt - shutting down.")
            rospy.signal_shutdown('KeyboardInterrupt')

    def set_circle_center(self, msg):
        rospy.loginfo("circle center received (radius: {})".format(self.circle_radius))

        msg_in_static = self.get_x_in_static_tf(msg.header.frame_id, msg.header.stamp)
        if msg_in_static is None:
            return
        msg_in_static_p2d = Pose2D(msg_in_static)

        center_in_msg = np.array([msg.pose.position.x, msg.pose.position.y])
        center_in_static = apply_tf(center_in_msg[None, :], msg_in_static_p2d)[0]

        with self.lock:
            # publish current waypoint
            self.circle_center = center_in_static
            self.publish_patrol()

    def publish_patrol_callback(self, event=None):
        with self.lock:
            self.publish_patrol()

    def check_position_callback(self, event=None):
        if self.circle_center is None:
            rospy.logwarn_once("Waiting for circle center.")
            return

        now = rospy.Time.now()

        robot_in_static = self.get_robot_in_static_tf(now)
        if robot_in_static is None:
            return
        robot_in_static_p2d = Pose2D(robot_in_static)

        # bring data into scope
        C = self.circle_center
        c_is_trig = self.circle_radius > 0
        cr = self.circle_radius

        max_rot = self.max_rot
        max_vel = self.max_vel

        # first we quantify the error between what we want and what we have
        # radius error, direction error

        # robot pose
        robot_xy = robot_in_static_p2d[:2]
        robot_th = robot_in_static_p2d[2]
        direction = np.array([np.cos(robot_th), np.sin(robot_th)])
        delta = robot_xy - C # robot pose in circle frame
        radius = np.linalg.norm(delta)

        # closest point
        delta_norm = delta / radius
        cp_xy = C + delta_norm * cr
        left_90 = np.array([[0, -1],
                            [1, 0]])
        right_90 = np.array([[0, 1],
                             [-1, 0]])
        rot = left_90 if c_is_trig else right_90
        cp_direction = np.dot(rot, delta_norm) # direction at closest point

        # now we think in circles. The smallest circle our robot can do at max speed is
        # f(max speed, max rot)
        # to make a smaller circle, speed has to go down
        # to make a bigger circle, rot has to go down

        # if we are going a completely wrong direction dot(cd, d) < 0,
        # we want to make the smallest possible circle towards a good direction
        # right if center is on our right, left if center is on our left

        # otherwise we need to decide what radius trajectory we want,
        # which takes us back to our circle / direction
        # case 0: pointing towards LP: go straight (R -> inf)
        # case 1: pointing aft of LP: go towards circle (R -> -CR)
        # case 2: pointing behind LP: go with circle (R -> CR)
        # case 3: pointing away from LP: go to/with circle (R -> +-0)

        # leading point
        LEAD_RATIO = 0.3
        RADIUS_DECAY = 0.05
        lead_length = LEAD_RATIO * cr
        lp_xy = cp_xy + cp_direction * lead_length
        delta_to_lp = lp_xy - robot_xy
        dist_to_lp = np.linalg.norm(delta_to_lp)
        direction_to_lp = delta_to_lp / dist_to_lp

        # phi is directly related to the trajectory radius. phi [-1, 1]
        # phi = 0: going straight to lp -> R = inf
        # phi = [-pi/2, pi/2]: pick circle radius which passes through LP
        # phi = +=pi/2: perpendicular to lp -> R = |LP - RXY| / 2
        # phi > pi/2: going away from lp -> R = +=0
        phi = angle_between_vectors(direction, direction_to_lp)
        if abs(phi) > 0.5:
            traj_radius = 0.001 * np.sign(phi)
        else:
            traj_radius = dist_to_lp / 2. / np.sin(phi)
        traj_radius = - RADIUS_DECAY * cr / phi

        # robot limits: at max vel radius is given. For smaller radius, need to reduce vel
        traj_vel = max_vel
        nominal_radius = max_vel / max_rot
        if abs(traj_radius) < abs(nominal_radius):
            traj_vel = abs(traj_radius) * max_rot
        traj_rot = traj_vel / traj_radius
        direction_to_traj_center = np.dot(left_90, direction)
        traj_center = robot_xy + direction_to_traj_center * traj_radius

        markers = MarkerArray()
        # trajectory
        N = 20 # subdivisions
        th = np.linspace(0, 2*np.pi, N)
        xy = traj_radius * np.array([np.cos(th), np.sin(th)]).T
        points = traj_center + xy
        color = [1., 0.5, 0., 1.] # rgba orange
        marker = self.path_as_marker(points, self.static_frame, 0.02, "circle_patrol", id_=0, color=color)
        markers.markers.append(marker)
        # leading point
        marker = self.point_as_marker(lp_xy, self.static_frame, 0.02, "circle_patrol", id_=1, color=color)
        markers.markers.append(marker)

        self.traj_pub.publish(markers)

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = traj_vel
        cmd_vel_msg.angular.z = traj_rot
        self.cmd_vel_pub.publish(cmd_vel_msg)

    def get_robot_in_static_tf(self, time):
        return self.get_x_in_static_tf(self.robot_frame, time)

    def get_x_in_static_tf(self, child_frame, time):
        try:
            tf_info = [self.static_frame, child_frame, time]
            self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
            tf_ = self.tf_listener.lookupTransform(*tf_info)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                TransformException) as e:
            rospy.logwarn_throttle(60, "[{}.{}] tf for time {}.{} not found: {}".format(
                rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
            return None
        return tf_

    def publish_patrol(self):
        if self.circle_center is None:
            return
        markers = MarkerArray()
        # trajectory
        N = 20 # subdivisions
        th = np.linspace(0, 2*np.pi, N)
        xy = self.circle_radius * np.array([np.cos(th), np.sin(th)]).T
        points = self.circle_center + xy
        color = [0., 0., 1., 1.] # rgba blue
        marker = self.path_as_marker(points, self.static_frame, 0.02, "circle_patrol", color=color)
        markers.markers.append(marker)
        self.patrol_pub.publish(markers)

    def path_as_marker(self, path_xy, frame, scale, namespace, time=None, color=None, id_=0, z=0):
        marker = Marker()
        time = rospy.Time.now() if time is None else time
        marker.header.stamp.secs = time.secs
        marker.header.stamp.nsecs = time.nsecs
        marker.header.frame_id = frame
        marker.ns = namespace
        marker.id = id_
        marker.type = 4 # LINE_STRIP
        marker.action = 0
        s = scale
        marker.scale.x = s
        marker.scale.y = s
        marker.scale.z = s
        if color is None:
            marker.color.g = 1.
            marker.color.a = 0.5
        else:
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
        marker.points = [Point(xy[0], xy[1], z) for xy in path_xy]
        return marker

    def point_as_marker(self, point_xy, frame, scale, namespace,
                        time=None,
                        marker_type=2, # SPHERE
                        text='',
                        color=[0., 0., 1., 1.], # rgba
                        id_=0,
                        z=0.,
                        ):
        if time is None:
            time = rospy.Time.now()
        if frame is None:
            frame = self.static_frame
        marker = Marker()
        marker.header.stamp.secs = time.secs
        marker.header.stamp.nsecs = time.nsecs
        marker.header.frame_id = frame
        marker.ns = namespace
        marker.id = id_
        marker.type = marker_type
        marker.action = 0
        s = scale
        marker.pose.position.x = point_xy[0]
        marker.pose.position.y = point_xy[1]
        marker.pose.position.z = z
        marker.scale.x = s
        marker.scale.y = s
        marker.scale.z = s
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.text = text
        return marker

def angle_between_vectors(va, vb):
    """ result: th(va) - th(vb) in radians """
    th_va = np.arctan2(va[1], va[0])
    th_vb = np.arctan2(vb[1], vb[0])
    return angle_difference_rad(th_va, th_vb)

def angle_difference_rad(angle, target_angle):
    """     / angle
           /
          / d
         /)___________ target
    result: angle - target_angle
    """
    delta_angle = angle - target_angle
    delta_angle = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))  # now in [-pi, pi]
    return delta_angle

def parse_args():
    import argparse
    # Arguments
    parser = argparse.ArgumentParser(description='ROS node for patrolling')
    parser.add_argument('--hz',
                        action='store_true',
                        help='if set, prints planner frequency to script output',
                        )
    ARGS, unknown_args = parser.parse_known_args()

    # deal with unknown arguments
    # ROS appends some weird args, ignore those, but not the rest
    if unknown_args:
        non_ros_unknown_args = rospy.myargv(unknown_args)
        if non_ros_unknown_args:
            print("unknown arguments:")
            print(non_ros_unknown_args)
            parser.parse_args(args=["--help"])
            raise ValueError
    return ARGS


if __name__ == "__main__":
    args = parse_args()
    circle_patroller = CirclePatroller(args)
