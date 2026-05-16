#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from bosdyn_api_msgs.msg import (
    GripperCommandRequest,
    GripperCommandRequestOneOfCommand,
    ClawGripperCommandRequest,
    ScalarTrajectory,
    ScalarTrajectoryPoint,
    PositionalInterpolation,
)
from spot_schemas_interfaces.srv import GraspHand, ReleaseHand, Rotate, Walk


class SpotSchemaServiceNode(Node):
    def __init__(self, parameter_defaults=None):
        super().__init__('spot_schema_service')

        defaults = parameter_defaults or {}

        self.declare_parameter('output_cmd_topic', defaults.get('output_cmd_topic', '/spot_ai/cmd_vel_raw'))
        self.declare_parameter('gripper_cmd_topic', defaults.get('gripper_cmd_topic', '/spot/gripper/command'))
        self.declare_parameter('publish_hz', defaults.get('publish_hz', 20.0))

        output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        gripper_cmd_topic = str(self.get_parameter('gripper_cmd_topic').value)
        publish_hz = max(1.0, float(self.get_parameter('publish_hz').value))

        self.cmd_vel_pub = self.create_publisher(Twist, output_cmd_topic, 10)
        self.gripper_pub = self.create_publisher(GripperCommandRequest, gripper_cmd_topic, 10)

        self.create_service(Walk, '/schemas/walk', self.handle_walk)
        self.create_service(Walk, '/schemas/walking', self.handle_walk)
        self.create_service(Rotate, '/schemas/rotate', self.handle_rotate)
        self.create_service(GraspHand, '/schemas/grasp_hand', self.handle_grasp)
        self.create_service(ReleaseHand, '/schemas/release_hand', self.handle_release)

        self.motion_twist = Twist()
        self.motion_end_sec = None
        self.motion_timer = self.create_timer(1.0 / publish_hz, self._motion_timer)

        self.get_logger().info(
            f'Spot Schema Service Node ready: cmd={output_cmd_topic} gripper={gripper_cmd_topic}'
        )

    def handle_walk(self, request, response):
        distance = request.distance_meters
        speed = request.speed_mps

        if distance <= 0.0:
            response.status.success = False
            response.status.message = 'distance_meters must be positive'
            self.get_logger().warn(response.status.message)
            return response

        if speed <= 0.0:
            response.status.success = False
            response.status.message = 'speed_mps must be positive'
            self.get_logger().warn(response.status.message)
            return response

        if distance > 10.0:
            response.status.success = False
            response.status.message = 'distance exceeds safety limit (10m)'
            self.get_logger().warn(response.status.message)
            return response

        duration = abs(distance) / speed
        self.get_logger().info(f'Starting walk: {distance:.2f} m at {speed:.2f} m/s for {duration:.2f} s')

        twist = Twist()
        twist.linear.x = float(speed if distance >= 0.0 else -speed)
        self._start_motion(twist, duration)

        response.status.success = True
        response.status.message = f'Walking {distance:.2f} m'
        return response

    def handle_rotate(self, request, response):
        angle_degrees = request.angle_degrees
        angular_speed_rps = request.angular_speed_rps

        if angular_speed_rps <= 0.0:
            response.status.success = False
            response.status.message = 'angular_speed_rps must be positive'
            self.get_logger().warn(response.status.message)
            return response

        if abs(angle_degrees) > 360.0:
            response.status.success = False
            response.status.message = 'angle_degrees exceeds limit (±360)'
            self.get_logger().warn(response.status.message)
            return response

        duration = abs(angle_degrees) * (3.141592653589793 / 180.0) / angular_speed_rps
        twist = Twist()
        twist.angular.z = angular_speed_rps if angle_degrees >= 0.0 else -angular_speed_rps

        self.get_logger().info(
            f'Starting rotate: {angle_degrees:.1f} deg at {angular_speed_rps:.2f} rad/s for {duration:.2f} s'
        )
        self._start_motion(twist, duration)

        response.status.success = True
        response.status.message = f'Rotating {angle_degrees:.1f}°'
        return response

    def _build_claw_command(self, position: float) -> GripperCommandRequest:
        req = GripperCommandRequest()
        req.command.command_choice = GripperCommandRequestOneOfCommand.CLAW_GRIPPER_COMMAND_SET
        claw = ClawGripperCommandRequest()
        claw.disable_force_on_contact = False
        claw.maximum_open_close_velocity = Float64()
        claw.maximum_open_close_velocity.data = 0.5
        claw.maximum_open_close_acceleration = Float64()
        claw.maximum_open_close_acceleration.data = 0.3
        claw.maximum_torque = Float64()
        claw.maximum_torque.data = 50.0

        point = ScalarTrajectoryPoint()
        point.point = float(position)
        point.velocity = Float64()
        point.velocity.data = 0.0
        point.time_since_reference.sec = 1
        point.time_since_reference.nanosec = 0
        claw.trajectory = ScalarTrajectory()
        claw.trajectory.points = [point]
        claw.trajectory.interpolation = PositionalInterpolation.POS_INTERP_LINEAR

        req.command.claw_gripper_command = claw
        return req

    def handle_grasp(self, request, response):
        self.get_logger().info(f'GraspHand request: strength={request.strength}')
        gripper_msg = self._build_claw_command(position=0.0)
        self.gripper_pub.publish(gripper_msg)

        response.status.success = True
        response.status.message = 'Gripper close command published'
        return response

    def handle_release(self, request, response):
        self.get_logger().info('ReleaseHand request')
        gripper_msg = self._build_claw_command(position=1.0)
        self.gripper_pub.publish(gripper_msg)

        response.status.success = True
        response.status.message = 'Gripper open command published'
        return response

    def _start_motion(self, twist: Twist, duration: float) -> None:
        self.motion_twist = twist
        self.motion_end_sec = self.get_clock().now().nanoseconds / 1e9 + max(0.05, float(duration))
        self.cmd_vel_pub.publish(twist)

    def _motion_timer(self) -> None:
        if self.motion_end_sec is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now >= self.motion_end_sec:
            self._stop_motion()
        else:
            self.cmd_vel_pub.publish(self.motion_twist)

    def _stop_motion(self) -> None:
        self.motion_end_sec = None
        self.motion_twist = Twist()
        self.cmd_vel_pub.publish(self.motion_twist)


def main(args=None, parameter_defaults=None):
    rclpy.init(args=args)
    node = SpotSchemaServiceNode(parameter_defaults=parameter_defaults)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
