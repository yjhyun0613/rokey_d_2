from rclpy.node import Node

class publisher(Node):
    def __init__(self):
        self.pub = self.create_publisher()