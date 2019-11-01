from my_collision.vector3 import Vector3


class Ray:

    def __init__(self, o:Vector3, d:Vector3):
        self.origin = o
        self.direction = d
        self.inv_direction = Vector3(1/d.x if d.x else float('inf'), 1/d.y if d.y else float('inf'), 1/d.z if d.z else float('inf'))
        self.sign = [0, 0, 0]
        self.sign[0] = self.inv_direction.x < 0
        self.sign[1] = self.inv_direction.y < 0
        self.sign[2] = self.inv_direction.z < 0