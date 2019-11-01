from my_collision.box import Box
from my_collision.vector3 import Vector3
from my_collision.ray import Ray

if __name__ == '__main__':
    box = Box(Vector3(0.1, 0.1, 0.1), Vector3(1, 1, 1))
    o = Vector3(0, 0, 0)
    d = Vector3(0.1, 0.1, 0.1)   # direction = end_point - start_point
    segment = Ray(o, d)
    print(box.intersect(segment, 0.0, 1.0 + 1e-10)) # Add an epsilon term to exclude the case when the ray touches the surface of the box