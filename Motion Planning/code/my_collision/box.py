from my_collision.vector3 import Vector3

class Box:
    def __init__(self, pos_min:Vector3, pos_max:Vector3):
        assert pos_min < pos_max
        self.parameters = [pos_min, pos_max]

    def intersect(self, r, t0=0.0, t1=1.0+1e-10):
        tmin = (self.parameters[r.sign[0]].x - r.origin.x) * r.inv_direction.x
        tmax = (self.parameters[1 - r.sign[0]].x - r.origin.x) * r.inv_direction.x
        tymin = (self.parameters[r.sign[1]].y - r.origin.y) * r.inv_direction.y
        tymax = (self.parameters[1 - r.sign[1]].y - r.origin.y) * r.inv_direction.y
        if (tmin > tymax) or (tymin > tmax):
            return False
        if (tymin > tmin):
            tmin = tymin
        if (tymax < tmax):
            tmax = tymax
        tzmin = (self.parameters[r.sign[2]].z - r.origin.z) * r.inv_direction.z
        tzmax = (self.parameters[1 - r.sign[2]].z - r.origin.z) * r.inv_direction.z
        if (tmin > tzmax) or (tzmin > tmax):
            return False
        if (tzmin > tmin):
            tmin = tzmin
        if (tzmax < tmax):
            tmax = tzmax
        return (tmin < t1 and tmax > t0)