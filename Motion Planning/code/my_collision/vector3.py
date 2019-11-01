from math import sqrt


class Vector3:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def length(self):
        return sqrt(self.x**2 + self.y ** 2 + self.z ** 2)

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other=None):
        if other:
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vector3(-self.x, -self.y, -self.z)

    def __mul__(self, other):
        return Vector3(self.x * other.x + self.y * other.y + self.z * other.z)

    def __truediv__(self, denominator):
        return Vector3(self.x /denominator, self.y / denominator, self.z / denominator)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.x <= other.x and self.y <= other.y and self.z <= other.z

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y and self.z < other.z

    def normalize(self):
        length = self.length()
        if length:
            self.x /= length
            self.y /= length
            self.z /= length