import random

class Vector2:
    dim = 2

    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            return -1

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return self.__class__(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self.__class__(self.x * other, self.y * other)

    def __truediv__(self, other):
        return self.__class__(self.x / other, self.y / other)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

    def clone(self):
        return self.__class__(x=self.x, y=self.y)

    def length(self):
        return (self.x * self.x + self.y * self.y) ** 0.5

    @classmethod
    def dot(cls, vec1, vec2):
        return cls.__class__(vec1.x * vec2.x, vec1.y * vec2.y)

    @classmethod
    def randint(cls, vec1, vec2=None):
        if vec2 is None:
            return cls(
                random.randint(int(vec1.x)),
                random.randint(int(vec1.y))
            )
        else:
            return cls(
                random.randint(int(vec1.x), int(vec2.x)),
                random.randint(int(vec1.y), int(vec2.y))
            )

class Vector3:
    dim = 3

    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        #return "({:.3f}, {:.3f}, {.3f})".format(self.x, self.y, self.z)
        return "({!r}, {!r}, {!r})".format(self.x, self.y, self.z)

    def __str__(self):
        #return "({:.3f}, {:.3f}, {.3f})".format(self.x, self.y, self.z)
        return "({!r}, {!r}, {!r})".format(self.x, self.y, self.z)

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        else:
            return -1

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value

    def toList(self):
        return [self.x, self.y, self.z]

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return self.__class__(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__class__(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return self.__class__(self.x / other, self.y / other, self.z / other)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def __str__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

    def clone(self):
        return self.__class__(x=self.x, y=self.y, z=self.z)

    def length(self):
        return (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5

    @classmethod
    def dot(cls, vec1, vec2):
        return cls.__class__(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z)

    @classmethod
    def randint(cls, vec1, vec2=None):
        if vec2 is None:
            return cls(
                random.randint(int(vec1.x)),
                random.randint(int(vec1.y)),
                random.randint(int(vec1.z))
            )
        else:
            return cls(
                random.randint(int(vec1.x), int(vec2.x)),
                random.randint(int(vec1.y), int(vec2.y)),
                random.randint(int(vec1.z), int(vec2.z))
            )

    @classmethod
    def randfloat(cls, vec1, vec2=None):
        if vec2 is None:
            return cls(
                random.uniform(0, vec1.x),
                random.uniform(0, vec1.y),
                random.uniform(0, vec1.z)
            )
        else:
            return cls(
                random.uniform(vec1.x, vec2.x),
                random.uniform(vec1.y, vec2.y),
                random.uniform(vec1.z, vec2.z)
            )
