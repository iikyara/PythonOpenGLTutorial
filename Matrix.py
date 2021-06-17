import numpy as np
import math

class Matrix(object):
    def __init__(self, matrix=None, size=4):
        self.matrix = None
        self.size = size
        if isinstance(matrix, Matrix):
            self.matrix = matrix.matrix.copy()
        elif matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = np.zeros((size, size), dtype=np.float)

    def __setitem__(self, key, item):
        self.matrix[key] = item

    def __getitem__(self, key):
        return self.matrix[key]

    def __str__(self):
        return str(self.matrix.T) + str(self.matrix.dtype)

    def __repr__(self):
        return str(self.matrix.T) + str(self.matrix.dtype)

    def __add__(self, other):
        return Matrix(self.matrix + other.matrix)

    def __iadd__(self, other):
        return Matrix(self.matrix + other.matrix)

    @staticmethod
    def _mul(a, b):
        a = a.matrix if isinstance(a, Matrix) else a
        b = b.matrix if isinstance(b, Matrix) else b
        return Matrix(np.dot(b, a))

    def __mul__(self, other):
        return Matrix._mul(self, other)

    def __imul__(sefl, other):
        return Matrix._mul(self, other)

    def __rmul__(self, x):
        return Matrix._mul(x, self)

    def __lmul__(self, x):
        return Matrix._mul(self, x)

    #単位行列をロード
    def loadIdentity(self):
        self.matrix = np.zeros((self.size, self.size), dtype=np.float)
        for i in range(4):
            self.matrix[i, i] = 1.0

    #便利なクラスメソッドを定義
    @staticmethod
    def zero():
        mat = Matrix(np.zeros((self.size, self.size), dtype=np.float))
        return mat

    @staticmethod
    def identity():
        mat = Matrix()
        mat.loadIdentity()
        return mat

    @staticmethod
    def translate(x, y, z):
        mat = Matrix()
        mat.loadIdentity();
        mat[3] = np.array([x, y, z, 1.0], dtype=np.float)
        return mat

    @staticmethod
    def scale(x, y, z):
        mat = Matrix()
        mat.loadIdentity()
        mat[0, 0] = x
        mat[1, 1] = y
        mat[2, 2] = z
        return mat

    @staticmethod
    def rotate_x(x):
        mat = Matrix()
        mat.loadIdentity()
        mat[1, 1] = math.cos(x)
        mat[1, 2] = math.sin(x)
        mat[2, 1] = -math.sin(x)
        mat[2, 2] = math.cos(x)
        return mat

    @staticmethod
    def rotate_y(y):
        mat = Matrix()
        mat.loadIdentity()
        mat[0, 0] = math.cos(y)
        mat[0, 2] = -math.sin(y)
        mat[2, 0] = math.sin(y)
        mat[2, 2] = math.cos(y)
        return mat

    @staticmethod
    def rotate_z(z):
        mat = Matrix()
        mat.loadIdentity()
        mat[0, 0] = math.cos(z)
        mat[0, 1] = math.sin(z)
        mat[1, 0] = -math.sin(z)
        mat[1, 1] = math.cos(z)
        return mat

    # 任意の回転軸で回転
    @staticmethod
    def rotate(a, x, y, z):
        mat = Matrix()
        mat.loadIdentity()
        d = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        if d > 0:
            l = x / d; m = y / d; n = z / d
            l2 = l ** 2; m2 = m ** 2; n2 = n ** 2
            lm = l * m; mn = m * n; nl = n * l
            c = math.cos(a); c1 = 1.0 - c; s = math.sin(a)

            mat[0, 0] = (1 - l2) * c + l2
            mat[0, 1] = lm * c1 + n * s
            mat[0, 2] = nl * c1 - m * s
            mat[1, 0] = lm * c1 - n * s
            mat[1, 1] = (1 - m2) * c + m2
            mat[1, 2] = mn * c1 + l * s
            mat[2, 0] = nl * c1 + m * s
            mat[2, 1] = mn * c1 - l * s
            mat[2, 2] = (1 - n2) * c + n2

        return mat

    #回転軸を３次元ベクトルで指定（np.arrayで）
    @staticmethod
    def rotate_v(a, vec):
        return rotate(a, vec[0], vec[1], vec[2])

    @staticmethod
    def rotate_eular(phi, theta, psi):
        mat = Matrix()
        mat.loadIdentity()
        c_phi = math.cos(phi)
        s_phi = math.sin(phi)
        c_theta = math.cos(theta)
        s_theta = math.sin(theta)
        c_psi = math.cos(psi)
        s_psi = math.sin(psi)
        mat[0, 0] = c_psi * c_theta * c_phi - s_psi * s_phi
        mat[0, 1] = c_psi * c_theta * s_phi + s_psi * c_phi
        mat[0, 2] = -c_psi * s_theta
        mat[1, 0] = -s_psi * c_theta * c_phi - c_psi * s_phi
        mat[1, 1] = -s_psi * c_theta * s_phi + c_psi * c_phi
        mat[1, 2] = s_psi * s_theta
        mat[2, 0] = s_theta * c_phi
        mat[2, 1] = s_theta * s_phi
        mat[2, 2] = c_theta

        return mat

    @staticmethod
    def lookat(ex, ey, ez, gx, gy, gz, ux, uy, uz):
        tv = Matrix.translate(-ex, -ey, -ez)
        tx = ex - gx
        ty = ey - gy
        tz = ez - gz
        rx = uy * tz - uz * ty
        ry = uz * tx - ux * tz
        rz = ux * ty - uy * tx
        sx = ty * rz - tz * ry
        sy = tz * rx - tx * rz
        sz = tx * ry - ty * rx
        s2 = sx ** 2 + sy ** 2 + sz ** 2
        if s2 == 0:
            return tv;

        rv = Matrix()
        rv.loadIdentity()
        r = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        rv[0, 0] = rx / r
        rv[1, 0] = ry / r
        rv[2, 0] = rz / r

        s = math.sqrt(s2)
        rv[0, 1] = sx / s
        rv[1, 1] = sy / s
        rv[2, 1] = sz / s

        t = math.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
        rv[0, 2] = tx / t
        rv[1, 2] = ty / t
        rv[2, 2] = tz / t

        return rv * tv;

    def orthogonal(left, right, bottom, top, zNear, zFar):
        t = Matrix()
        t.loadIdentity()
        dx = right - left
        dy = top - bottom
        dz = zFar - zNear

        if dx != 0 and dy != 0 and dz != 0:
            t[0, 0] = 2 / dx
            t[1, 1] = 2 / dy
            t[2, 2] = -2 / dz
            t[3, 0] = -(right + left) / dx
            t[3, 1] = -(top + bottom) / dy
            t[3, 2] = -(zFar + zNear) / dz

        return t

    def frustum(left, right, bottom, top, zNear, zFar):
        t = Matrix()
        t.loadIdentity()
        dx = right - left
        dy = top - bottom
        dz = zFar - zNear

        if dx != 0 and dy != 0 and dz != 0:
            t[0, 0] = 2 * zNear / dx
            t[1, 1] = 2 * zNear / dy
            t[2, 0] = (right + left) / dx
            t[2, 1] = (top + bottom) / dy
            t[2, 2] = -(zFar + zNear) / dz
            t[2, 3] = -1.0
            t[3, 2] = -2 * zFar * zNear / dz
            t[3, 3] = 0.0

        return t

    def perspective(fovy, aspect, zNear, zFar):
        t = Matrix()
        t.loadIdentity()
        dz = zFar - zNear

        if dz != 0:
            t[1, 1] = 1 / math.tan(fovy * 0.5)
            t[0, 0] = t[1, 1] / aspect
            t[2, 2] = -(zFar + zNear) / dz
            t[2, 3] = -1.0
            t[3, 2] = -2 * zFar * zNear / dz
            t[3, 3] = 0.0

        return t

    #現在の変換行列に対して，法線ベクトルに実施すべき回転行列を返します．
    def getNormalMatrix(self):
        m = Matrix(size=3)
        m[0, 0] = self[1, 1] * self[2, 2] - self[1, 2] * self[2, 1]
        m[0, 1] = self[1, 2] * self[2, 0] - self[1, 0] * self[2, 2]
        m[0, 2] = self[1, 0] * self[2, 1] - self[1, 1] * self[2, 0]
        m[1, 0] = self[2, 1] * self[0, 2] - self[2, 2] * self[0, 1]
        m[1, 1] = self[2, 2] * self[0, 0] - self[2, 0] * self[0, 2]
        m[1, 2] = self[2, 0] * self[0, 1] - self[2, 1] * self[0, 0]
        m[2, 0] = self[0, 1] * self[1, 2] - self[0, 2] * self[1, 1]
        m[2, 1] = self[0, 2] * self[1, 0] - self[0, 0] * self[1, 2]
        m[2, 2] = self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        return m
