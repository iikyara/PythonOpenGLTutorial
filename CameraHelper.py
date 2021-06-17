import math
from Matrix import *
from Core import *

class CameraHelper:
    def __init__(self, context, camera,
        rotate_button=0, move_button=1, rotate_speed=0.01, scale_speed=0.1,
        reverse_rotate=False, reverse_move=True
    ):
        self.camera = camera
        self.context = context
        self.rotate_button = rotate_button
        self.move_button = move_button
        self.rotate_speed = rotate_speed
        self.scale_speed = scale_speed
        self.reverse_rotate = reverse_rotate
        self.reverse_move = reverse_move

        self._pre_mouse_pos = (0, 0)
        self._rotate_button_is_down = False
        self._move_button_is_down = False

        self.context.on_mouseButton.append(self.on_mouseButton)
        self.context.on_mousePosition.append(self.on_mousePosition)
        self.context.on_mouseScroll.append(self.on_mouseScroll)

    def on_mouseButton(self, window, button, action, mods):
        if button is self.rotate_button:
            self._rotate_button_is_down = action is 1
        if button is self.move_button:
            self._move_button_is_down = action is 1

    def on_mousePosition(self, window, x, y):
        if self._rotate_button_is_down:
            self.camera.pos = CameraHelper.rotatePositionUsingMouseMoving(
                self.camera.pos,
                self.camera.center,
                self.camera.up,
                *self.context.window_size,
                *self._pre_mouse_pos,
                x, y,
                self.rotate_speed,
                self.reverse_rotate
            )
        elif self._move_button_is_down:
            self.camera.pos, self.camera.center = CameraHelper.moveCenterUsingMouseMoving(
                self.camera.pos,
                self.camera.center,
                self.camera.up,
                *self.context.window_size,
                *self._pre_mouse_pos,
                x, y,
                self.rotate_speed
            )
        self._pre_mouse_pos = (x, y)

    def on_mouseScroll(self, window, x, y):
        self.camera.pos = CameraHelper.changeLengthUsingMouseWheel(
            self.camera.pos,
            self.camera.center,
            y,
            self.scale_speed
        )

    @staticmethod
    def rotatePositionUsingMouseMoving(position, center, axis, window_w, window_h, pre_x, pre_y, x, y, speed, reverse_rotate=False):
        #print(position, center, axis, window_w, window_h, pre_x, pre_y, x, y, speed)
        pos = np.array([*position, 1.0])
        m_center = tuple([-x for x in center])
        ang_v = speed * (y - pre_y) * (1 if reverse_rotate else -1)
        ang_h = speed * (x - pre_x) * (1 if reverse_rotate else -1)
        a = (
            pos[0] - center[0],
            pos[1] - center[1],
            pos[2] - center[2]
        )
        v_poll = (
            axis[1] * a[2] - axis[2] * a[1],
            axis[2] * a[0] - axis[0] * a[2],
            axis[0] * a[1] - axis[1] * a[0]
        )
        len = math.sqrt(v_poll[0] ** 2 + v_poll[1] ** 2 + v_poll[2] ** 2)
        v_poll = (
            v_poll[0] / len,
            v_poll[1] / len,
            v_poll[2] / len
        )
        pos = Matrix.translate(*m_center) * pos
        pos_t = Matrix.rotate(ang_v, *v_poll) * pos
        #上下共に5度以内に入らないようにする．
        len_a = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        len_axis = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
        dot_a_axis = (a[0] * axis[0] + a[1] * axis[1] + a[2] * axis[2])
        esp = len_a * (1 - math.cos(5 / 180 * math.pi))
        e1 = len_a - dot_a_axis / len_axis
        e2 = 2 * len_a - e1
        if (e1 < esp and y - pre_y > 0) or (e2 < esp and y - pre_y < 0):
            pos_t = pos
        pos = Matrix.rotate(ang_h, *axis) * pos_t
        pos = Matrix.translate(*center) * pos
        pos = pos.matrix
        return (pos[0], pos[1], pos[2])

    @staticmethod
    def moveCenterUsingMouseMoving(position, center, axis, window_w, window_h, pre_x, pre_y, x, y, speed):
        #print(position, center, axis, window_w, window_h, pre_x, pre_y, x, y, speed)
        return position, center

    @staticmethod
    def changeLengthUsingMouseWheel(position, center, dx, speed):
        pos = np.array([*position, 1.0])
        m_center = tuple([-x for x in center])
        pos = Matrix.translate(*m_center) * pos
        scale = 1.0 - dx * speed
        pos = Matrix.scale(scale, scale, scale) * pos
        pos = Matrix.translate(*center) * pos
        pos = pos.matrix
        return (pos[0], pos[1], pos[2])
