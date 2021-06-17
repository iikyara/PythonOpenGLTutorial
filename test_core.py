import math
import threading
from utils.utils import *
from OpenGL.GL import *
from Core import *
from CameraHelper import CameraHelper

def main():
    app = App()
    camerahelper = CameraHelper(app.context, app.frames[0].scenes[0].current_camera)
    app.context.show_version()
    #crear mesh
    app.frames[0].scenes[0].objects[0].meshes = []

    n = 100
    data = createSphere(r=0.1, yoko=100, tate=100)
    mesh = Mesh(**data)
    for i in range(n):
        object = Object(meshes=[mesh])
        object.setPosition([
            math.sin(i / n * 2 * math.pi),
            math.sin(i / n * math.pi),
            math.cos(i / n * 2 * math.pi)
        ])
        app.frames[0].scenes[0].addObject(object)
    thread = threading.Thread(target=app.run())
    thread.start()

if __name__ == '__main__':
    main()
