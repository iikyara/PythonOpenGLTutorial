import threading
from utils.utils import *
from OpenGL.GL import *
import cv2
from Core import *
from CameraHelper import CameraHelper
from SketchToModel import *
from SwarmAgent import *

def main():
    app = App()
    camerahelper = CameraHelper(app.context, app.frames[0].scenes[0].current_camera)
    app.context.show_version()

    swarm_agent = SwarmAgent(
        numParticles=50,
        mesh=Mesh(**createSphere(r=0.1, yoko=10, tate=10))
    )
    env = STMEnvironment(agents=[swarm_agent])

    scene = app.frames[0].scenes[0]
    scene.objects = []
    scene.addObjects(swarm_agent.objects)

    thread = threading.Thread(target=env_tick, args=(env, app))
    thread.start()

    app.run()

def env_tick(env, app):
    print("--start env_tick--")
    while app.isRunning:
        env.update()
    print("--end env_tick--")

if __name__ == '__main__':
    main()
