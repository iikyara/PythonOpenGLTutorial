# OpenGLとGLFWをインポートします
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import numpy as np
import atexit
from utils.utils import *
from Matrix import Matrix
from Object import *

cubeVertex = np.array([
    [ -1.0, -1.0, -1.0 ],
    [ -1.0, -1.0,  1.0 ],
    [ -1.0,  1.0,  1.0 ],
    [ -1.0,  1.0, -1.0 ],
    [  1.0,  1.0, -1.0 ],
    [  1.0, -1.0, -1.0 ],
    [  1.0, -1.0,  1.0 ],
    [  1.0,  1.0,  1.0 ]
], dtype=np.float)

wireCubeIndex = np.array([
    1, 0,
    2, 7,
    3, 0,
    4, 7,
    5, 0,
    6, 7,
    1, 2,
    2, 3,
    3, 4,
    4, 5,
    5, 6,
    6, 1
], dtype=np.uint)

class Model:
    def __init__(self):
        # 3角形
        self.indices = np.array([0, 1, 2], dtype=np.uint)
        self.positions = np.array([[0.0, 0.5, 0.0, 1.0], [0.5, -0.5, 0.0, 1.0], [-0.5, -0.5, 0.0, 1.0]], dtype=np.float32)
        self.colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

    def cube(self):
        # キューブ
        self.positions = np.array([
            [ 0.0, 0.0, 0.0 ],
            [ 1.0, 0.0, 0.0 ],
            [ 1.0, 1.0, 0.0 ],
            [ 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 1.0 ],
            [ 1.0, 0.0, 1.0 ],
            [ 1.0, 1.0, 1.0 ],
            [ 0.0, 1.0, 1.0 ]
        ], dtype=np.uint)
        self.indices = np.array([
            [ 0, 1, 2, 3 ],
            [ 1, 5, 6, 2 ],
            [ 5, 4, 7, 6 ],
            [ 4, 0, 3, 7 ],
            [ 4, 5, 1, 0 ],
            [ 3, 2, 6, 7 ]
        ], dtype=np.uint)
        self.normals = np.array([
            [ 0.0, 0.0,-1.0 ],
            [ 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 1.0 ],
            [-1.0, 0.0, 0.0 ],
            [ 0.0,-1.0, 0.0 ],
            [ 0.0, 1.0, 0.0 ]
        ], dtype=np.uint)

    def move(self):
        pass
        '''
        for p in self.positions:
            p[0] = p[0] + 0.005;
            p[1] = p[1] + 0.005;
        '''

    def create_vao(self):
        return create_vao(self.indices, self.positions, self.colors)

def all_done():
    glfw.terminate()
    print("program is finished.")

aspect = 0.0
scale = 100.0 * 2
size = np.array([0,0], dtype=np.float)
def resize(window, width, height):
    global aspect
    # ビューポートの更新
    glViewport(0, 0, width, height)

    #アスペクト比の計算
    aspect = width / height

    size[0] = width
    size[1] = height

    # 透視投影
    #gluPerspective(30.0, width / height, 1.0, 100.0)

def main():
    # GLFW初期化
    if not glfw.init():
        return

    atexit.register(all_done)

    width = 640
    height = 480

    # ウィンドウを作成
    window = glfw.create_window(width, height, 'Hello World', None, None)
    if not window:
        print('Failed to create window')
        return

    # コンテキストを作成します
    glfw.make_context_current(window)

    #OpenGLのバージョンを指定
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    #glfwの設定
    glfw.swap_interval(1)
    glfw.set_window_size_callback(window, resize)

    #OpenGLの初期化
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    #glEnable(GL_LIGHTING)
    glCullFace(GL_FRONT)
    glClearColor(1,1,1,1)

    program = create_program('shader.vert', 'shader.frag')

    aspectLoc = glGetUniformLocation(program, "aspect")
    sizeLoc = glGetUniformLocation(program, "size")
    scaleLoc = glGetUniformLocation(program, "scale")
    mouseLoc = glGetUniformLocation(program, "mouse")
    modelLoc = glGetUniformLocation(program, "model")
    modelviewLoc = glGetUniformLocation(program, "modelview")
    projectionLoc = glGetUniformLocation(program, "projection")

    resize(window, width, height)

    #モデルの生成
    shape = Shape(3, cubeVertex, wireCubeIndex)
    #model = Model()
    #model.cube()
    #vao = model.create_vao()

    # OpenGLのバージョン等を表示します
    print('Vendor :', glGetString(GL_VENDOR))
    print('GPU :', glGetString(GL_RENDERER))
    print('OpenGL version :', glGetString(GL_VERSION))

    while not glfw.window_should_close(window):
        # バッファを指定色で初期化
        #glClearColor(0.2, 0.2, 0.2, 1)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 現在のウィンドウの大きさを取得
        #width, height = glfw.get_framebuffer_size(window)

        gluLookAt(3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # シェーダを有効化
        glUseProgram(program)

        # マウスの位置
        mou_x, mou_y = glfw.get_cursor_pos(window)
        mouse = np.array([
            mou_x * 2.0 / size[0] - 1.0,
            1.0 - mou_y * 2 / size[1]
        ], np.float)

        #モデル変換行列を生成
        scaling = Matrix.scale(scale / size[0], scale / size[1], 1)
        translation = Matrix.translate(mouse[0], mouse[1], 0)
        modelMat = translation * scaling

        view = Matrix.lookat(3,4,5,0,0,0,0,1,0)
        modelviewMat = view * modelMat

        w = size[0] / scale
        h = size[1] / scale
        fovy = scale * 0.01
        #projectionMat = Matrix.frustum(-w, w, -h, h, 1.0, 10.0)
        projectionMat = Matrix.perspective(fovy, aspect, 1.0, 10.0)

        glUniform1f(aspectLoc, float(aspect))
        glUniform1f(scaleLoc, float(scale))
        glUniform2fv(sizeLoc, 1, size)
        glUniform2fv(mouseLoc, 1, mouse)
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, modelMat.matrix)
        glUniformMatrix4fv(modelviewLoc, 1, GL_FALSE, modelviewMat.matrix)
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projectionMat.matrix)

        # vaoをバインド
        #glBindVertexArray(vao)
        shape.draw()

        # バインドしたVAOを用いて描画
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, None)

        # アンバインド
        glBindVertexArray(0)

        #モデルの更新
        #model.move()
        #vao = model.create_vao()

        # バッファを入れ替えて画面を更新
        glfw.swap_buffers(window)

        # イベントを受け付けます
        glfw.poll_events()

    # ウィンドウを破棄してGLFWを終了します
    glfw.destroy_window(window)


# Pythonのメイン関数はこんな感じで書きます
if __name__ == "__main__":
    main()
