from OpenGL.GL import *
import glfw
import numpy as np
import atexit
from ctypes import c_void_p
from Matrix import Matrix
from utils.utils import *

aspect = 0.0
scale = 100.0
size = np.array([0,0], dtype=np.float)
def resize(window, width, height):
    global aspect
    # ビューポートの更新
    glViewport(0, 0, width, height)

    #アスペクト比の計算
    aspect = width / height

    size[0] = width
    size[1] = height

def mouseButtonCB(window, button, action, mods):
    print("button : ", button, action, mods)

def mousePosCB(window, x, y):
    print("position : ", x, y)

def mouseScrollCB(window, x, y):
    print("scroll : ", x, y)

def keyFuncCB(widndow, key, scancode, action, mods):
    print("key : ", key, scancode, action, mods)

def charFuncCB(window, charInfo):
    print("char : ", charInfo)

def dropCB(window, paths):
    print("drop : ", paths)

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
    #リスナーの設定
    glfw.set_window_size_callback(window, resize)
    glfw.set_mouse_button_callback(window, mouseButtonCB)
    glfw.set_cursor_pos_callback(window, mousePosCB)
    glfw.set_scroll_callback(window, mouseScrollCB)
    glfw.set_key_callback(window, keyFuncCB)
    glfw.set_char_callback(window, charFuncCB)
    glfw.set_drop_callback(window, dropCB)

    #OpenGLの初期化
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    #glEnable(GL_LIGHTING)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glFrontFace(GL_CCW)
    glCullFace(GL_BACK)
    glClearColor(1,1,1,1)

    #モデルの生成
    #positions, a, colors, b, indices, *c = createSphere()
    #positions, a, colors, b, indices, *c = createTriangle()
    data = createCube()
    positions = data["vertices"] if "vertices" in data.keys() else None
    normals = data["normals"] if "normals" in data.keys() else None
    colors = data["colors"] if "colors" in data.keys() else None
    texcoords = data["texcoords"] if "texcoords" in data.keys() else None
    indices = data["indices"] if "indices" in data.keys() else None

    vertices = joinArrays(positions, normals, colors, texcoords)

    #ライトの設定
    Light = [
        {
            "pos" : [3.0, 3.0, 3.0, 1.0],
            "amb" : [0.2, 0.2, 0.2],
            "diff" : [1.0, 1.0, 1.0],
            "spec" : [1.0, 1.0, 1.0]
        },
        {
            "pos" : [-2.0, 3.0, 3.0, 1.0],
            "amb" : [0.2, 0.2, 0.2],
            "diff" : [1.0, 1.0, 1.0],
            "spec" : [1.0, 1.0, 1.0]
        }
    ]
    Lpos = np.array([l["pos"] for l in Light], dtype=np.float32)
    Lamb = np.array([l["amb"] for l in Light], dtype=np.float32)
    Ldiff = np.array([l["diff"] for l in Light], dtype=np.float32)
    Lspec = np.array([l["spec"] for l in Light], dtype=np.float32)

    #素材の設定
    Material = {
        "amb" : [0.2, 0.2, 0.2],
        "diff" : [1.0, 0.2, 0.2],
        "spec" : [1.0, 1.0, 1.0],
        "shin" : 30.0
    }
    material = np.array([
        *Material["amb"], 0.0,
        *Material["diff"], 0.0,
        *Material["spec"], Material["shin"]
    ], dtype=np.float32)

    #vboの登録
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    #iboの登録
    ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    #uboの登録
    ubo = glGenBuffers(1)
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferData(GL_UNIFORM_BUFFER, material.nbytes, material, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_UNIFORM_BUFFER, 0)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)

    #頂点バッファを設定、セット
    #glEnableClientState(GL_VERTEX_ARRAY);
    #glEnableClientState(GL_NORMAL_ARRAY);
    #glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    program = create_program('shader.vert', 'shader.frag')

    posloc = glGetAttribLocation(program, 'position')
    norloc = glGetAttribLocation(program, 'normal')
    colloc = glGetAttribLocation(program, 'color')
    #print(posloc,norloc,colloc)
    glEnableVertexAttribArray(posloc)
    glEnableVertexAttribArray(norloc)
    #glEnableVertexAttribArray(colloc)
    floatsize = 4
    stride = floatsize * 11
    glVertexAttribPointer(posloc, 4, GL_FLOAT, GL_FALSE, stride, c_void_p(floatsize * 0))
    glVertexAttribPointer(norloc, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(floatsize * 4))
    #glVertexAttribPointer(colloc, 4, GL_FLOAT, GL_FALSE, stride, c_void_p(floatsize * 7))

    model_loc = glGetUniformLocation(program, 'modelview')
    projection_loc = glGetUniformLocation(program, 'projection')
    normalMatrix_loc = glGetUniformLocation(program, 'normalMatrix')
    Lpos_loc = glGetUniformLocation(program, 'Lpos')
    Lamb_loc = glGetUniformLocation(program, 'Lamb')
    Ldiff_loc = glGetUniformLocation(program, 'Ldiff')
    Lspec_loc = glGetUniformLocation(program, 'Lspec')

    mat_loc = glGetUniformBlockIndex(program, "Material")
    glUniformBlockBinding(program, mat_loc, 0)

    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo)

    print(mat_loc)
    print(model_loc)
    print(projection_loc)
    print(Lpos_loc)
    print(Lamb_loc)
    print(Ldiff_loc)
    print(Lspec_loc)

    resize(window, width, height)

    # OpenGLのバージョン等を表示します
    print('Vendor :', glGetString(GL_VENDOR))
    print('GPU :', glGetString(GL_RENDERER))
    print('OpenGL version :', glGetString(GL_VERSION))

    count = 0

    while not glfw.window_should_close(window):
        # バッファを指定色で初期化
        #glClearColor(0.2, 0.2, 0.2, 1)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # シェーダを有効化
        glUseProgram(program)

        #変換行列の作成
        scaling = Matrix.scale(scale / size[0], scale / size[1], 1)
        translation = Matrix.translate(0.5 * math.sin(count / 60 * 2 * math.pi / 2), 0.5 * math.cos(count / 60 * 2 * math.pi / 2), 0)
        modelMat = translation

        view = Matrix.lookat(*(3,2,1),*(0,0,0),*(0,1,0))
        #view = Matrix.lookat(0,0,0,-1,-1,-1,0,1,0)
        modelviewMat = view * modelMat

        fovy = scale * 0.01
        projectionMat = Matrix.perspective(fovy, aspect, 1.0, 10.0)

        normalMatrix = projectionMat.getNormalMatrix()

        # uniformの設定
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, modelviewMat.matrix)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projectionMat.matrix)
        glUniformMatrix3fv(normalMatrix_loc, 1, GL_FALSE, normalMatrix.matrix)
        glUniform4fv(Lpos_loc, 2, Lpos)
        glUniform3fv(Lamb_loc, 2, Lamb)
        glUniform3fv(Ldiff_loc, 2, Ldiff)
        glUniform3fv(Lspec_loc, 2, Lspec)

        # バインドしたVBOを用いて描画
        glDrawElements(GL_TRIANGLES, len(indices) * 3, GL_UNSIGNED_INT, None)

        # バッファを入れ替えて画面を更新
        glfw.swap_buffers(window)

        # イベントを受け付けます
        glfw.poll_events()

        count += 1

    # ウィンドウを破棄してGLFWを終了します
    glfw.destroy_window(window)

if __name__ == '__main__':
    main()
