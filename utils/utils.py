from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import numpy as np
import math

MY_FLOAT = np.float32
MY_UINT = np.uint

def all_done():
    glfw.terminate()
    print("program is finished.")

def printShaderInfoLog(shader, str=""):
    #コンパイル結果の取得
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status == GL_FALSE:
        print("Compile Error in ", str)
    # シェーダのコンパイル時のログの長さを取得する
    bufSize = glGetShaderiv(shader, GL_INFO_LOG_LENGTH).astype(int)
    if bufSize > 1:
        infoLog = glGetShaderInfoLog(shader)
        print(infoLog)
    return status

def printProgramInfoLog(program):
    status = glGetProgramiv(program, GL_LINK_STATUS)
    if status == GL_FALSE:
        print("Link Error.")

    bufSize = glGetProgramiv(program, GL_INFO_LOG_LENGTH).astype(int)
    if bufSize > 1:
        infoLog = glGetProgramInfoLog(program)
        print(infoLog)

    return status

def joinArrays(*args):
    result = []
    data = []
    for arg in args:
        if arg is None:
            continue
        elif type(arg) is np.ndarray:
            data.append(arg.tolist())
            #print(arg.tolist())
        elif type(arg) is list:
            data.append(arg)
    #print(data)
    for elem in zip(*data):
        for array in elem:
            result += array
    return np.array(result, dtype=MY_FLOAT)

def create_program(vertex_shader_file, fragment_shader_file):
    # シェーダーファイルからソースコードを読み込む
    with open(vertex_shader_file, 'r', encoding='utf-8') as f:
        vertex_shader_src = f.read()

    # 作成したシェーダオブジェクトにソースコードを渡しコンパイルする
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_src)
    glCompileShader(vertex_shader)

    with open(fragment_shader_file, 'r', encoding='utf-8') as f:
        fragment_shader_src = f.read()

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_src)
    glCompileShader(fragment_shader)

    # プログラムオブジェクト作成しアタッチ
    program = glCreateProgram()
    if printShaderInfoLog(vertex_shader, "vertex shader"):
        glAttachShader(program, vertex_shader)
    if printShaderInfoLog(fragment_shader, "fragment shader"):
        glAttachShader(program, fragment_shader)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    # 作成したプログラムオブジェクトをリンク
    glLinkProgram(program)

    if printProgramInfoLog(program):
        return program

    glDeleteProgram(program)
    return 0

def create_vao(indices, positions, colors):
    # 座標バッファオブジェクトを作成してデータをGPU側に送る
    position_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)

    # 色バッファオブジェクトを作成してデータをGPU側に送る
    color_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # VAOを作成してバインド
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # 0と1のアトリビュート変数を有効化
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    # 座標バッファオブジェクトの位置を指定(location = 0)
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)

    # 色バッファオブジェクトの位置を指定(location = 1)
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)

    # インデックスオブジェクトを作成してデータをGPU側に送る
    index_vbo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_vbo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)

    # バッファオブジェクトとVAOをアンバインド
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao

def createPatchSphere(r=0.3, sep=5):
    vertices = []
    normals = []
    colors = []
    indices = []
    color = [1,0,0,1]

    seek = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    first = [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    for y in range(0, sep + 1):
        #pos_y = -r / 2 + r * y / sep
        pos_y = y / sep - 0.5
        for x in range(0, 4 * sep):
            f = first[x // sep]
            s = seek[x // sep]
            pos_x = f[0] + s[0] * x / sep
            pos_z = f[1] + s[1] * x / sep
            length = math.sqrt(pos_x ** 2 + pos_y ** 2 + pos_z ** 2)
            vertices.append([
                pos_x / length * r,
                pos_y / length * r,
                pos_z / length * r
            ])
            normals.append([
                pos_x / length,
                pos_y / length,
                pos_z / length
            ])
            colors.append(color)
    print(vertices)
    for y in range(0, sep):
        offset1 = 4 * sep * y
        offset2 = 4 * sep * (y + 1)
        for x in range(0, 4 * sep):
            p1 = offset1 + x % (4 * sep)
            p2 = offset1 + (x + 1) % (4 * sep)
            p3 = offset2 + x % (4 * sep)
            p4 = offset2 + (x + 1) % (4 * sep)
            indices.append([p1, p3, p4])
            indices.append([p4, p2, p1])
    print("v : ", len(vertices))
    print("i : ", len(indices))
    return {
        "vertices" : np.array(vertices, dtype=MY_FLOAT),
        "normals" : np.array(normals, dtype=MY_FLOAT),
        "colors" : np.array(colors, dtype=MY_FLOAT),
        "indices" : np.array(indices, dtype=MY_UINT)
    }

def createSphere_vertexonly(r=0.3, yoko=10, tate=5):
    vertices = []
    normals = []
    colors = []
    indices = []
    color = [1,0,0,1]
    #てっぺん
    vertices.append([0, r, 0, 1])
    normals.append([0, 1, 0])
    colors.append(color)
    for t in range(1, tate - 1):
        y = r * math.cos(t / (tate - 1) * math.pi)
        for v in range(0, yoko):
            x =  r * math.sin(t / (tate - 1) * math.pi) * math.cos(v / yoko * 2 * math.pi)
            z = -r * math.sin(t / (tate - 1) * math.pi) * math.sin(v / yoko * 2 * math.pi)
            vertices.append([x, y, z, 1])
            normals.append([x / r, y / r, z / r])
            colors.append(color)

    vertices.append([0, -r, 0, 1])
    normals.append([0, -1, 0])
    colors.append(color)
    indices = [x for x in range(0, len(vertices))]
    return {
        "vertices" : np.array(vertices, dtype=MY_FLOAT),
        "normals" : np.array(normals, dtype=MY_FLOAT),
        "colors" : np.array(colors, dtype=MY_FLOAT),
        "indices" : np.array(indices, dtype=MY_UINT)
    }

def createSphere(r=0.3, yoko=10, tate=5):
    vertices = []
    normals = []
    colors = []
    indices = []
    color = [1,0,0,1]
    #てっぺん
    vertices.append([0, r, 0, 1])
    normals.append([0, 1, 0])
    colors.append(color)
    for t in range(1, tate - 1):
        y = r * math.cos(t / (tate - 1) * math.pi)
        for v in range(0, yoko):
            x =  r * math.sin(t / (tate - 1) * math.pi) * math.cos(v / yoko * 2 * math.pi)
            z = -r * math.sin(t / (tate - 1) * math.pi) * math.sin(v / yoko * 2 * math.pi)
            vertices.append([x, y, z, 1])
            normals.append([x / r, y / r, z / r])
            colors.append(color)

            if t is 1:
                indices.append([0, v % yoko + 1, (v + 1) % yoko + 1])
            else:
                offset1 = 1 + (t - 2) * yoko
                offset2 = 1 + (t - 1) * yoko
                indices.append([offset2 + v % yoko, offset2 + (v + 1) % yoko, offset1 + v % yoko])
                indices.append([offset2 + (v + 1) % yoko, offset1 + (v + 1) % yoko, offset1 + v % yoko])
            if t is tate - 2:
                offset = (tate - 3) * yoko + 1
                indices.append([offset + v, (tate - 2) * yoko + 1, offset + (v + 1) % yoko])

    vertices.append([0, -r, 0, 1])
    normals.append([0, -1, 0])
    colors.append(color)
    return {
        "vertices" : np.array(vertices, dtype=MY_FLOAT),
        "normals" : np.array(normals, dtype=MY_FLOAT),
        "colors" : np.array(colors, dtype=MY_FLOAT),
        "indices" : np.array(indices, dtype=MY_UINT)
    }

def createTriangle():
    indices = np.array([0, 1, 2], dtype=np.uint)
    vertices = np.array([
        [0.0, 0.5, 0.0, 1.0],
        [-0.5, -0.5, 0.0, 1.0],
        [0.5, -0.5, 0.0, 1.0]
    ], dtype=np.float32)
    colors = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ], dtype=np.float32)
    return [np.array(vertices, dtype=MY_FLOAT), None, np.array(colors, dtype=MY_FLOAT), None,  np.array(indices, dtype=MY_UINT)]

def createCube(r=0.5):
    indices = [
        #前面
        [0, 1, 2],
        [2, 3, 0],
        #左
        [4, 5, 1],
        [1, 0, 4],
        #右
        [3, 2, 6],
        [6, 7, 3],
        #奥
        [7, 6, 5],
        [5, 4, 7],
        #上
        [4, 0, 3],
        [3, 7, 4],
        #下
        [1, 5, 6],
        [6, 2, 1]
    ]
    t = r / 2
    vertices = [
        [-t,  t,  t, 1.0],
        [-t, -t,  t, 1.0],
        [ t, -t,  t, 1.0],
        [ t,  t,  t, 1.0],
        [-t,  t, -t, 1.0],
        [-t, -t, -t, 1.0],
        [ t, -t, -t, 1.0],
        [ t,  t, -t, 1.0]
    ]
    normals = []
    for v in vertices:
        sq = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        normals.append([v[0] / sq, v[1] / sq, v[2] / sq])
    colors = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0]
    ]
    return {
        "vertices" : np.array(vertices, dtype=MY_FLOAT),
        "normals" : np.array(normals, dtype=MY_FLOAT),
        "colors" : np.array(colors, dtype=MY_FLOAT),
        "indices" : np.array(indices, dtype=MY_UINT)
    }
