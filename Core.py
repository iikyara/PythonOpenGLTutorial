from OpenGL.GL import *
from OpenGL.GLUT import *
import glfw
import numpy as np
import atexit
from ctypes import c_void_p
from Matrix import Matrix
from utils.utils import *

class App:
    instance = None
    def __init__(self, frames=None):
        #アプリを２つ一度に起動することを禁止
        if App.instance is not None:
            raise Exception("Don't generate two App instance simultaneously.")
        App.instance = self

        #セットアップ
        self.context = Context()
        #ウィンドウを登録
        self.frames = frames or [Frame()]
        #ウィンドウサイズマネージャ
        self.window_size_manager = WindowSizeManager(self.context, self.frames)

        self.isRunning = False

    #１ループでの操作
    def tick(self):
        #画面を指定色で初期化
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #ウィンドウの描画
        for frame in self.frames:
            if frame.visible:
                frame.parent = self
                frame.draw()
        #コンテキストの処理
        self.context.tick()

    #実行
    def run(self):
        self.isRunning = True
        while not self.context.window_should_close():
            self.tick()
        glfw.destroy_window(self.context.window)
        self.isRunning = False

    #削除
    def delete(self):
        App.instance = None
        map(lambda x:x.delete(), self.frames)


# 各種セットアップをする．
class Context:
    instance = None
    def __init__(
        self, width=640, height=480, title="IiGraphics",
        swap_interval=1, settings=None, cull_face=GL_FRONT,
        clear_color=(0.2, 0.2, 0.2, 1.0),
        on_resize = None,
        on_mouseButton = None,
        on_mousePosition = None,
        on_mouseScroll = None,
        on_keyboard = None,
        on_char = None,
        on_drop = None
    ):
        if Context.instance is not None:
            raise Exception("Don't generate two Context instance simultaneously.")
        Context.instance = self
        # GLFW初期化
        if not glfw.init():
            raise Exception('Failed to initialize glfw.')
        #glfw.terminateを登録
        atexit.register(all_done)

        # ウィンドウを作成
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            raise Exception('Failed to create window')

        # コンテキストを作成
        glfw.make_context_current(self.window)

        #OpenGLのバージョンを指定
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        #描画間隔を設定
        self.swap_interval = swap_interval
        self.set_swap_interval(self.swap_interval)
        #リスナーの登録
        glfw.set_window_size_callback(self.window, self.resizeListener)
        glfw.set_mouse_button_callback(self.window, self.mouseButtonListener)
        glfw.set_cursor_pos_callback(self.window, self.mousePositionListener)
        glfw.set_scroll_callback(self.window, self.mouseScrollListener)
        glfw.set_key_callback(self.window, self.keyboardListener)
        glfw.set_char_callback(self.window, self.charListener)
        glfw.set_drop_callback(self.window, self.dropListener)
        #イベント時に実行する項目を設定
        self.on_resize = on_resize or []
        self.on_mouseButton = on_mouseButton or []
        self.on_mousePosition = on_mousePosition or []
        self.on_mouseScroll = on_mouseScroll or []
        self.on_keyboard = on_keyboard or []
        self.on_char = on_char or []
        self.on_drop = on_drop or []

        #画面リサイズ
        self.window_size = (0, 0)
        self.resizeListener(self.window, width, height)

        self.settings = []
        self.set_settings(settings or [
            GL_DEPTH_TEST,
            GL_CULL_FACE,
            #GL_LIGHTING,
        ])
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glFrontFace(GL_CCW)
        glCullFace(GL_BACK)

        glClearColor(*clear_color)

    def set_swap_interval(self, swap_interval):
        self.swap_interval = swap_interval
        glfw.swap_interval(swap_interval)

    def set_settings(self, new_settings=[]):
        for setting in self.settings:
            glDisable(setting)
        self.settings = new_settings
        for setting in self.settings:
            glEnable(setting)

    def window_should_close(self):
        return glfw.window_should_close(self.window)

    def tick(self):
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def show_version(self):
        # OpenGLのバージョン等を表示
        log('Vendor :', glGetString(GL_VENDOR))
        log('GPU :', glGetString(GL_RENDERER))
        log('OpenGL version :', glGetString(GL_VERSION))

    def resizeListener(self, window, width, height):
        self.window_size = (width, height)
        glViewport(0, 0, width, height)
        for func in self.on_resize:
            func(window, width, height)

    def mouseButtonListener(self, window, button, action, mods):
        #print("button : ", button, action, mods)
        for func in self.on_mouseButton:
            func(window, button, action, mods)

    def mousePositionListener(self, window, x, y):
        #print("position : ", x, y)
        for func in self.on_mousePosition:
            func(window, x, y)

    def mouseScrollListener(self, window, x, y):
        #print("scroll : ", x, y)
        for func in self.on_mouseScroll:
            func(window, x, y)

    def keyboardListener(self, widndow, key, scancode, action, mods):
        #print("key : ", key, scancode, action, mods)
        for func in self.on_keyboard:
            func(window, key, scancode, action, mods)

    def charListener(self, window, charInfo):
        #print("char : ", charInfo)
        for func in self.on_char:
            func(window, charInfo)

    def dropListener(self, window, paths):
        #print("drop : ", paths)
        for func in self.on_drop:
            func(window, paths)

#描画オブジェクトに付与
class meta:
    def __init__(self):
        #self.enable = True
        self.visible = True
        self.matrix = Matrix.identity()
        self.parent = None

    #描画を許可
    def show(self):
        self.visible = True

    #描画を禁止
    def hide(self):
        self.visible = False

    def draw(self):
        raise Exception("Draw method should be overrided.")

class Frame(meta):
    def __init__(self, width=640, height=480, scenes=None, uimanager=None):
        super(Frame, self).__init__()
        self.scenes = []
        self.addScenes(scenes or [Scene()])
        self.UIManager = uimanager or UIManager()
        self.projection_uniform = Uniform("projection", GL_FLOAT, self.matrix.matrix)

        #パラメータ
        self.pos = (0, 0)
        self.rotate = 0     #ウィンドウの回転（cwが正）
        self.size = (width, height)
        self.fovy = 60 / 180 * math.pi
        self.z_depth = (0.1, 300.0)
        self.scale = (1, 1)

        self.calcMatrix()

    #シーンを追加
    def addScene(self, scene):
        scene.parent = self
        self.scenes.append(scene)

    #シーンを複数追加
    def addScenes(self, scenes):
        for scene in scenes:
            self.addScene(scene)

    def setPosition(self, x, y):
        self.pos = (x, y)
        self.calcMatrix()

    def setRotate(self, r):
        self.rotate = r
        self.calcMatrix()

    def setScale(self, x, y):
        self.scale = (x, y)
        self.calcMatrix()

    def setSize(self, x, y):
        self.size = (x, y)

    def calcMatrix(self):
        #Matrix.translate(*self.pos, 0) * Matrix.rotate_z(self.rotate) * Matrix.scale(*self.scale, 1) *
        self.matrix = Matrix.perspective(self.fovy, self.size[0] / self.size[1], *self.z_depth)
        self.projection_uniform.setData(GL_FLOAT, self.matrix.matrix)

    #描画
    def draw(self):
        for scene in self.scenes:
            if scene.visible:
                scene.parent = self
                scene.draw(self.matrix)
        self.UIManager.draw()

class UIManager(meta):
    def __init__(self):
        super(UIManager, self).__init__()
        pass

    def draw(self):
        pass

# 3次元シーン
class Scene(meta):
    def __init__(self, objects=None, cameras=None, lights=None):
        super(Scene, self).__init__()
        self.objects = []
        self.addObjects(objects or [Object()])
        self.cameras = cameras or [Camera()]
        self.current_camera = self.cameras[0]
        self.lights = []
        self.lights_uniforms = []
        self.addLights(lights or [Light(), Light()])
        #name, type, data, size=1, transpose=GL_FALSE
        #self.readyLightsUniforms()

    #オブジェクトを追加
    def addObject(self, object):
        object.parent = self
        self.objects.append(object)

    def addObjects(self, objects):
        for object in objects:
            self.addObject(object)

    def addLights(self, lights):
        self.lights += lights
        self.readyLightsUniforms()

    def readyLightsUniforms(self):
        self.lights_uniforms = [
            Uniform("Lpos", GL_FLOAT, np.zeros((len(self.lights), 4)), size=len(self.lights)),
            Uniform("Lamb", GL_FLOAT, np.zeros((len(self.lights), 3)), size=len(self.lights)),
            Uniform("Ldiff", GL_FLOAT, np.zeros((len(self.lights), 3)), size=len(self.lights)),
            Uniform("Lspec", GL_FLOAT, np.zeros((len(self.lights), 3)), size=len(self.lights))
        ]

    def setLightDataToUniform(self):
        for uniform, data in zip(self.lights_uniforms, Light.joinLight(self.lights, self.current_camera.matrix)):
            uniform.setData(GL_FLOAT, data, data.shape[0])

    def enableLightsUniforms(self, program):
        for uniform in self.lights_uniforms:
            uniform.enable(program)

    def draw(self, matrix):
        for object in self.objects:
            if object.visible:
                object.parent = self
                object.draw(matrix * self.matrix)

class Camera(meta):
    def __init__(self, pos=(1, 1, 1), center=(0, 0, 0), up=(0, 1, 0)):
        super(Camera, self).__init__()
        self.pos = pos
        self.center = center
        self.up = up
        self.calcMatrix()

    def calcMatrix(self):
        self.matrix = self.lookat()

    def lookat(self):
        return Matrix.lookat(*self.pos, *self.center, *self.up)

    def setPosition(self, pos):
        self.pos = pos
        self.calcMatrix()

    def setCenter(self, center):
        self.center = center
        self.calcMatrix()

    def setUp(self, up):
        self.up = up
        self.calcMatrix()

    def setAll(self, pos, center, up):
        self.pos = pos
        self.center = center
        self.up = up
        self.calcMatrix()

class Light(meta):
    def __init__(self, position=(3.0, 3.0, 3.0, 1.0), ambient=(0.2, 0.2, 0.2), diffuse=(1.0, 1.0, 1.0), specular=(1.0, 1.0, 1.0)):
        self.position = position
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

    @staticmethod
    def joinLight(lights, view):
        poss = []
        ambs = []
        difs = []
        spes = []
        for light in lights:
            poss.append(list((view * np.array(light.position)).matrix))
            ambs.append(list(light.ambient))
            difs.append(list(light.diffuse))
            spes.append(list(light.specular))
        return (
            np.array(poss, dtype=MY_FLOAT),
            np.array(ambs, dtype=MY_FLOAT),
            np.array(difs, dtype=MY_FLOAT),
            np.array(spes, dtype=MY_FLOAT),
        )

# 3次元オブジェクト
class Object(meta):
    def __init__(self, meshes=None):
        super(Object, self).__init__()
        self.meshes = []
        self.addMeshes(meshes or [Mesh(**createCube())])
        self.position = (0.0, 0.0, 0.0)
        self.rotate = (0.0, 0.0, 0.0) #オイラー角
        self.scale = (1.0, 1.0, 1.0)

    def addMesh(self, mesh):
        mesh.parent = self
        self.meshes.append(mesh)

    def addMeshes(self, meshes):
        for mesh in meshes:
            self.addMesh(mesh)

    def setPosition(self, position):
        self.position = position
        self.calcMatrix()

    def setRotate(self, rotate):
        self.rotate = rotate
        self.calcMatrix()

    def setScale(self, scale):
        self.scale = scale
        self.calcMatrix()

    def setAll(self, position, rotate, scale):
        self.position = position
        self.rotate = rotate
        self.scale = scale
        self.calcMatrix()

    def calcMatrix(self):
        self.matrix = Matrix.translate(*self.position) * Matrix.rotate_eular(*self.rotate) * Matrix.scale(*self.scale)

    def draw(self, matrix):
        for mesh in self.meshes:
            if mesh.visible:
                mesh.parent = self
                mesh.draw(matrix * self.matrix)

class Mesh(meta):
    def __init__(
        self, vertices=None, normals=None, colors=None, texcoords=None, indices=None,
        attrs=None, draw_mode=GL_TRIANGLES,
        program=None, material=None
    ):
        super(Mesh, self).__init__()
        self.vertex_buffer = VertexBuffer(joinArrays(vertices, normals, colors, texcoords))
        self.index_buffer = IndexBuffer(indices)
        self.program = program or Program.create("shader.vert", "shader.frag")
        self.material = material or Material()
        self.draw_mode = draw_mode

        self.vertices_count = indices.size
        self.modelview_uniform = Uniform("modelview", GL_FLOAT, self.matrix.matrix)
        self.normalMatrix_uniform = Uniform("normalMatrix", GL_FLOAT, Matrix(size=3).matrix)

        #各要素を含むか
        self.have_vertices  = vertices is not None
        self.have_normals   = normals is not None
        self.have_colors    = colors is not None
        self.have_texcoords = texcoords is not None
        self.have_indices   = indices is not None

        self.have_vertices_uniform = Uniform("have_vertices", GL_INT, self.have_vertices)
        self.have_normals_uniform = Uniform("have_normals", GL_INT, self.have_normals)
        self.have_colors_uniform = Uniform("have_colors", GL_INT, self.have_colors)
        self.have_texcoords_uniform = Uniform("have_texcoords", GL_INT, self.have_texcoords)

        #attributeの設定
        float_size = 4
        attr_sizes = [
            4 if self.have_vertices else 0,
            3 if self.have_normals else 0,
            4 if self.have_colors else 0,
            2 if self.have_texcoords else 0
        ]

        self.stride = float_size * sum(attr_sizes)
        self.attrs = attrs or [
            Attribute('position', 4, GL_FLOAT, GL_FALSE, self.stride, float_size * sum(attr_sizes[0:0])),
            Attribute('normal', 3, GL_FLOAT, GL_FALSE, self.stride, float_size * sum(attr_sizes[0:1])),
            Attribute('color', 4, GL_FLOAT, GL_FALSE, self.stride, float_size * sum(attr_sizes[0:2])),
            Attribute('texcoord', 2, GL_FLOAT, GL_FALSE, self.stride, float_size * sum(attr_sizes[0:3]))
        ]
        #name, size, type, normalized, stride, offset
        self.showInfo()

    @property
    def vertices(self):
        pass

    @property
    def normals(self):
        pass

    @property
    def colors(self):
        pass

    @property
    def texcoords(self):
        pass

    @property
    def face(self):
        pass

    @property
    def _frame(self):
        return self.parent.parent.parent

    @property
    def _scene(self):
        return self.parent.parent

    @property
    def _camera(self):
        return self._scene.current_camera

    @property
    def _object(self):
        return self.parent

    def calcMatrix(self):
        # view * model
        self._camera.calcMatrix()
        self.matrix = self._camera.matrix * self._object.matrix
        self.modelview_uniform.setData(GL_FLOAT, self.matrix.matrix)
        self.normalMatrix_uniform.setData(GL_FLOAT, self.matrix.getNormalMatrix().matrix)

    def enableUniform(self):
        #モデルビュープロジェクション変換ユニフォームの有効化
        self._frame.projection_uniform.enable(self.program.program)
        self.modelview_uniform.enable(self.program.program)
        self.normalMatrix_uniform.enable(self.program.program)
        #ライトユニフォームの有効化
        self._scene.enableLightsUniforms(self.program.program)
        #マテリアルユニフォームの有効化
        self.material.ubo.enable(self.program.program)
        #各種パラメータが含まれるかを表すユニフォームの有効化
        self.have_vertices_uniform.enable(self.program.program)
        self.have_normals_uniform.enable(self.program.program)
        self.have_colors_uniform.enable(self.program.program)
        self.have_texcoords_uniform.enable(self.program.program)
        #self.showUniformInfo()

    def enableAttribute(self):
        for attr in self.attrs:
            attr.enable(self.program.program)
            #attr.showInfo()

    def disableAttribute(self):
        for attr in self.attrs:
            attr.disable()

    #draw_modeで描画
    def draw(self, matrix):
        #ライトの再計算
        self._scene.setLightDataToUniform()
        #行列の計算
        self.calcMatrix()
        #シェーダの有効化
        glUseProgram(self.program.program)
        #頂点・インデックスバッファをバインド
        self.vertex_buffer.bind()
        self.index_buffer.bind()
        #頂点バッファをシェーダから参照出来るようにする．
        self.enableAttribute()
        #uniformを有効化
        self.enableUniform()
        #[x.showInfo() for x in self.attrs]
        #描画
        glDrawElements(self.draw_mode, self.vertices_count, GL_UNSIGNED_INT, None)
        #頂点・インデックスバッファをアンバインド
        self.vertex_buffer.unbind()
        self.index_buffer.unbind()
        #紐づけ解除
        self.disableAttribute()

    def showInfo(self):
        log("【Mesh : {!r}】".format(self))
        log("have_vertices : ", self.have_vertices)
        log("have_normals", self.have_normals)
        log("have_colors", self.have_colors)
        log("have_texcoords", self.have_texcoords)
        log("have_indices", self.have_indices)
        for attr in self.attrs:
            log(attr)

    def showUniformInfo(self):
        #モデルビュープロジェクション変換ユニフォームの情報
        log(self._frame.projection_uniform)
        log(self.modelview_uniform)
        log(self.normalMatrix_uniform)
        #ライトユニフォームの情報
        for u in self._scene.lights_uniforms:
            log(u)
        #マテリアルユニフォームの情報
        log(self.material.ubo)

class Material:
    def __init__(self, ambient=(0.2, 0.2, 0.2), diffuse=(1.0, 0.2, 0.2), specular=(1.0, 1.0, 1.0), shininess = 30.0):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.ubo = UniformBuffer("Material", Material.joinData(
            self.ambient, self.diffuse, self.specular, self.shininess
        ))

    def setData(self, ambient=None, diffuse=None, specular=None, shininess=None):
        self.ambient = ambient or self.ambient
        self.diffuse = diffuse or self.diffuse
        self.specular = specular or self.specular
        self.shininess = shininess or self.shininess
        self.ubo.setData(Material.joinData(
            self.ambient, self.diffuse, self.specular, self.shininess
        ))

    @staticmethod
    def joinData(ambient, diffuse, specular, shininess):
        data = [*ambient, 0.0, *diffuse, 0.0, *specular, shininess]
        return np.array(data, dtype=MY_FLOAT)

class Uniform:
    def __init__(self, name, type, data, size=1, transpose=GL_FALSE):
        self.name = name
        self.type = type
        self.data = None
        self.size = size
        self.transpose = transpose
        self.func = None
        self.location = -1
        self.setData(self.type, data, self.size)
        log(self)

    def __str__(self):
        return "name : {!r}, type: {!r}, data: {!r}, size: {!r}, transpose: {!r}, func: {!r}, location: {!r}".format(self.name, self.type, type(self.data), self.size, self.transpose, self.func, self.location)

    def __repr__(self):
        return "name : {!r}, type: {!r}, data: {!r}, size: {!r}, transpose: {!r}, func: {!r}, location: {!r}".format(self.name, self.type, type(self.data), self.size, self.transpose, self.func, self.location)

    def setData(self, type, data, size=1):
        self.size = size
        self.data = np.array(data)
        self.func = self._selectFunc(
            self.type,
            (self.data.ndim - 1) if size > 1 else self.data.ndim,
            tuple(self.data.shape[i] for i in (range(1, len(self.data.shape)) if size > 1 else range(0, len(self.data.shape))))
        )

    def enable(self, program):
        self.location = glGetUniformLocation(program, self.name)
        if self.location is -1:
            return
        dim = (self.data.ndim - 1) if self.size > 1 else self.data.ndim
        if dim == 0:
            self.func(self.location, self.data)
        elif dim == 1:
            self.func(self.location, self.size, self.data)
        else:
            self.func(self.location, self.size, self.transpose, self.data)

    def _selectFunc(self, type, ndim, shape):
        return {
            (GL_INT, 0, ()) : glUniform1i,
            (GL_INT, 1, (1,)) : glUniform1iv,
            (GL_INT, 1, (2,)) : glUniform2iv,
            (GL_INT, 1, (3,)) : glUniform3iv,
            (GL_INT, 1, (4,)) : glUniform4iv,
            (GL_UNSIGNED_INT, 0, ()) : glUniform1ui,
            (GL_UNSIGNED_INT, 1, (1,)) : glUniform1uiv,
            (GL_UNSIGNED_INT, 1, (2,)) : glUniform2uiv,
            (GL_UNSIGNED_INT, 1, (3,)) : glUniform3uiv,
            (GL_UNSIGNED_INT, 1, (4,)) : glUniform4uiv,
            (GL_FLOAT, 0, ()) : glUniform1f,
            (GL_FLOAT, 1, (1,)) : glUniform1fv,
            (GL_FLOAT, 1, (2,)) : glUniform2fv,
            (GL_FLOAT, 1, (3,)) : glUniform3fv,
            (GL_FLOAT, 1, (4,)) : glUniform4fv,
            (GL_FLOAT, 2, (2, 2)) : glUniformMatrix2fv,
            (GL_FLOAT, 2, (3, 3)) : glUniformMatrix3fv,
            (GL_FLOAT, 2, (4, 4)) : glUniformMatrix4fv,
            (GL_FLOAT, 2, (2, 3)) : glUniformMatrix2x3fv,
            (GL_FLOAT, 2, (3, 2)) : glUniformMatrix3x2fv,
            (GL_FLOAT, 2, (2, 4)) : glUniformMatrix2x4fv,
            (GL_FLOAT, 2, (4, 2)) : glUniformMatrix4x2fv,
            (GL_FLOAT, 2, (3, 4)) : glUniformMatrix3x4fv,
            (GL_FLOAT, 2, (4, 3)) : glUniformMatrix4x3fv
        }[type, ndim, shape]

class Attribute:
    def __init__(self, name, size, type, normalized, stride, offset):
        self.name = name
        self.size = size
        self.type = type
        self.normalized = normalized
        self.stride = stride
        self.offset = offset
        self.enabled = False
        self.location = -1

    def __str__(self):
        return "name : {!r}, size: {!r}, stride: {!r}, offset: {!r}, location: {!r}".format(self.name, self.size, self.stride, self.offset, self.location)

    def __repr__(self):
        return "name : {!r}, size: {!r}, stride: {!r}, offset: {!r}, location: {!r}".format(self.name, self.size, self.stride, self.offset, self.location)

    def enable(self, program):
        self.location = glGetAttribLocation(program, self.name)
        if self.location is -1:
            return
        glEnableVertexAttribArray(self.location)
        glVertexAttribPointer(
            self.location,
            self.size,
            self.type,
            self.normalized,
            self.stride,
            c_void_p(self.offset)
        )
        self.enabled = True

    def disable(self):
        if self.enabled:
            glDisableVertexAttribArray(self.location)
            self.enabled = False

    def showInfo(self):
        log("Attribute[{!r}] : {!r}".format(self.name, "bind" if self.enabled else "unbind"))

class BufferObject:
    def __init__(self, type, data, buffer_mode=GL_STATIC_DRAW):
        self.type = type
        self.data = data
        self.buffer_mode = buffer_mode
        self.size = 0
        #buffer objectの作成
        self.buffer_object = glGenBuffers(1)
        self.setData(self.data)

    def setData(self, data):
        size = data.nbytes
        glBindBuffer(self.type, self.buffer_object)
        if self.size is not size:
            glBufferData(self.type, self.data.nbytes, self.data, self.buffer_mode)
        else:
            glBufferSubData(self.type, 0, self.data.nbytes, self.data)
        glBindBuffer(self.type, 0)
        #更新
        self.data = data
        self.size = size

    def bind(self):
        glBindBuffer(self.type, self.buffer_object)

    def unbind(self):
        glBindBuffer(self.type, 0)

class VertexBuffer(BufferObject):
    def __init__(self, vertices, buffer_mode=GL_STATIC_DRAW):
        super(VertexBuffer, self).__init__(GL_ARRAY_BUFFER, vertices, buffer_mode)

class IndexBuffer(BufferObject):
    def __init__(self, indices, buffer_mode=GL_STATIC_DRAW):
        super(IndexBuffer, self).__init__(GL_ELEMENT_ARRAY_BUFFER, indices, buffer_mode)

class UniformBuffer(BufferObject):
    def __init__(self, name, data, buffer_mode=GL_DYNAMIC_DRAW):
        super(UniformBuffer, self).__init__(GL_UNIFORM_BUFFER, data, buffer_mode)
        self.name = name
        self.location = -1

    def __str__(self):
        return "name : {!r}, data: {!r}, buffer_mode: {!r}, location: {!r}".format(self.name, type(self.data), self.buffer_mode, self.location)

    def __repr__(self):
        return "name : {!r}, data: {!r}, buffer_mode: {!r}, location: {!r}".format(self.name, type(self.data), self.buffer_mode, self.location)

    def enable(self, program):
        self.location = glGetUniformBlockIndex(program, self.name)
        if self.location is -1:
            return
        glUniformBlockBinding(program, self.location, 0)
        glBindBufferBase(self.type, 0, self.buffer_object)

class Vertex:
    def __init__(self,
        id,
        position=np.array([0,0,0,0], dtype=MY_FLOAT),
        normal=np.array([0,0,0], dtype=MY_FLOAT),
        color=np.array([0,0,0], dtype=MY_UINT),
        texcoord=np.array([0,0], dtype=MY_FLOAT)
    ):
        self.id = id
        self.position = position
        self.normal = normal
        self.color = color
        self.texcoord = texcoord

class Face:
    def __init__(self, vertices):
        self.dimension = len(vertices)
        self.vertices = vertices

class Program:
    @staticmethod
    def create(vertex_shader_file, fragment_shader_file):
        vertex_shader = VertexShader(vertex_shader_file=vertex_shader_file)
        fragment_shader = FragmentShader(fragment_shader_file=fragment_shader_file)
        program = Program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program

    def __init__(self, vertex_shader=None, fragment_shader=None):
        self.vertex_shader = vertex_shader or VertexShader()
        self.fragment_shader = fragment_shader or FragmentShader()
        self.program = None
        self.link()
        self.showInfo()

    def link(self):
        self.program = glCreateProgram()
        if self.vertex_shader.isCompiled() and self.fragment_shader.isCompiled():
            glAttachShader(self.program, self.vertex_shader.shader)
            glAttachShader(self.program, self.fragment_shader.shader)
            self.vertex_shader.delete()
            self.fragment_shader.delete()
            glLinkProgram(self.program)

    def isLinked(self):
        return glGetProgramiv(self.program, GL_LINK_STATUS) == GL_TRUE

    def delete(self):
        glDeleteProgram(self.program)

    def showInfo(self):
        log("【Program : {!r}】".format(self))
        log("program is linked : {!r}".format(self.isLinked()))
        log("vertex shader is compiled : {!r}".format(self.vertex_shader.isCompiled()))
        log("fragment shader is compiled : {!r}".format(self.fragment_shader.isCompiled()))
        if not self.vertex_shader.isCompiled():
            printShaderInfoLog(self.vertex_shader.shader)
        if not self.fragment_shader.isCompiled():
            printShaderInfoLog(self.fragment_shader.shader)

class Shader:
    def __init__(self, shader_type, shader_file):
        self.shader_type = shader_type
        self.shader_file = shader_file
        self.shader = None
        self.compile()

    def compile(self):
        # シェーダーファイルからソースコードを読み込む
        with open(self.shader_file, 'r', encoding='utf-8') as f:
            shader_src = f.read()

        # 作成したシェーダオブジェクトにソースコードを渡しコンパイルする
        self.shader = glCreateShader(self.shader_type)
        glShaderSource(self.shader, shader_src)
        glCompileShader(self.shader)

    def isCompiled(self):
        return glGetShaderiv(self.shader, GL_COMPILE_STATUS) == GL_TRUE

    def delete(self):
        glDeleteShader(self.shader)

class VertexShader(Shader):
    def __init__(self, vertex_shader_file):
        super(VertexShader, self).__init__(GL_VERTEX_SHADER, vertex_shader_file)

class FragmentShader(Shader):
    def __init__(self, fragment_shader_file):
        super(FragmentShader, self).__init__(GL_FRAGMENT_SHADER, fragment_shader_file)

class WindowSizeManager:
    def __init__(self, context, frames=[]):
        self.context = context
        self.frames = {k : (-1, 1, 1, -1) for k in frames}#自動調整するフレーム
        self.context.on_resize.append(self.on_resize)

    def on_resize(self, window, width, height):
        #print("window size manager", width, height)
        for frame, param in self.frames.items():
            x0, y0, x1, y1 = param
            frame.setPosition(
                width * (x0 + 1) / 2,
                height * (1 - y0) / 2
            )
            frame.setSize(
                width * (x1 - x0 + 1) / 2,
                height * (1 - y1 + y0) / 2,
            )

    def addAutoSizeFrame(self, frame, x0, y0, x1, y1):
        self.frames[frame] = (x0, y0, x1, y1)

    def removeFrame(self, frame):
        del self.frames[frame]


DEBUG = True
def log(*str, **arg):
    if DEBUG:
        print(*str, **arg)

def run():
    app = App()
    app.context.show_version()
    app.run()

if __name__ == '__main__':
    run()
