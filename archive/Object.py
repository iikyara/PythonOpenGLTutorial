from OpenGL.GL import *

class Element:
    def __init__(self, size, vertices, indices):
        # 頂点配列オブジェクト
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # 頂点バッファオブジェクト
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        # 頂点バッファオブジェクトをin変数から参照できるようにする
        glVertexAttribPointer(0, size, GL_FLOAT, GL_FALSE, 0, 0)
        glEnableVertexAttribArray(0)
        # インテックスの頂点バッファオブジェクト
        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    def __del__(self):
        pass
        '''
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ibo])
        '''

    def __copy__(self):
        return None

    def __deepcopy__(self):
        return None

    def bind(self):
        glBindVertexArray(self.vao)

class Shape(Element):
    def __init__(self, size, vertices, indices):
        super(Shape, self).__init__(size, vertices, indices)
        self.indexcount = len(indices)

    def draw(self):
        super(Shape, self).bind()
        self.execute()

    def execute(self):
        glDrawElements(GL_LINES, self.indexcount, GL_UNSIGNED_INT, 0)
