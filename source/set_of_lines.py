'''
MIT License

Copyright (c) 2023 BJLAB LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# pip install numpy, imgui[full], PyOpenGL, PyGLM

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
import numpy as np
import glm
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geom_tools
import tools_tex

def createShader():
    vshader = """
        #version 330 core
        layout (location = 0) in vec3 vtx_pos;
        layout (location = 1) in vec3 vtx_normal;

        uniform mat4 mvpMat;
        uniform mat3 normMat;

        out vec3 normal;

        void main()
        {
            normal = normMat * vtx_normal;
            vec4 pos = vec4(vtx_pos.xyz, 1.0);
            gl_Position = mvpMat * pos;
        }
    """
    
    fshader = """
        #version 330 core
        out vec4 color;

        in vec3 normal;

        uniform vec4 color_0;
        uniform vec4 color_1;

        void main()
        {
            //color.xyz = 0.5 + 0.5*normal;
            float d = dot(normal, vec3(0.5774, 0.5774, 0.5774));
            color.xyz = color_0.xyz * (0.5 + 0.5*d)   +   0.01 * color_1.xyz;
            color.w = 1.;
        }
        """

    vertexshader = shaders.compileShader(vshader, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(fshader, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    shaderLocs = {
        'mvpMat'  : glGetUniformLocation(shaderProgram, "mvpMat"),
        'normMat' : glGetUniformLocation(shaderProgram, "normMat"),
        'color_0' : glGetUniformLocation(shaderProgram, "color_0"),
        'color_1' : glGetUniformLocation(shaderProgram, "color_1"),
    }

    shaderInfo = {
        'program': shaderProgram,
        'locs'   : shaderLocs,
    }

    return shaderInfo

class LineSet:

    def __init__(self, maxNum=0):

        self.init(maxNum)

        self.color_0 = [0,1,0,1]
        self.color_1 = [0,0,0,0]

        self.shaderInfo = createShader()
        self.shaderProgram  = self.shaderInfo['program']
        self.shader_mvpMat  = self.shaderInfo['locs']["mvpMat"]
        self.shader_normMat = self.shaderInfo['locs']["normMat"]

    def init(self, maxNum):

        self.vtxStride = 3 + 3 + 2
        self.linesVAO = glGenVertexArrays(1)
        self.linesVBO = glGenBuffers(1)
        self.linesIBO = glGenBuffers(1)

        if maxNum > 0:
            self.generateLines([1,1,1],[0,0,0],maxNum)
            self.uploadMesh()

    def generateLines(self, hxyz, cxyz, maxNum):
        self.vtxData = np.zeros(2 * maxNum * self.vtxStride, np.float32)
        self.idxData = np.zeros(2 * maxNum, np.int32)

        for i in range(maxNum):
            vi0 = (i*2    ) * self.vtxStride
            vi1 = (i*2 + 1) * self.vtxStride
            self.vtxData[vi0+0] = random.uniform(cxyz[0]-hxyz[0], cxyz[0]+hxyz[0])
            self.vtxData[vi0+1] = random.uniform(cxyz[1]-hxyz[1], cxyz[1]+hxyz[1])
            self.vtxData[vi0+2] = random.uniform(cxyz[2]-hxyz[2], cxyz[2]+hxyz[2])
            self.vtxData[vi1+0] = random.uniform(cxyz[0]-hxyz[0], cxyz[0]+hxyz[0])
            self.vtxData[vi1+1] = random.uniform(cxyz[1]-hxyz[1], cxyz[1]+hxyz[1])
            self.vtxData[vi1+2] = random.uniform(cxyz[2]-hxyz[2], cxyz[2]+hxyz[2])

            self.idxData[i*2    ] = i*2
            self.idxData[i*2 + 1] = i*2 + 1

    def setLines(self, lines, scale, center):
        numLines = 0
        for line in lines:
            numLines += int(len(line) / 4)
        self.vtxData = np.zeros(2 * numLines * self.vtxStride, np.float32)
        self.idxData = np.zeros(2 * numLines, np.int32)

        for i in range(numLines):
            line = lines[i]
            vi0 = (i*2    ) * self.vtxStride
            vi1 = (i*2 + 1) * self.vtxStride
            self.vtxData[vi0+0] = (line[0] - center[0]) * scale
            self.vtxData[vi0+1] = 0
            self.vtxData[vi0+2] = (line[1] - center[1]) * scale
            self.vtxData[vi1+0] = (line[2] - center[0]) * scale
            self.vtxData[vi1+1] = 0
            self.vtxData[vi1+2] = (line[3] - center[1]) * scale

            self.idxData[i*2    ] = i*2
            self.idxData[i*2 + 1] = i*2 + 1

        self.uploadMesh()

    def uploadMesh(self):
        glBindVertexArray(self.linesVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.linesVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vtxData.nbytes, None, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.linesVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.vtxData.nbytes, self.vtxData)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(3*4))
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(6*4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.linesIBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.idxData.nbytes, self.idxData, GL_STATIC_DRAW)
    
    def render(self, modlMat, viewMat, projMat):

        glUseProgram(self.shaderProgram)

        mvpMat = projMat * viewMat * modlMat
        normMat4 = glm.transpose(glm.inverse(viewMat))
        normMat = glm.mat3(normMat4)
        glUniformMatrix4fv(self.shader_mvpMat, 1, GL_FALSE, glm.value_ptr(mvpMat))
        glUniformMatrix3fv(self.shader_normMat, 1, GL_FALSE, glm.value_ptr(normMat))
        glUniform4fv(self.shaderInfo['locs']["color_0"], 1, self.color_0)
        glUniform4fv(self.shaderInfo['locs']["color_1"], 1, self.color_1)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.linesVAO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.linesIBO)
        glDrawElements(GL_LINES, len(self.idxData), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

if __name__ == '__main__':
    import glfw
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
    import obj_cframe

    imgui.create_context()

    window_w = 1024
    window_h = 1024
    glfw.init()
    glfw_window = glfw.create_window(window_w, window_h,"Object Test Renderer", None, None)    
    glfw.make_context_current(glfw_window)
    glEnable(GL_DEPTH_TEST)

    glfw_renderer = GlfwRenderer(glfw_window)
    io = imgui.get_io()
    glfw_renderer.refresh_font_texture()

    cframe = obj_cframe.CFrame()
    lineSet = LineSet(1024)

    viewMat = glm.translate(glm.mat4(), glm.vec3(0,-2,-10))
    modlMat = glm.identity(glm.mat4)

    prev_mouse_pos = imgui.Vec2(0,0)
    prev_mouse_down = False

    while not glfw.window_should_close(glfw_window):
        window_size = glfw.get_window_size(glfw_window)
        glViewport(0,0,window_size[0],window_size[1])
        ar = window_size[0] / float(window_size[1])
        projMat = glm.perspective(45., ar, .1, 1000.0)

        glfw.poll_events()
        glfw_renderer.process_inputs()
        imgui.new_frame()
        imgui.begin("LineSet")
        imgui.text('Hello Lines')

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        lineSet.render(modlMat, viewMat, projMat)
        cframe.render(viewMat, projMat)

        mouse_pos = imgui.core.get_mouse_position()
        mouse_down = imgui.core.is_mouse_down(0)
        if imgui.core.is_window_focused() == False:
            if mouse_down and prev_mouse_down:
                dx = mouse_pos.x - prev_mouse_pos.x
                dy = mouse_pos.y - prev_mouse_pos.y
                viewMat = glm.rotate(viewMat, dx*0.001, glm.vec3(glm.row(viewMat,1)))
                viewMat = glm.rotate(viewMat, dy*0.001, glm.vec3(glm.row(viewMat,0)))
        prev_mouse_pos  = mouse_pos
        prev_mouse_down = mouse_down

        imgui.end()
        imgui.render()
        glfw_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(glfw_window)

    glfw.terminate()
