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
from OpenGL.GL import shaders
from OpenGL.GLU import *
import numpy as np
import glm
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geom_tools

def createShader():
    obj_vshader = """
        #version 330 core
        layout (location = 0) in vec3 vtx_pos;
        layout (location = 1) in vec3 vtx_normal;
        layout (location = 2) in vec4 vtx_v4;

        uniform mat4 mvpMat;
        uniform mat3 normMat;
        uniform vec4 size_xxyz;

        out vec3 normal;

        void main()
        {
            normal = normMat * vtx_normal;
            vec4 pos = vec4(vtx_pos.xyz, 1.0);
            if(vtx_v4.x == -1.) {
                pos.x = -size_xxyz.x;
            } else
            if(vtx_v4.x == +1.) {
                pos.x =  size_xxyz.y;
            } else {
                pos.y *= size_xxyz.z;
                pos.z *= size_xxyz.w;
            }
            
            gl_Position = mvpMat * pos;
        }
    """
    
    obj_fshader = """
        #version 330 core
        out vec4 color;

        in vec3 normal;

        uniform vec4 color_0;

        void main()
        {
            float d = dot(normal, vec3(0.5774, 0.5774, 0.5774));
            color.xyz = color_0.xyz * (0.5 + 0.5*d);
            color.w = 1.;
        }
        """

    vertexshader = shaders.compileShader(obj_vshader, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(obj_fshader, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    shaderLocs = {
        'mvpMat'  : glGetUniformLocation(shaderProgram, "mvpMat"),
        'normMat' : glGetUniformLocation(shaderProgram, "normMat"),
        'color_0' : glGetUniformLocation(shaderProgram, "color_0"),
        'size_xxyz' : glGetUniformLocation(shaderProgram, "size_xxyz"),
    }

    shaderInfo = {
        'program': shaderProgram,
        'locs'   : shaderLocs,
    }

    return shaderInfo

class Bone:

    def __init__(self, size_xxyz=[0.4, 5.0, 0.5, 0.5]):

        hxyz = (1, 1, 1)
        cxyz = (0,0,0)
        self.initVBIB(hxyz, cxyz)

        self.color_0 = [0,1,0,1]
        self.size_xxyz = size_xxyz

        self.shaderInfo     = createShader()
        self.shaderProgram  = self.shaderInfo['program']
        self.shader_mvpMat  = self.shaderInfo['locs']["mvpMat"]
        self.shader_normMat = self.shaderInfo['locs']["normMat"]

    def initVBIB(self, sizes, centerPos):

        numRingVertices = 4
        self.vtxStride = 3+3+4
        self.vtxData, self.idxData = geom_tools.generateDoublePyramid(self.vtxStride, numRingVertices, sizes, centerPos)

        for vi in range(numRingVertices):

            angle0 = 2*math.pi *  vi    / numRingVertices
            angle1 = 2*math.pi * (vi+1) / numRingVertices
            
            y0 = centerPos[1] + sizes[1] * math.cos(angle0)
            z0 = centerPos[2] + sizes[2] * math.sin(angle0)
            y1 = centerPos[1] + sizes[1] * math.cos(angle1)
            z1 = centerPos[2] + sizes[2] * math.sin(angle1)

            vi6s = (vi*6+0) * self.vtxStride + 6
            self.vtxData[vi6s:(vi6s+4)] = [-1.,  0.,  0.,  0.]
            vi6s = (vi*6+1) * self.vtxStride + 6
            self.vtxData[vi6s:(vi6s+4)] = [ 0.,  0.,  0.,  0.]
            vi6s = (vi*6+2) * self.vtxStride + 6
            self.vtxData[vi6s:(vi6s+4)] = [ 0.,  0.,  0.,  0.]
            vi6s = (vi*6+3) * self.vtxStride + 6
            self.vtxData[vi6s:(vi6s+4)] = [ 0.,  0.,  0.,  0.]
            vi6s = (vi*6+4) * self.vtxStride + 6
            self.vtxData[vi6s:(vi6s+4)] = [ 0.,  0.,  0.,  0.]
            vi6s = (vi*6+5) * self.vtxStride + 6
            self.vtxData[vi6s:(vi6s+4)] = [ 1.,  0.,  0.,  0.]

        self.boxVAO = glGenVertexArrays(1)
        glBindVertexArray(self.boxVAO)

        self.boxVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.boxVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vtxData.nbytes, None, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.boxVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.vtxData.nbytes, self.vtxData)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(3*4))
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(6*4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.boxIBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.idxData.nbytes, self.idxData, GL_STATIC_DRAW)

    def render(self, length, modlMat, viewMat, projMat):

        glUseProgram(self.shaderProgram)

        mvpMat = projMat * viewMat * modlMat
        normMat4 = glm.transpose(glm.inverse(viewMat))
        normMat = glm.mat3(normMat4)
        glUniformMatrix4fv(self.shader_mvpMat, 1, GL_FALSE, glm.value_ptr(mvpMat))
        glUniformMatrix3fv(self.shader_normMat, 1, GL_FALSE, glm.value_ptr(normMat))

        glUniform4fv(self.shaderInfo['locs']["color_0"], 1, self.color_0)
        self.size_xxyz[1] = length
        glUniform4fv(self.shaderInfo['locs']["size_xxyz"], 1, self.size_xxyz)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.boxVAO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
        glDrawElements(GL_TRIANGLES, len(self.idxData), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def render_AtoB(self, posA, posB, viewMat, projMat):

        glUseProgram(self.shaderProgram)

        vecAB = posB - posA
        self.size_xxyz[1] = glm.length(vecAB)
        vecAB = glm.normalize(vecAB)
        rotVec = glm.cross(glm.vec3(1,0,0), vecAB)
        angle = math.asin(glm.length(rotVec))
        if(vecAB[0] < 0):
            angle = math.pi - angle
        modlMat = glm.translate(glm.identity(glm.mat4), posA)
        modlMat = glm.rotate(modlMat, angle, glm.normalize(rotVec))

        mvpMat = projMat * viewMat * modlMat
        normMat4 = glm.transpose(glm.inverse(viewMat))
        normMat = glm.mat3(normMat4)
        glUniformMatrix4fv(self.shader_mvpMat, 1, GL_FALSE, glm.value_ptr(mvpMat))
        glUniformMatrix3fv(self.shader_normMat, 1, GL_FALSE, glm.value_ptr(normMat))

        glUniform4fv(self.shaderInfo['locs']["color_0"], 1, self.color_0)
        glUniform4fv(self.shaderInfo['locs']["size_xxyz"], 1, self.size_xxyz)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.boxVAO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
        glDrawElements(GL_TRIANGLES, len(self.idxData), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

if __name__ == '__main__':
    import glfw
    import imgui
    import mesh_sphere
    import shader_default
    import obj_cframe
    from imgui.integrations.glfw import GlfwRenderer

    imgui.create_context()

    window_w = 512
    window_h = 512
    glfw.init()
    glfw_window = glfw.create_window(window_w, window_h,"Object Test Renderer", None, None)    
    glfw.make_context_current(glfw_window)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    glfw_renderer = GlfwRenderer(glfw_window)
    io = imgui.get_io()
    glfw_renderer.refresh_font_texture()

    posB = glm.vec3(-1,-2,-3)
    posA = glm.vec3( 2, 1, 3)
    bone = Bone()

    cframe = obj_cframe.CFrame()
    sphere = mesh_sphere.Sphere(.5,(0,0,0), 2, False, shader_default.createShader())

    viewMat = glm.translate(glm.mat4(), glm.vec3(0,0,-10))
    modlMat = glm.identity(glm.mat4)

    prev_mouse_pos = imgui.Vec2(0,0)
    prev_mouse_down = False

    renderFrameIndex = 0

    while not glfw.window_should_close(glfw_window):
        window_size = glfw.get_window_size(glfw_window)
        glViewport(0,0,window_size[0],window_size[1])
        ar = window_size[0] / float(window_size[1])
        projMat = glm.perspective(45., ar, .1, 1000.0)

        glfw.poll_events()
        glfw_renderer.process_inputs()

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #modlMat = glm.rotate(modlMat, .03, glm.vec3(1,1,1))
        projMat = glm.perspective(45., window_w/float(window_h), .1, 1000.0)

        if False:
            L = 3 + 0.5*math.sin(renderFrameIndex*0.1)
            bone.render(L, modlMat, viewMat, projMat)
        else:
            modlMat = glm.translate(glm.identity(glm.mat4), posA)
            sphere.render(modlMat, viewMat, projMat)
            modlMat = glm.translate(glm.identity(glm.mat4), posB)
            sphere.render(modlMat, viewMat, projMat)
            bone.render_AtoB(posA, posB, viewMat, projMat)
        cframe.render(viewMat, projMat)

        imgui.new_frame()
        mouse_pos = imgui.core.get_mouse_position()
        mouse_down = imgui.core.is_mouse_down(0)
        if mouse_down and prev_mouse_down:
            dx = mouse_pos.x - prev_mouse_pos.x
            dy = mouse_pos.y - prev_mouse_pos.y
            viewMat = glm.rotate(viewMat, dx*0.005, glm.vec3(glm.row(viewMat,1)))
            viewMat = glm.rotate(viewMat, dy*0.005, glm.vec3(glm.row(viewMat,0)))
        prev_mouse_pos  = mouse_pos
        prev_mouse_down = mouse_down

        imgui.render()
        glfw_renderer.render(imgui.get_draw_data())

        glfw.swap_buffers(glfw_window)

        renderFrameIndex += 1

    glfw.terminate()
