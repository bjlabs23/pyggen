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
import numpy as np
import glm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geom_tools
import tools_tex

class TexPlane:

    def __init__(self, hxyz, cxyz, shaderInfo):

        self.init(hxyz, cxyz)

        self.shaderInfo     = shaderInfo
        self.shaderProgram  = shaderInfo['program']
        self.shader_mvpMat  = shaderInfo['locs']["mvpMat"]
        self.shader_normMat = shaderInfo['locs']["normMat"]
        self.shader_tex_0   = shaderInfo['locs']["tex_0"]

        self.textures = []
        self.textureIndex = 0

    def init(self, hxyz, cxyz):

        self.vtxStride = 3 + 3+ 2
        self.boxVAO = glGenVertexArrays(1)
        self.boxVBO = glGenBuffers(1)
        self.boxIBO = glGenBuffers(1)

        self.updateMesh(hxyz, cxyz)
        self.uploadMesh()

    def uploadMesh(self):
        glBindVertexArray(self.boxVAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.boxVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vtxData.nbytes, None, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.boxVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.vtxData.nbytes, self.vtxData)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(3*4))
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(6*4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.idxData.nbytes, self.idxData, GL_STATIC_DRAW)

    def updateSize(self, hxyz):
        self.updateMesh(hxyz, self.cxyz)

    def updateMesh(self, hxyz, cxyz):
        self.hxyz = hxyz
        self.cxyz = cxyz
        self.vtxData, self.idxData = geom_tools.generateUVPlane(self.vtxStride, hxyz, cxyz)
        self.x0 = cxyz[0] - hxyz[0]
        self.x1 = cxyz[0] + hxyz[0]
        self.y0 = cxyz[1] - hxyz[1]
        self.y1 = cxyz[1] + hxyz[1]
        self.z0 = cxyz[2] - hxyz[2]
        self.z1 = cxyz[2] + hxyz[2]
        self.uploadMesh()

    def rayAABB(self, rayPos, rayDir, x0,x1, y0,y1, z0,z1):

        def sort(a,b):
            if(a<=b): return a, b
            else:     return b, a

        x_min, x_max = sort((x0 - rayPos.x) / rayDir.x,
                            (x1 - rayPos.x) / rayDir.x)
        
        y_min, y_max = sort((y0 - rayPos.y) / rayDir.y,
                            (y1 - rayPos.y) / rayDir.y)
    
        z_min, z_max = sort((z0 - rayPos.z) / rayDir.z,
                            (z1 - rayPos.z) / rayDir.z)

        t_min = max(x_min, y_min, z_min)
        t_max = min(x_max, y_max, z_max)

        if t_max < 0:
            return False

        if t_min > t_max:
            return False

        return True;         

    def rayHit(self, rayPos, rayDir):

        hit = self.rayAABB(rayPos,rayDir, self.x0,self.x1, self.y0,self.y1, self.z0,self.z1)

        return hit

    def render(self, modlMat, viewMat, projMat):

        glUseProgram(self.shaderProgram)

        mvpMat = projMat * viewMat * modlMat
        normMat4 = glm.transpose(glm.inverse(viewMat))
        normMat = glm.mat3(normMat4)
        glUniformMatrix4fv(self.shader_mvpMat, 1, GL_FALSE, glm.value_ptr(mvpMat))
        glUniformMatrix3fv(self.shader_normMat, 1, GL_FALSE, glm.value_ptr(normMat))

        if self.textureIndex < len(self.textures):
            glUniform1i(self.shader_tex_0, 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.textures[self.textureIndex])

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.boxVAO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
        glDrawElements(GL_TRIANGLES, len(self.idxData), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

if __name__ == '__main__':
    import glfw
    import shader_default
    import imgui
    from imgui.integrations.glfw import GlfwRenderer

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

    shaderInfo = shader_default.createShaderUVTex()
    texplane = TexPlane((1,1,0),(0,0,0), shaderInfo)
    texplane.textures.append(tools_tex.createRandomTex(128,128))

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
        imgui.begin("TexPlane Test")
        imgui.text('Hello TexPlane')

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        texplane.render(modlMat, viewMat, projMat)

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
