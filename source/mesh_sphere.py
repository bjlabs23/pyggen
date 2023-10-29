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
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geom_tools

class Sphere:

    def __init__(self, r, cxyz, numSubdiv, genTexUV, shaderInfo):
        self.numSubdiv = numSubdiv
        self.uv = genTexUV
        if genTexUV:
            self.vtxStride = 6 + 2
        else:
            self.vtxStride = 6
        self.initIco(r, cxyz, genTexUV)

        self.color_0 = (0,1,0,1)
        self.color_1 = (0,0,0,0)

        self.shaderInfo     = shaderInfo
        self.shaderProgram  = shaderInfo['program']

        self.pbl_specularTypeIndex = int(2)

        locs = shaderInfo['locs']
        if 'albedo' in locs:
            self.shader_modlMat   = locs["modlMat"]
            self.shader_viewMat   = locs["viewMat"]
            self.shader_projMat   = locs["projMat"] 
            self.shader_iparams   = locs['iparams']
            self.shader_fparams   = locs['fparams']
            self.shader_albedo    = locs['albedo']
            self.shader_material  = locs['material']
            self.shader_light_pos4 = locs['light_pos4']
            self.shader_light_col4 = locs['light_col4']
            self.light_pos4 = [2, 3, 6, 1,  -2, -3,  2, 1]
            self.light_col4 = [1, 1, 1, 0,   1, 1, 1, 0]
            self.material = [1., 1., 0., 0.5]
            self.albedo4  = [1., 1., 0., 1.0]
            self.metalic   = 0.5
            self.roughness = 0.8
            self.rim       = 0.2
        else:
            self.shader_mvpMat  = locs["mvpMat"]
        self.shader_normMat = locs["normMat"]

    def initIco(self, r, centerPos, genTexUV):

        self.vtxData, self.idxData = geom_tools.generateSphere(self.vtxStride, r, centerPos, self.numSubdiv, genTexUV)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vtxData.nbytes, None, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.vtxData.nbytes, self.vtxData)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(3*4))
        if self.uv:
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(6*4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.IBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.idxData.nbytes, self.idxData, GL_STATIC_DRAW)

    def render(self, modlMat, viewMat, projMat):

        glUseProgram(self.shaderProgram)

        mvpMat = projMat * viewMat * modlMat
        normMat4 = glm.transpose(glm.inverse(viewMat))
        normMat = glm.mat3(normMat4)
        glUniformMatrix3fv(self.shader_normMat, 1, GL_FALSE, glm.value_ptr(normMat))
        glUniformMatrix4fv(self.shader_mvpMat,  1, GL_FALSE, glm.value_ptr(mvpMat))

        glUniform4fv(self.shaderInfo['locs']["color_0"], 1, self.color_0)
        glUniform4fv(self.shaderInfo['locs']["color_1"], 1, self.color_1)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        glDrawElements(GL_TRIANGLES, len(self.idxData), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def render_pbr(self, modlMat, viewMat, projMat):

        glUseProgram(self.shaderProgram)

        mvMat = viewMat * modlMat
        normMat4 = glm.transpose(glm.inverse(mvMat))
        normMat = glm.mat3(normMat4)
        glUniformMatrix3fv(self.shader_normMat, 1, GL_FALSE, glm.value_ptr(normMat))
        glUniformMatrix4fv(self.shader_modlMat, 1, GL_FALSE, glm.value_ptr(modlMat))
        glUniformMatrix4fv(self.shader_viewMat, 1, GL_FALSE, glm.value_ptr(viewMat))
        glUniformMatrix4fv(self.shader_projMat, 1, GL_FALSE, glm.value_ptr(projMat))

        glUniform4iv(self.shader_iparams,  1, [self.pbl_specularTypeIndex,0,0,0])
        glUniform4fv(self.shader_fparams,  1, [1,0,0,0])
        glUniform4fv(self.shader_albedo,   1, self.albedo4)
        glUniform4fv(self.shader_material, 1, self.material)
        glUniform4fv(self.shader_light_pos4, 2, self.light_pos4)
        glUniform4fv(self.shader_light_col4, 2, self.light_col4)

        glUniform4fv(self.shaderInfo['locs']["material"], 1, self.material)
        glUniform4fv(self.shaderInfo['locs']["albedo"  ], 1, self.albedo4)

        tex = self.shaderInfo['tex']
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex['albedo'])
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex['normal'])
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, tex['factor'])
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, tex['brdf'])

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.IBO)
        glDrawElements(GL_TRIANGLES, len(self.idxData), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

if __name__ == '__main__':
    import glfw
    import shader_default

    window_w = 512
    window_h = 512
    glfw.init()
    window = glfw.create_window(window_w, window_h,"Object Test Renderer", None, None)    
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    genTexUV = True
    if genTexUV:
        shaderInfo = shader_default.createShaderUV()
    else:
        shaderInfo = shader_default.createShader()
    sphere = Sphere(2,(0,0,0), 3, genTexUV, shaderInfo)

    viewMat = glm.translate(glm.mat4(), glm.vec3(0,0,-10))
    modlMat = glm.identity(glm.mat4)
    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.1,0.1,0.4,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        modlMat = glm.rotate(modlMat, .03, glm.vec3(1,1,1))
        projMat = glm.perspective(45., window_w/float(window_h), .1, 1000.0)
        sphere.render(modlMat, viewMat, projMat)
        glfw.swap_buffers(window)

    glfw.terminate()
