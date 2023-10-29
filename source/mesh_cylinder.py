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

def generateCylinder_capsSplit(numRingVertices, vtxStride, sizes, centerPos):

    vtxDisk0, idxDisk0 = geom_tools.generateDisk(vtxStride, -sizes[0], sizes[1],sizes[2], numRingVertices, False)
    vtxDisk1, idxDisk1 = geom_tools.generateDisk(vtxStride, +sizes[0], sizes[1],sizes[2], numRingVertices, True)
    geom_tools.translateVertices(vtxDisk0, vtxStride, [centerPos[0], centerPos[1], centerPos[2]])
    geom_tools.translateVertices(vtxDisk1, vtxStride, [centerPos[0], centerPos[1], centerPos[2]])

    vtxTube, idxTube = geom_tools.generateTube(vtxStride, -sizes[0], sizes[1],sizes[2], +sizes[0], sizes[1],sizes[2], numRingVertices)

    vtxData,idxData = geom_tools.mergeVtxIdx([vtxDisk0,vtxDisk1,vtxTube],[idxDisk0,idxDisk1,idxTube], vtxStride)

    return vtxData, idxData

def generateCylinder_allVerticesSplit(numRingVertices, vtxStride, sizes, centerPos):

    x0 = centerPos[0] - sizes[0]
    x1 = centerPos[0] + sizes[0]
    yc = centerPos[1]
    zc = centerPos[2]

    vtxData = np.zeros(((numRingVertices*2*3) * 2 * vtxStride), np.float32)

    vi = 0
    for rvi in range(numRingVertices):

        angle0 = 2*math.pi *  vi    / numRingVertices
        angle1 = 2*math.pi * (vi+1) / numRingVertices
        
        y0 = centerPos[1] + sizes[1] * math.cos(angle0)
        z0 = centerPos[2] + sizes[2] * math.sin(angle0)
        y1 = centerPos[1] + sizes[1] * math.cos(angle1)
        z1 = centerPos[2] + sizes[2] * math.sin(angle1)

        vi6s = (vi*6+0) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, yc, zc,   0.,  0.,  0.]
        vi6s = (vi*6+1) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, y1, z1,   0.,  0.,  0.]
        vi6s = (vi*6+2) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, y0, z0,   0.,  0.,  0.]
        vi6s = (vi*6+3) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x1, y0, z0,   0.,  0.,  0.]
        vi6s = (vi*6+4) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x1, y1, z1,   0.,  0.,  0.]
        vi6s = (vi*6+5) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x1, yc, zc,   0.,  0.,  0.]
        vi += 1

    for rvi in range(numRingVertices):

        angle0 = 2*math.pi *  vi    / numRingVertices
        angle1 = 2*math.pi * (vi+1) / numRingVertices
        
        y00 = centerPos[1] + sizes[1] * math.cos(angle0)
        z00 = centerPos[2] + sizes[2] * math.sin(angle0)
        y01 = centerPos[1] + sizes[1] * math.cos(angle1)
        z01 = centerPos[2] + sizes[2] * math.sin(angle1)

        y10 = centerPos[1] + sizes[1] * math.cos(angle0)
        z10 = centerPos[2] + sizes[2] * math.sin(angle0)
        y11 = centerPos[1] + sizes[1] * math.cos(angle1)
        z11 = centerPos[2] + sizes[2] * math.sin(angle1)

        vi6s = (vi*6+0) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, y00, z00,   0.,  0.,  0.]
        vi6s = (vi*6+1) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x1, y11, z11,   0.,  0.,  0.]
        vi6s = (vi*6+2) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x1, y10, z10,   0.,  0.,  0.]
        vi6s = (vi*6+3) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x1, y11, z11,   0.,  0.,  0.]
        vi6s = (vi*6+4) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, y00, z00,   0.,  0.,  0.]
        vi6s = (vi*6+5) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, y01, z01,   0.,  0.,  0.]
        vi += 1

    idxData = np.ndarray(shape=(numRingVertices*6*2), dtype=np.uint32)
    for v in range(numRingVertices*2*2):
        v3 = v*3
        idxData[v*3:(v*3+3)] = (v3+0,v3+1,v3+2)

    geom_tools.computeNormal_allVerticesSplit(vtxData, 3, vtxStride, idxData)

    return vtxData, idxData

class Cylinder:

    def __init__(self, hxyz, cxyz, shaderInfo):

        self.generateGeometry(hxyz, cxyz)

        self.color_0 = [0,1,0,1]
        self.color_1 = [0,0,0,0]

        self.shaderInfo     = shaderInfo
        self.shaderProgram  = shaderInfo['program']
        self.shader_mvpMat  = shaderInfo['locs']["mvpMat"]
        self.shader_normMat = shaderInfo['locs']["normMat"]

    def generateGeometry(self, sizes, centerPos):

        self.vtxStride = 3+3
        self.vtxData, self.idxData = generateCylinder_capsSplit(20, self.vtxStride, sizes, centerPos)
        self.boxVAO = glGenVertexArrays(1)
        glBindVertexArray(self.boxVAO)

        self.boxVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.boxVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vtxData.nbytes, None, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.boxVBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.vtxData.nbytes, self.vtxData)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(3*4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.boxIBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
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

    window_w = 512
    window_h = 512
    glfw.init()
    glfw_window = glfw.create_window(window_w, window_h,"Object Test Renderer", None, None)    
    glfw.make_context_current(glfw_window)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    glfw_renderer = GlfwRenderer(glfw_window)
    io = imgui.get_io()
    glfw_renderer.refresh_font_texture()

    shaderInfo = shader_default.createShader()
    cylinder = Cylinder((4, 1.6, 2.4), (0,0,0), shaderInfo)

    viewMat = glm.translate(glm.mat4(), glm.vec3(0,0,-10))
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

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #modlMat = glm.rotate(modlMat, .03, glm.vec3(1,1,1))
        projMat = glm.perspective(45., window_w/float(window_h), .1, 1000.0)
        cylinder.render(modlMat, viewMat, projMat)

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

    glfw.terminate()
