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

def fillVertexData_box(vtxData, vtxStride, hsizes, centerPos):

    x0 = centerPos[0] - hsizes[0]
    x1 = centerPos[0] + hsizes[0]
    y0 = centerPos[1] - hsizes[1]
    y1 = centerPos[1] + hsizes[1]
    z0 = centerPos[2] - hsizes[2]
    z1 = centerPos[2] + hsizes[2]

    #-x
    vtxData[( 0*vtxStride):( 0*vtxStride+6)] = [ x0, y1, z1, -1.,  0.,  0.,]
    vtxData[( 1*vtxStride):( 1*vtxStride+6)] = [ x0, y0, z1, -1.,  0.,  0.,]
    vtxData[( 2*vtxStride):( 2*vtxStride+6)] = [ x0, y1, z0, -1.,  0.,  0.,]
    vtxData[( 3*vtxStride):( 3*vtxStride+6)] = [ x0, y0, z0, -1.,  0.,  0.,]
    #+x
    vtxData[( 4*vtxStride):( 4*vtxStride+6)] = [ x1, y1, z1, +1.,  0.,  0.,]
    vtxData[( 5*vtxStride):( 5*vtxStride+6)] = [ x1, y1, z0, +1.,  0.,  0.,]
    vtxData[( 6*vtxStride):( 6*vtxStride+6)] = [ x1, y0, z1, +1.,  0.,  0.,]
    vtxData[( 7*vtxStride):( 7*vtxStride+6)] = [ x1, y0, z0, +1.,  0.,  0.,]
    #-y
    vtxData[( 8*vtxStride):( 8*vtxStride+6)] = [ x1, y0, z1,  0., -1.,  0.,]
    vtxData[( 9*vtxStride):( 9*vtxStride+6)] = [ x1, y0, z0,  0., -1.,  0.,]
    vtxData[(10*vtxStride):(10*vtxStride+6)] = [ x0, y0, z1,  0., -1.,  0.,]
    vtxData[(11*vtxStride):(11*vtxStride+6)] = [ x0, y0, z0,  0., -1.,  0.,]
    #+y
    vtxData[(12*vtxStride):(12*vtxStride+6)] = [ x1, y1, z1,  0., +1.,  0.,]
    vtxData[(13*vtxStride):(13*vtxStride+6)] = [ x0, y1, z1,  0., +1.,  0.,]
    vtxData[(14*vtxStride):(14*vtxStride+6)] = [ x1, y1, z0,  0., +1.,  0.,]
    vtxData[(15*vtxStride):(15*vtxStride+6)] = [ x0, y1, z0,  0., +1.,  0.,]
    #-z
    vtxData[(16*vtxStride):(16*vtxStride+6)] = [ x0, y0, z0,  0.,  0., -1.,]
    vtxData[(17*vtxStride):(17*vtxStride+6)] = [ x1, y0, z0,  0.,  0., -1.,]
    vtxData[(18*vtxStride):(18*vtxStride+6)] = [ x0, y1, z0,  0.,  0., -1.,]
    vtxData[(19*vtxStride):(19*vtxStride+6)] = [ x1, y1, z0,  0.,  0., -1.,]
    #+z
    vtxData[(20*vtxStride):(20*vtxStride+6)] = [ x0, y0, z1,  0.,  0., +1.,]
    vtxData[(21*vtxStride):(21*vtxStride+6)] = [ x0, y1, z1,  0.,  0., +1.,]
    vtxData[(22*vtxStride):(22*vtxStride+6)] = [ x1, y0, z1,  0.,  0., +1.,]
    vtxData[(23*vtxStride):(23*vtxStride+6)] = [ x1, y1, z1,  0.,  0., +1.,]

def generateBox(vtxStride, hsizes, centerPos):

    vtxData = np.zeros((4*6*vtxStride), np.float32)
    fillVertexData_box(vtxData, vtxStride, hsizes, centerPos)

    idxData = np.ndarray(shape=(6*(3+3)), dtype=np.uint32)
    for v in range(6):
        v4 = v*4
        idxData[v*6:(v*6+6)] = (v4+0,v4+1,v4+2,  v4+2,v4+1,v4+3)

    return vtxData, idxData

class Box:

    def __init__(self, shaderInfo):
        self.hxyz = [1,1,1]
        self.cxyz = [0,0,0]
        self.initBox(self.hxyz, self.cxyz)

        self.color_0 = [0,1,0,1]
        self.color_1 = [0,0,0,0]

        self.shaderInfo     = shaderInfo
        self.shaderProgram  = shaderInfo['program']
        self.shader_mvpMat  = shaderInfo['locs']["mvpMat"]
        self.shader_normMat = shaderInfo['locs']["normMat"]

    def __init__(self, hxyz, cxyz, shaderInfo):

        self.hxyz = hxyz
        self.cxyz = cxyz
        self.initBox(hxyz, cxyz)

        self.color_0 = [0,1,0,1]
        self.color_1 = [0,0,0,0]

        self.shaderInfo     = shaderInfo
        self.shaderProgram  = shaderInfo['program']
        self.shader_mvpMat  = shaderInfo['locs']["mvpMat"]
        self.shader_normMat = shaderInfo['locs']["normMat"]

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

    def initBox(self, hxyz, cxyz):

        self.hxyz = hxyz
        self.cxyz = cxyz

        self.vtxStride = 3 + 3
        self.vtxData, self.idxData = generateBox(self.vtxStride, hxyz, cxyz)
        self.boxVAO = glGenVertexArrays(1)
        self.boxVBO = glGenBuffers(1)
        self.boxIBO = glGenBuffers(1)
        self.uploadMesh()

    def uploadMesh(self):
        self.x0 = self.cxyz[0] - self.hxyz[0]
        self.x1 = self.cxyz[0] + self.hxyz[0]
        self.y0 = self.cxyz[1] - self.hxyz[1]
        self.y1 = self.cxyz[1] + self.hxyz[1]
        self.z0 = self.cxyz[2] - self.hxyz[2]
        self.z1 = self.cxyz[2] + self.hxyz[2]

        glBindVertexArray(self.boxVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.boxVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vtxData.nbytes, self.vtxData, GL_DYNAMIC_DRAW)
        #glBufferData(GL_ARRAY_BUFFER, self.vtxData.nbytes, None, GL_DYNAMIC_DRAW)
        #glBufferSubData(GL_ARRAY_BUFFER, 0, self.vtxData.nbytes, self.vtxData)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vtxStride*4, ctypes.c_void_p(3*4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.idxData.nbytes, self.idxData, GL_STATIC_DRAW)

    def updateSize(self, hxyz):
        self.hxyz = hxyz
        fillVertexData_box(self.vtxData, self.vtxStride, hxyz, self.cxyz)
        self.uploadMesh()

    def updateMesh(self, hxyz, cxyz):
        self.hxyz = hxyz
        self.cxyz = cxyz
        fillVertexData_box(self.vtxData, self.vtxStride, hxyz, cxyz)
        self.uploadMesh()

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
    import math
    import obj_cframe

    window_w = 512
    window_h = 512
    glfw.init()
    window = glfw.create_window(window_w, window_h,"Object Test Renderer", None, None)    
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    shaderInfo = shader_default.createShader()
    box = Box((1,0.4,0.6),(0,0,0), shaderInfo)
    cframe = obj_cframe.CFrame()

    viewMat = glm.translate(glm.mat4(), glm.vec3(0,0,-10))
    modlMat = glm.identity(glm.mat4)

    t = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()

        hsizes = [1 + 0.2*math.sin(2*t),  1 + 0.2*math.cos(5*t),  1 + 0.3*math.sin(2+3*t)]
        box.updateMesh(hsizes, box.cxyz)

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        modlMat = glm.rotate(modlMat, .03, glm.vec3(1,1,1))
        projMat = glm.perspective(45., window_w/float(window_h), .1, 1000.0)
        box.render(modlMat, viewMat, projMat)
        cframe.render(viewMat*modlMat, projMat)
        glfw.swap_buffers(window)

        t += 1/60
    glfw.terminate()
