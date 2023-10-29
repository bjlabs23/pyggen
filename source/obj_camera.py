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
import obj_cframe
import shader_default
import mesh_texplane

def createShader():
    obj_vshader = """
        #version 330 core
        layout (location = 0) in vec3 vtx_pos;
        layout (location = 1) in vec3 vtx_normal;
        layout (location = 2) in vec4 vtx_v4;

        uniform mat4 mvpMat;
        uniform mat3 normMat;
        uniform vec4 param_0;

        out vec3 normal;
        out float a;
        out float z;
        out vec3  bcoord;

        out float faceId;

        void main()
        {
            float zNear = param_0.x;
            float zFar  = param_0.y;
            vec2  FOV = param_0.zw;

            z = - vtx_pos.z * zFar;
            vec4 pos = vec4(vtx_pos.xy * zFar * FOV * 3.141592654/180. / 2, -z, 1.0);

            normal = normMat * vtx_normal;

            if(vtx_v4.x == 0) {
                a = 0.5;
                bcoord = vec3(1,0,0);
            } else
            if(vtx_v4.x == 1) {
                a = 0.2;
                bcoord = vec3(0,zFar,0);
            } else {
                bcoord = vec3(0,0,zFar);
                a = 0.2;
            }

            faceId = (vtx_v4.y);

            gl_Position = mvpMat * pos;
        }
    """
    
    obj_fshader = """
        #version 330 core
        out vec4 color;

        in vec3 normal;
        in float a;
        in float z;
        in vec3 bcoord;
        in float facdId;

        uniform vec4 param_0;
        uniform vec4 color_0;

        float rand()
        {
            vec2 s = gl_FragCoord.xy * vec2(facdId+1.,facdId+1.);
            return fract(sin(dot(s, vec2(12.9898, 78.233))) * 43758.5453);
        }

        float pattern()
        {
            vec2 s = gl_FragCoord.xy * vec2(facdId+1.,facdId+1.)*0.14537;
            return fract(dot(s,vec2(12.9898, 78.233)));
        }

        void main()
        {
            float zNear = param_0.x;

            vec2 dz = abs(vec2(dFdx(z), dFdy(z)));
            if (z < zNear && ((z+dz.x) > zNear  || (z+dz.y) > zNear)) {
                color = color_0;
                return;
            }

            if(z < zNear) {
                float c = min(bcoord.x, min(bcoord.y,bcoord.z));
                if(c > 0.03)
                    discard;
                color = color_0;
            } else {
                float c = min(bcoord.x, min(bcoord.y,bcoord.z));
                if(c > 0.03)
                    discard;

                color.w = color_0.a * a;
                if(pattern() > color.w) {
                    discard;
                }
                float d = dot(normal, vec3(0.5774, 0.5774, 0.5774));
                color.xyz = color_0.xyz;
            }
        }
        """

    vertexshader = shaders.compileShader(obj_vshader, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(obj_fshader, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    shaderLocs = {
        'mvpMat'  : glGetUniformLocation(shaderProgram, "mvpMat"),
        'normMat' : glGetUniformLocation(shaderProgram, "normMat"),
        'color_0' : glGetUniformLocation(shaderProgram, "color_0"),
        'param_0' : glGetUniformLocation(shaderProgram, "param_0"),
    }

    shaderInfo = {
        'program': shaderProgram,
        'locs'   : shaderLocs,
    }

    return shaderInfo

class Camera:

    def __init__(self, zNear = 4, FOV = (25,45)):

        self.zNear = zNear
        self.FOV = FOV
        self.zFar_zNearMultiple = 10

        self.initVBIB()

        self.color_0 = [0,1,0,1]
        self.param_0 = [self.zNear, self.zFar_zNearMultiple*self.zNear, self.FOV[0], self.FOV[1]]

        self.shaderInfo     = createShader()
        self.shaderProgram  = self.shaderInfo['program']
        self.shader_mvpMat  = self.shaderInfo['locs']["mvpMat"]
        self.shader_normMat = self.shaderInfo['locs']["normMat"]
        self.shader_color_0 = self.shaderInfo['locs']["color_0"]

    def initVBIB(self):

        self.vtxStride = 3+3+4

        sizes = [1,1,1]
        self.vtxData, self.idxData  = geom_tools.generateOpenPyramid(self.vtxStride, 4, sizes, (1,0,0))
        geom_tools.setVertexData(self.vtxData, self.vtxStride, 6, [0,0,0,0])

        pts = [[1,1], [-1,1], [-1,-1], [1,-1]]
        for i in range(4):
            i3 = i*3
            self.vtxData[(i3+0)*self.vtxStride + 0] = 0
            self.vtxData[(i3+0)*self.vtxStride + 1] = 0
            self.vtxData[(i3+0)*self.vtxStride + 2] = 0
            self.vtxData[(i3+0)*self.vtxStride + 6] = 0
            self.vtxData[(i3+0)*self.vtxStride + 7] = i

            self.vtxData[(i3+1)*self.vtxStride + 0] = pts[i][0]
            self.vtxData[(i3+1)*self.vtxStride + 1] = pts[i][1]
            self.vtxData[(i3+1)*self.vtxStride + 2] = -1 #zFar
            self.vtxData[(i3+1)*self.vtxStride + 6] = 1
            self.vtxData[(i3+1)*self.vtxStride + 7] = i

            self.vtxData[(i3+2)*self.vtxStride + 0] = pts[(i+1)%4][0]
            self.vtxData[(i3+2)*self.vtxStride + 1] = pts[(i+1)%4][1]
            self.vtxData[(i3+2)*self.vtxStride + 2] = -1 #zFar
            self.vtxData[(i3+2)*self.vtxStride + 6] = 2
            self.vtxData[(i3+2)*self.vtxStride + 7] = i

        geom_tools.computeNormal(self.vtxData, 3, self.vtxStride, self.idxData)

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

    def render(self, viewMat, projMat):

        glUseProgram(self.shaderProgram)

        mvpMat = projMat * viewMat
        normMat4 = glm.transpose(glm.inverse(viewMat))
        normMat = glm.mat3(normMat4)
        glUniformMatrix4fv(self.shader_mvpMat, 1, GL_FALSE, glm.value_ptr(mvpMat))
        glUniformMatrix3fv(self.shader_normMat, 1, GL_FALSE, glm.value_ptr(normMat))

        self.param_0 = [self.zNear, self.zFar_zNearMultiple*self.zNear, self.FOV[0], self.FOV[1]]
        glUniform4fv(self.shaderInfo['locs']["param_0"], 1, self.param_0)
        glUniform4fv(self.shaderInfo['locs']["color_0"], 1, self.color_0)

        #glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(self.boxVAO)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.boxIBO)
        glDrawElements(GL_TRIANGLES, len(self.idxData), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        #glEnable(GL_CULL_FACE)

if __name__ == '__main__':
    import glfw
    import imgui
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

    cframe = obj_cframe.CFrame()
    camera = Camera(1)

    viewMat = glm.translate(glm.mat4(), glm.vec3(0,0,-10))
    modlMat = glm.identity(glm.mat4)

    prev_mouse_pos = imgui.Vec2(0,0)
    prev_mouse_down = False

    renderFrameIndex = 0

    while not glfw.window_should_close(glfw_window):
        window_size = glfw.get_window_size(glfw_window)
        glViewport(0,0,window_size[0],window_size[1])
        ar = window_size[0] / float(window_size[1])

        glfw.poll_events()
        glfw_renderer.process_inputs()

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        projMat = glm.perspective(45., window_w/float(window_h), .1, 1000.0)

        cframe.render(viewMat, projMat)
        camera.render(viewMat, projMat)

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
