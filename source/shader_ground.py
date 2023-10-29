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

def createShader():
    obj_vshader = """
        #version 330 core
        layout (location = 0) in vec3 vtx_pos;
        layout (location = 1) in vec3 vtx_normal;

        uniform mat4 mvpMat;
        uniform mat3 normMat;

        out vec3 normal;
        out vec2 grid_uv;

        void main()
        {
            normal = normMat * vtx_normal;
            grid_uv = vtx_pos.xz;
            vec4 pos = vec4(vtx_pos.xyz, 1.0);
            gl_Position = mvpMat * pos;
        }
    """
    
    obj_fshader = """
        #version 330 core
        out vec4 color;

        in vec3 normal;
        in vec2 grid_uv;

        uniform vec4 color_0;
        uniform vec4 color_1;

        bool isGridPoint(in vec2 guv, in vec2 duv_dx, in vec2 duv_dy, in float tick)
        {
            vec2 guv0 = grid_uv - 0.5*vec2(duv_dx.x, duv_dy.y);
            vec2 guv1 = grid_uv + 0.5*vec2(duv_dx.x, duv_dy.y);

            vec2 t_guv = floor(guv / tick);
            bool grid = t_guv.x != floor(guv0.x / tick)
                    ||  t_guv.x != floor(guv1.x / tick)
                    ||  t_guv.y != floor(guv0.y / tick)
                    ||  t_guv.y != floor(guv1.y / tick);

            return grid;
        }

        bool isPolarGridPoint(in float tick)
        {
            float r = sqrt(dot(grid_uv,grid_uv));
            float drdx = 0.5 * dFdx(r);
            float drdy = 0.5 * dFdy(r);

            float t_r = floor(r / tick);
            bool grid = t_r != floor((r-drdx) / tick)
                    ||  t_r != floor((r+drdx) / tick)
                    ||  t_r != floor((r-drdy) / tick)
                    ||  t_r != floor((r+drdy) / tick);

            return grid;
        }

        void main()
        {
            vec2 duv_dx = dFdx(grid_uv);
            vec2 duv_dy = dFdy(grid_uv);

            bool fragKill = true;

            color = vec4(0., 0., 0., 0.);
            if(isPolarGridPoint(10.)) {
                color = color*color.w + vec4(.0, .4, .2, 1.);
                fragKill = false;
            }
            if(isGridPoint(grid_uv, duv_dx, duv_dy, 10.)) {
                color = color*color.w + vec4(.9, .9, .9, 1.);
                fragKill = false;
            }
            if(isGridPoint(grid_uv, duv_dx, duv_dy, 1.)) {
                color = color*color.w + vec4(.2, .2, .2, 1.);
                fragKill = false;
            }
            color = clamp(color, 0., 1.);

            if(fragKill)
                discard;
        }
        """

    vertexshader = shaders.compileShader(obj_vshader, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(obj_fshader, GL_FRAGMENT_SHADER)
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

if __name__ == '__main__':
    import glfw
    glfw.init()
    window = glfw.create_window(64,64,"Shader Test", None, None)    
    glfw.make_context_current(window)

    createShader()
