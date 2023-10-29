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

        void main()
        {
            normal = normMat * vtx_normal;
            vec4 pos = vec4(vtx_pos.xyz, 1.0);
            gl_Position = mvpMat * pos;
        }
    """
    
    obj_fshader = """
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

def createShaderUV():
    obj_vshader = """
        #version 330 core
        layout (location = 0) in vec3 vtx_pos;
        layout (location = 1) in vec3 vtx_normal;
        layout (location = 2) in vec2 vtx_uv;

        uniform mat4 mvpMat;
        uniform mat3 normMat;

        out vec3 normal;
        out vec2 uv;

        void main()
        {
            normal = normMat * vtx_normal;
            vec4 pos = vec4(vtx_pos.xyz, 1.0);
            gl_Position = mvpMat * pos;

            uv = vtx_uv;
        }
    """
    
    obj_fshader = """
        #version 330 core
        out vec4 color;

        in vec3 normal;
        in vec2 uv;

        uniform vec4 color_0;
        uniform vec4 color_1;

        void main()
        {
            //color.xyz = 0.5 + 0.5*normal;
            float d = dot(normal, vec3(0.5774, 0.5774, 0.5774));
            color.xyz = color_0.xyz * (0.5 + 0.5*d)   +   0.01 * color_1.xyz;
            color.xy = uv;
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
        'color_1' : glGetUniformLocation(shaderProgram, "color_1"),
    }

    shaderInfo = {
        'program': shaderProgram,
        'locs'   : shaderLocs,
    }

    return shaderInfo

def createShaderUVTex():
    obj_vshader = """
        #version 330 core
        layout (location = 0) in vec3 vtx_pos;
        layout (location = 1) in vec3 vtx_normal;
        layout (location = 2) in vec2 vtx_uv;

        uniform mat4 mvpMat;
        uniform mat3 normMat;

        out vec3 normal;
        out vec2 uv;

        void main()
        {
            normal = normMat * vtx_normal;
            vec4 pos = vec4(vtx_pos.xyz, 1.0);
            gl_Position = mvpMat * pos;

            uv = vtx_uv;
        }
    """
    
    obj_fshader = """
        #version 330 core
        out vec4 color;

        in vec3 normal;
        in vec2 uv;

        uniform sampler2D tex_0;

        void main()
        {
            color.xyz = texture2D(tex_0, vec2(uv.x, 1.-uv.y)).xyz;
            //color.xy = uv;
            color.w = 1.;
        }
        """

    vertexshader = shaders.compileShader(obj_vshader, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(obj_fshader, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    shaderLocs = {
        'mvpMat'  : glGetUniformLocation(shaderProgram, "mvpMat"),
        'normMat' : glGetUniformLocation(shaderProgram, "normMat"),
        'tex_0'   : glGetUniformLocation(shaderProgram, "tex_0"),
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
    createShaderUV()
    createShaderUVTex()
