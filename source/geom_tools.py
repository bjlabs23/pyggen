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

def nextCorner(c):
    c = int(c)
    return int(int(int(c/3)*3)) + ((c+1)%3)

def prevCorner(c):
    c = int(c)
    return int(int(int(c/3)*3)) + ((c+2)%3)

def buildOpp(idx):

    s = []

    for c in range(len(idx)):
        vn = idx[nextCorner(c)]
        vp = idx[prevCorner(c)]
        if(vn<vp):
            key = vn + (int(vp)<<32)
        else:
            key = vp + (int(vn)<<32)
        
        s.append((key,c))

    def comp(e):
        return e[0]
    s.sort(key=comp)

    opp = - np.ones(len(idx), np.int32)
    numManifoldEdges = 0

    for i in range(1,len(s)):
        if(s[i][0]==s[i-1][0]):
            opp[s[i  ][1]] = s[i-1][1]
            opp[s[i-1][1]] = s[i  ][1]
            numManifoldEdges += 1

    numBoundaryEdges = len(idx) - numManifoldEdges * 2

    return opp, numBoundaryEdges, numManifoldEdges

def subdivide(stride, vtx0, idx0):
    opp, numBoundaryEdges, numManifoldEdges = buildOpp(idx0)
    numNewVertices = numManifoldEdges + numBoundaryEdges
    numTriangles = int(len(idx0) / 3)

    vei = int(len(vtx0) / stride)
    newVe = - np.ones(len(idx0), np.int32)

    vtx = np.zeros(len(vtx0) + numNewVertices*stride, np.float32)
    vtx[0:len(vtx0)] = vtx0
    idx = np.zeros(numTriangles*4*3, np.uint32)
    trNewV = [0,0,0]

    for c in range(len(idx0)):
        cn = nextCorner(c)
        cp = prevCorner(c)
        co = opp[c]
        assert co == -1 or opp[co] == c

        if newVe[c]==-1 and (co==-1 or newVe[co]==-1):
            next_pos = idx0[cn] * stride
            prev_pos = idx0[cp] * stride
            for si in range(stride):
                vtx[vei*stride + si] = (vtx0[next_pos + si] + vtx0[prev_pos + si]) * 0.5

            newVe[c] = vei
            if co >= 0:
                newVe[co] = vei
            vei += 1

        trNewV[c%3] = newVe[c]

        ni = int(int(c/3) * int(3*4))

        if int(c%3) == 2:
            v0 = idx0[int(c/3)*3  ]   #       v0
            v1 = idx0[int(c/3)*3+1]   #      /  \     .
            v2 = idx0[int(c/3)*3+2]   #     /    \    .
            e0 = trNewV[0]            #    e2-----e1
            e1 = trNewV[1]            #   /  \  /  \  .
            e2 = trNewV[2]            #  v1 --e0--- v2

            idx[ni:ni+12] = [e1,e2,e0,  v0,e2,e1,  e2,v1,e0,  e1,e0,v2]

    return vtx, idx

def computeNormal(vtxData, normalPos, vtxStride, idxData):

    numTr = int(len(idxData) / 3)
    numVtx = int(len(vtxData) / vtxStride)
    counts = np.zeros((numVtx), np.float32)

    for vi in range(numVtx):
        vtxData[vi*vtxStride + normalPos + 0] = 0
        vtxData[vi*vtxStride + normalPos + 1] = 0
        vtxData[vi*vtxStride + normalPos + 2] = 0
        counts[vi] = 0

    for ti in range(numTr):
        v0 = idxData[ti*3 + 0]
        v1 = idxData[ti*3 + 1]
        v2 = idxData[ti*3 + 2]

        p0 = glm.vec3(vtxData[v0*vtxStride], vtxData[v0*vtxStride + 1], vtxData[v0*vtxStride + 2])
        p1 = glm.vec3(vtxData[v1*vtxStride], vtxData[v1*vtxStride + 1], vtxData[v1*vtxStride + 2])
        p2 = glm.vec3(vtxData[v2*vtxStride], vtxData[v2*vtxStride + 1], vtxData[v2*vtxStride + 2])

        n = glm.cross(p1-p0, p2-p1)
        n = glm.normalize(n)

        vtxData[v0*vtxStride + normalPos + 0] += n.x
        vtxData[v0*vtxStride + normalPos + 1] += n.y
        vtxData[v0*vtxStride + normalPos + 2] += n.z
        vtxData[v1*vtxStride + normalPos + 0] += n.x
        vtxData[v1*vtxStride + normalPos + 1] += n.y
        vtxData[v1*vtxStride + normalPos + 2] += n.z
        vtxData[v2*vtxStride + normalPos + 0] += n.x
        vtxData[v2*vtxStride + normalPos + 1] += n.y
        vtxData[v2*vtxStride + normalPos + 2] += n.z

        counts[v0] += 1
        counts[v1] += 1
        counts[v2] += 1

    for vi in range(numVtx):
        vtxData[vi*vtxStride + normalPos + 0] /= counts[vi]
        vtxData[vi*vtxStride + normalPos + 1] /= counts[vi]
        vtxData[vi*vtxStride + normalPos + 2] /= counts[vi]

    return vtxData

def computeNormal_allVerticesSplit(vtxData, normalPos, vtxStride, idxData):

    numTr = int(len(idxData) / 3)
    numVtx = int(len(vtxData) / vtxStride)
    assert numVtx == numTr * 3

    for ti in range(numTr):
        v0 = idxData[ti*3 + 0]
        v1 = idxData[ti*3 + 1]
        v2 = idxData[ti*3 + 2]

        p0 = glm.vec3(vtxData[v0*vtxStride], vtxData[v0*vtxStride + 1], vtxData[v0*vtxStride + 2])
        p1 = glm.vec3(vtxData[v1*vtxStride], vtxData[v1*vtxStride + 1], vtxData[v1*vtxStride + 2])
        p2 = glm.vec3(vtxData[v2*vtxStride], vtxData[v2*vtxStride + 1], vtxData[v2*vtxStride + 2])

        n = glm.cross(p1-p0, p2-p1)
        n = glm.normalize(n)

        vtxData[v0*vtxStride + normalPos + 0] = n.x
        vtxData[v0*vtxStride + normalPos + 1] = n.y
        vtxData[v0*vtxStride + normalPos + 2] = n.z
        vtxData[v1*vtxStride + normalPos + 0] = n.x
        vtxData[v1*vtxStride + normalPos + 1] = n.y
        vtxData[v1*vtxStride + normalPos + 2] = n.z
        vtxData[v2*vtxStride + normalPos + 0] = n.x
        vtxData[v2*vtxStride + normalPos + 1] = n.y
        vtxData[v2*vtxStride + normalPos + 2] = n.z

    return vtxData

def mergeVtxIdx(vtxDataList, idxDataList, vtxStride):
    assert len(vtxDataList) == len(idxDataList)

    vtxDataMerged = np.concatenate(vtxDataList)
    idxDataMerged = np.concatenate(idxDataList)

    vtxOffset = 0
    vtxOffsets = []
    for vtxData in vtxDataList:
        vtxOffsets.append(vtxOffset)
        vtxOffset += int(len(vtxData) / vtxStride)

    idxDataOffset = len(idxDataList[0])
    for i in range(1, len(idxDataList)):
        numIdx = len(idxDataList[i])
        idxDataMerged[idxDataOffset:(idxDataOffset+numIdx)] += vtxOffsets[i]
        idxDataOffset += numIdx

    return vtxDataMerged, idxDataMerged

def translateVertices(vtxData, vtxStride, tr):
    numVtx = int(len(vtxData) / vtxStride)
    for v in range(numVtx):
        vs = v*vtxStride
        vtxData[vs+0] += tr[0]
        vtxData[vs+1] += tr[1]
        vtxData[vs+2] += tr[2]

def rotateVertices(vtxData, vtxStride, matR):
    numVtx = int(len(vtxData) / vtxStride)
    for v in range(numVtx):
        vs = v*vtxStride
        xyz = glm.vec3(vtxData[vs+0], vtxData[vs+1], vtxData[vs+2])
        xyz = matR * xyz
        vtxData[vs+0] = xyz.x
        vtxData[vs+1] = xyz.y
        vtxData[vs+2] = xyz.z

def setVertexData(vtxData, vtxStride, offset, data):
    numVtx = int(len(vtxData) / vtxStride)
    sz = len(data)
    for v in range(numVtx):
        vs = v*vtxStride + offset
        for i in range(sz):
            vtxData[vs + i] = data[i]

#####################################################################################################################

def generateUVPlane(vtxStride, hxyz, centerPos):

    assert vtxStride >= 3 + 3 + 2

    vtxData = np.zeros((4*vtxStride), np.float32)

    if hxyz[0] == 0:
        x  = centerPos[0]
        y0 = centerPos[1] - hxyz[1]
        y1 = centerPos[1] + hxyz[1]
        z0 = centerPos[2] - hxyz[2]
        z1 = centerPos[2] + hxyz[2]
        vtxData[0*vtxStride : (0*vtxStride+8)] = [x, y0, z0,  1., 0., 0.,   0., 0.,]
        vtxData[1*vtxStride : (1*vtxStride+8)] = [x, y1, z0,  1., 0., 0.,   1., 0.,]
        vtxData[2*vtxStride : (2*vtxStride+8)] = [x, y0, z1,  1., 0., 0.,   0., 1.,]
        vtxData[3*vtxStride : (3*vtxStride+8)] = [x, y1, z1,  1., 0., 0.,   1., 1.,]
    elif hxyz[1] == 0:
        x0 = centerPos[0] - hxyz[0]
        x1 = centerPos[0] + hxyz[0]
        y  = centerPos[1]
        z0 = centerPos[2] - hxyz[2]
        z1 = centerPos[2] + hxyz[2]
        vtxData[0*vtxStride : (0*vtxStride+8)] = [x0, y, z0,  0., 1., 0.,   0., 0.,]
        vtxData[1*vtxStride : (1*vtxStride+8)] = [x0, y, z1,  0., 1., 0.,   1., 0.,]
        vtxData[2*vtxStride : (2*vtxStride+8)] = [x1, y, z0,  0., 1., 0.,   0., 1.,]
        vtxData[3*vtxStride : (3*vtxStride+8)] = [x1, y, z1,  0., 1., 0.,   1., 1.,]
    elif hxyz[2] == 0:
        x0 = centerPos[0] - hxyz[0]
        x1 = centerPos[0] + hxyz[0]
        y0 = centerPos[1] - hxyz[1]
        y1 = centerPos[1] + hxyz[1]
        z  = centerPos[2]
        vtxData[0*vtxStride : (0*vtxStride+8)] = [x0, y0, z,  0., 0., 1.,   0., 0.,]
        vtxData[1*vtxStride : (1*vtxStride+8)] = [x1, y0, z,  0., 0., 1.,   1., 0.,]
        vtxData[2*vtxStride : (2*vtxStride+8)] = [x0, y1, z,  0., 0., 1.,   0., 1.,]
        vtxData[3*vtxStride : (3*vtxStride+8)] = [x1, y1, z,  0., 0., 1.,   1., 1.,]

    idxData = np.ndarray(shape=(1*(3+3)), dtype=np.uint32)
    for v in range(1):
        v4 = v*4
        idxData[v*vtxStride:((v+1)*vtxStride)] = (v4+0,v4+1,v4+2,  v4+2,v4+1,v4+3)

    return vtxData, idxData

def generateDisk(vtxStride, x, ry, rz, n, normal_to_positive_x):

    numVtx = 1 + n
    if normal_to_positive_x:
        normal = [ 1, 0, 0] 
    else:
        normal = [-1, 0, 0]

    vtxData = np.zeros(numVtx * vtxStride, np.float32)
    vtxData[0:3] = [x,0,0]
    vtxData[3:6] = normal
    vtxData[vtxStride+0 : vtxStride+3] = [0,ry,0]
    vtxData[vtxStride+3 : vtxStride+6] = normal

    for i in range(n):

        angle = 2*math.pi * i / n
        
        yr = ry * math.cos(angle)
        zr = rz * math.sin(angle)

        vis = (i+1) * vtxStride
        vtxData[(vis+0):(vis+3)] = [x, yr, zr]
        vtxData[(vis+3):(vis+6)] = normal

    idxData = np.zeros(n*3, np.int32)

    if normal_to_positive_x:
        for i in range(n):
            i1 = i + 1
            if i1 == n:
                i1 = 0
            idxData[i*3:i*3+3] = [0, i+1, i1+1]
    else:
        for i in range(n):
            i1 = i + 1
            if i1 == n:
                i1 = 0
            idxData[i*3:i*3+3] = [0, i1+1, i+1]

    return vtxData, idxData

def generateCone(vtxStride, x0, x1, ry, rz, n):

    numVtx = 1 + n

    vtxData = np.zeros(numVtx * vtxStride, np.float32)
    vtxData[0:3] = [x1,0,0]

    for i in range(n):

        angle = 2*math.pi * i / n
        
        yr = ry * math.cos(angle)
        zr = rz * math.sin(angle)

        vis = (i+1) * vtxStride
        vtxData[(vis+0):(vis+3)] = [x0, yr, zr]

    idxData = np.zeros(n*3, np.int32)

    for i in range(n):
        i1 = i + 1
        if i1 == n:
            i1 = 0
        idxData[i*3:i*3+3] = [0, i+1, i1+1]

    computeNormal(vtxData, 3, vtxStride, idxData)

    return vtxData, idxData


def generateTube(vtxStride, x0, ry0, rz0, x1, ry1, rz1, nS):

    numVtx = nS * 2

    vtxData = np.zeros(numVtx * vtxStride, np.float32)

    vis = 0
    for i in range(nS):

        angle = 2*math.pi * i / nS
        
        y0 = ry0 * math.cos(angle)
        z0 = rz0 * math.sin(angle)
        vtxData[(vis+0):(vis+3)] = [x0, y0, z0]
        vis += vtxStride
        
        y1 = ry1 * math.cos(angle)
        z1 = rz1 * math.sin(angle)
        vtxData[(vis+0):(vis+3)] = [x1, y1, z1]
        vis += vtxStride

    idxData = np.zeros(nS*2*3, np.int32)

    for i0 in range(nS):
        i1 = i0 + 1
        if i1 == nS:
            i1 = 0
        i6 = i0 * 6
        v00  = i0*2
        v01  = i0*2 + 1
        v10  = i1*2
        v11  = i1*2 + 1
        idxData[i6+0:i6+3] = [v00, v11, v01]
        idxData[i6+3:i6+6] = [v11, v00, v10]

    computeNormal(vtxData, 3, vtxStride, idxData)

    return vtxData, idxData

def generateOpenPyramid(vtxStride, numRingVertices, sizes, centerPos):

    xc = centerPos[0]
    yc = centerPos[1]
    zc = centerPos[2]
    x0 = centerPos[0] - sizes[0]
    x1 = centerPos[0] + sizes[0]

    vtxData = np.zeros(((numRingVertices*3) * vtxStride), np.float32)

    for vi in range(numRingVertices):

        angle0 = 2*math.pi *  vi    / numRingVertices
        angle1 = 2*math.pi * (vi+1) / numRingVertices
        
        y0 = centerPos[1] + sizes[1] * math.cos(angle0)
        z0 = centerPos[2] + sizes[2] * math.sin(angle0)
        y1 = centerPos[1] + sizes[1] * math.cos(angle1)
        z1 = centerPos[2] + sizes[2] * math.sin(angle1)

        vi6s = (vi*3+0) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, yc, zc,   0.,  0.,  0.]
        vi6s = (vi*3+1) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [xc, y1, z1,   0.,  0.,  0.]
        vi6s = (vi*3+2) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [xc, y0, z0,   0.,  0.,  0.]

    idxData = np.ndarray(shape=(numRingVertices*3), dtype=np.uint32)
    for v in range(numRingVertices):
        v3 = v*3
        idxData[v*3:(v*3+3)] = (v3+0,v3+1,v3+2)

    computeNormal_allVerticesSplit(vtxData, 3, vtxStride, idxData)

    return vtxData, idxData

def generateDoublePyramid(vtxStride, numRingVertices, sizes, centerPos):

    xc = centerPos[0]
    yc = centerPos[1]
    zc = centerPos[2]
    x0 = centerPos[0] - sizes[0]
    x1 = centerPos[0] + sizes[0]

    vtxData = np.zeros(((numRingVertices*6) * vtxStride), np.float32)

    for vi in range(numRingVertices):

        angle0 = 2*math.pi *  vi    / numRingVertices
        angle1 = 2*math.pi * (vi+1) / numRingVertices
        
        y0 = centerPos[1] + sizes[1] * math.cos(angle0)
        z0 = centerPos[2] + sizes[2] * math.sin(angle0)
        y1 = centerPos[1] + sizes[1] * math.cos(angle1)
        z1 = centerPos[2] + sizes[2] * math.sin(angle1)

        vi6s = (vi*6+0) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x0, yc, zc,   0.,  0.,  0.]
        vi6s = (vi*6+1) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [xc, y1, z1,   0.,  0.,  0.]
        vi6s = (vi*6+2) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [xc, y0, z0,   0.,  0.,  0.]
        vi6s = (vi*6+3) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [xc, y0, z0,   0.,  0.,  0.]
        vi6s = (vi*6+4) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [xc, y1, z1,   0.,  0.,  0.]
        vi6s = (vi*6+5) * vtxStride
        vtxData[vi6s:(vi6s+6)] = [x1, yc, zc,   0.,  0.,  0.]

    idxData = np.ndarray(shape=(numRingVertices*6), dtype=np.uint32)
    for v in range(numRingVertices*2):
        v3 = v*3
        idxData[v*3:(v*3+3)] = (v3+0,v3+1,v3+2)

    computeNormal_allVerticesSplit(vtxData, 3, vtxStride, idxData)

    return vtxData, idxData

def generateIco(vtxStride, r, centerPos, genTexUV):

    X = .525731112119133606
    Z = .850650808352039932
    N = 0.

    xyzData = np.asarray([
        -X, N, Z,    +X, N, Z,    -X, N,-Z,    +X, N,-Z,    +N, Z, X,    +N, Z,-X,
        +N,-Z, X,    +N,-Z,-X,    +Z, X, N,    -Z, X, N,    +Z,-X, N,    -Z,-X, N,
    ], np.float32)

    numVtx = int(len(xyzData)/3)
    vtxData = np.zeros((numVtx * vtxStride), np.float32)

    for i in range(numVtx):

        ist = i*vtxStride

        vtxData[ist + 0] = centerPos[0] + xyzData[i*3 + 0] * r
        vtxData[ist + 1] = centerPos[1] + xyzData[i*3 + 1] * r
        vtxData[ist + 2] = centerPos[2] + xyzData[i*3 + 2] * r

        vtxData[ist + 3] = xyzData[i*3 + 0]
        vtxData[ist + 4] = xyzData[i*3 + 1]
        vtxData[ist + 5] = xyzData[i*3 + 2]

    if genTexUV:
        for i in range(numVtx):
            ist = i*vtxStride

            nx = vtxData[ist + 3]
            ny = vtxData[ist + 4]
            nz = vtxData[ist + 5]

            vtxData[ist + 6] = (math.pi + math.atan2(ny,nx)) / (math.pi * 2)
            vtxData[ist + 7] = (math.pi/2 + math.atan(nz)) / math.pi

    idxData = np.asarray([
        0, 1, 4,     0, 4, 9,     9, 4, 5,     4, 8, 5,     4, 1, 8,
        8, 1,10,     8,10, 3,     5, 8, 3,     5, 3, 2,     2, 3, 7,
        7, 3,10,     7,10, 6,     7, 6,11,    11, 6, 0,     0, 6, 1,
        6,10, 1,     9,11, 0,     9, 2,11,     9, 5, 2,     7,11, 2
    ], np.uint32)

    return vtxData, idxData

def generateSphere(vtxStride, r, centerPos, numSubdiv, genTexUV):
    vtxData, idxData = generateIco(vtxStride, r, centerPos, genTexUV)

    for s in range(numSubdiv):
        vtxData, idxData = subdivide(vtxStride, vtxData, idxData)
        for i in range(int(len(vtxData) / vtxStride)):
            ist = i*vtxStride

            normal = glm.vec3(vtxData[ist] - centerPos[0], vtxData[ist+1] - centerPos[1], vtxData[ist+2] - centerPos[2])
            normal = glm.normalize(normal)

            vtxData[ist + 0] = centerPos[0] + r * normal.x
            vtxData[ist + 1] = centerPos[1] + r * normal.y
            vtxData[ist + 2] = centerPos[2] + r * normal.z

            vtxData[ist + 3] = normal.x
            vtxData[ist + 4] = normal.y
            vtxData[ist + 5] = normal.z

    if genTexUV:
        for i in range(int(len(vtxData) / vtxStride)):
            ist = i*vtxStride

            nx = vtxData[ist + 3]
            ny = vtxData[ist + 4]
            nz = vtxData[ist + 5]

            vtxData[ist + 6] = (math.pi + math.atan2(ny,nx)) / math.pi / 2
            vtxData[ist + 7] = (math.pi/2 + math.atan(nz)) / math.pi

    return vtxData, idxData

if __name__ == '__main__':
    import glfw
    glfw.init()
    window = glfw.create_window(64,64,"Shader Test", None, None)    
    glfw.make_context_current(window)

    import mesh_box
    vtxStride = 6
    vtxData, idxData = mesh_box.generateBox(vtxStride, (1,1,1),(0,0,0))    
    vtx, idx = subdivide(vtxStride, vtxData, idxData)
    print(vtx)
    print(idx)
