from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from pygltflib import GLTF2
from OpenGL.GL import *
import numpy as np
from PIL import Image
import io
from PyQt6.QtCore import QTimer
import math

class VRMWidget(QOpenGLWidget):
    def __init__(self, vrm_path, parent=None):
        super().__init__(parent)
        self.vrm_path = vrm_path
        self.gltf = None
        self.meshes = []  # (positions, indices, texcoords, tex_id, joints, weights)
        self.textures = {}  # image index -> OpenGL tex id
        self.skins = []  # skin info
        self.animations = []  # animation info
        self.joint_nodes = []  # joint node indices
        self.inverse_bind_matrices = []
        self.joint_matrices = []
        self.anim_time = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_animation)
        self.timer.start(33)  # ~30fps
        # 鼠标控制相关
        self._mouse_dragging = False
        self._mouse_last_pos = None
        self._mouse_yaw = 0.0
        self._mouse_pitch = 0.0
        self._mouse_override = False

    def initializeGL(self):
        print('[调试] VRMWidget initializeGL called')
        self.gltf = GLTF2().load_binary(self.vrm_path)
        glClearColor(0, 0, 0, 0)
        self._load_textures()
        self._parse_skins()
        self._parse_animations()
        self._parse_meshes()
        # 输出所有joint的node索引和名称
        if hasattr(self, 'joint_nodes') and self.joint_nodes:
            print('[调试] VRM骨骼joint列表:')
            for i, node_idx in enumerate(self.joint_nodes):
                node = self.gltf.nodes[node_idx]
                name = getattr(node, 'name', f'node_{node_idx}')
                print(f'  joint[{i}] = node_idx {node_idx}, name: {name}')
        # TODO: 解析 mesh、材质、纹理，上传到 OpenGL

    def _load_textures(self):
        # 解析所有图片并上传到OpenGL
        if not hasattr(self.gltf, 'images') or not self.gltf.images:
            return
        for idx, img in enumerate(self.gltf.images):
            if img.uri:
                # 外部图片（不常见于VRM）
                continue
            # VRM/glb内嵌图片
            bv = self.gltf.bufferViews[img.bufferView]
            buffer = self.gltf.buffers[bv.buffer]
            buffer_data = self.gltf.get_data_from_buffer_uri(buffer.uri)
            img_bytes = buffer_data[bv.byteOffset:bv.byteOffset+bv.byteLength]
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            img_data = np.array(pil_img)
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pil_img.width, pil_img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            self.textures[idx] = tex_id
        glBindTexture(GL_TEXTURE_2D, 0)

    def _parse_skins(self):
        self.skins = []
        self.joint_nodes = []
        self.inverse_bind_matrices = []
        if not hasattr(self.gltf, 'skins') or not self.gltf.skins:
            return
        for skin in self.gltf.skins:
            self.joint_nodes = skin.joints
            ibm_accessor = self.gltf.accessors[skin.inverseBindMatrices]
            ibm_view = self.gltf.bufferViews[ibm_accessor.bufferView]
            buffer = self.gltf.buffers[ibm_view.buffer]
            buffer_data = self.gltf.get_data_from_buffer_uri(buffer.uri)
            ibm_data = buffer_data[ibm_view.byteOffset:ibm_view.byteOffset+ibm_view.byteLength]
            self.inverse_bind_matrices = np.frombuffer(ibm_data, dtype=np.float32).reshape(-1, 4, 4)
            break  # 只取第一个skin

    def _parse_animations(self):
        self.animations = []
        if not hasattr(self.gltf, 'animations') or not self.gltf.animations:
            print('[调试] 没有找到动画数据')
            return
        print(f'[调试] 动画数量: {len(self.gltf.animations)}')
        # 只取第一个动画
        anim = self.gltf.animations[0]
        print(f'[调试] 动画通道数: {len(anim.channels)}')
        # 只支持rotation通道
        channels = []
        for ch in anim.channels:
            sampler = anim.samplers[ch.sampler]
            target = ch.target
            print(f'[调试] 通道target: node={target.node}, path={target.path}')
            if target.path == 'rotation':
                # 解析关键帧时间
                input_acc = self.gltf.accessors[sampler.input]
                input_view = self.gltf.bufferViews[input_acc.bufferView]
                buffer = self.gltf.buffers[input_view.buffer]
                buffer_data = self.gltf.get_data_from_buffer_uri(buffer.uri)
                times = np.frombuffer(buffer_data[input_view.byteOffset:input_view.byteOffset+input_view.byteLength], dtype=np.float32)
                # 解析四元数
                output_acc = self.gltf.accessors[sampler.output]
                output_view = self.gltf.bufferViews[output_acc.bufferView]
                buffer = self.gltf.buffers[output_view.buffer]
                buffer_data = self.gltf.get_data_from_buffer_uri(buffer.uri)
                quats = np.frombuffer(buffer_data[output_view.byteOffset:output_view.byteOffset+output_view.byteLength], dtype=np.float32).reshape(-1, 4)
                print(f'[调试] rotation关键帧数: {len(times)}')
                channels.append({'node': target.node, 'times': times, 'quats': quats})
        self.animations = channels
        print(f'[调试] rotation通道数: {len(self.animations)}')

    def _parse_meshes(self):
        # 只支持单buffer、单primitive的简单VRM
        if not self.gltf.meshes or not self.gltf.accessors:
            return
        buffer = self.gltf.buffers[0]
        buffer_data = self.gltf.get_data_from_buffer_uri(buffer.uri)
        self.meshes = []
        for mesh in self.gltf.meshes:
            for prim in mesh.primitives:
                # 解析顶点
                pos_accessor = self.gltf.accessors[prim.attributes.POSITION]
                pos_view = self.gltf.bufferViews[pos_accessor.bufferView]
                pos_data = buffer_data[pos_view.byteOffset:pos_view.byteOffset+pos_view.byteLength]
                positions = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 3)
                # 解析三角面索引
                idx_accessor = self.gltf.accessors[prim.indices]
                idx_view = self.gltf.bufferViews[idx_accessor.bufferView]
                idx_data = buffer_data[idx_view.byteOffset:idx_view.byteOffset+idx_view.byteLength]
                if idx_accessor.componentType == 5123:  # UNSIGNED_SHORT
                    indices = np.frombuffer(idx_data, dtype=np.uint16)
                elif idx_accessor.componentType == 5125:  # UNSIGNED_INT
                    indices = np.frombuffer(idx_data, dtype=np.uint32)
                else:
                    indices = np.frombuffer(idx_data, dtype=np.uint8)
                # 纹理坐标
                if hasattr(prim.attributes, "TEXCOORD_0"):
                    tex_accessor_idx = getattr(prim.attributes, "TEXCOORD_0")
                    tex_accessor = self.gltf.accessors[tex_accessor_idx]
                    tex_view = self.gltf.bufferViews[tex_accessor.bufferView]
                    tex_data = buffer_data[tex_view.byteOffset:tex_view.byteOffset+tex_view.byteLength]
                    texcoords = np.frombuffer(tex_data, dtype=np.float32).reshape(-1, 2)
                else:
                    texcoords = None
                # joints/weights
                if hasattr(prim.attributes, "JOINTS_0") and hasattr(prim.attributes, "WEIGHTS_0"):
                    joints_accessor_idx = getattr(prim.attributes, "JOINTS_0")
                    joints_accessor = self.gltf.accessors[joints_accessor_idx]
                    joints_view = self.gltf.bufferViews[joints_accessor.bufferView]
                    joints_data = buffer_data[joints_view.byteOffset:joints_view.byteOffset+joints_view.byteLength]
                    joints = np.frombuffer(joints_data, dtype=np.uint16).reshape(-1, 4)
                    weights_accessor_idx = getattr(prim.attributes, "WEIGHTS_0")
                    weights_accessor = self.gltf.accessors[weights_accessor_idx]
                    weights_view = self.gltf.bufferViews[weights_accessor.bufferView]
                    weights_data = buffer_data[weights_view.byteOffset:weights_view.byteOffset+weights_view.byteLength]
                    weights = np.frombuffer(weights_data, dtype=np.float32).reshape(-1, 4)
                else:
                    joints = None
                    weights = None
                # 材质贴图
                tex_id = None
                if prim.material is not None:
                    mat = self.gltf.materials[prim.material]
                    if hasattr(mat, 'pbrMetallicRoughness') and mat.pbrMetallicRoughness:
                        base_color = mat.pbrMetallicRoughness.baseColorTexture
                        if base_color and base_color.index is not None:
                            tex_idx = self.gltf.textures[base_color.index].source
                            tex_id = self.textures.get(tex_idx)
                self.meshes.append((positions, indices, texcoords, tex_id, joints, weights))

    def _advance_animation(self):
        self.anim_time += 0.033  # ~30fps
        self.update()

    def _compute_joint_matrices(self):
        joint_mats_local = [np.eye(4, dtype=np.float32) for _ in self.joint_nodes]
        joint_mats_global = [np.eye(4, dtype=np.float32) for _ in self.joint_nodes]
        # 鼠标控制优先
        if self.joint_nodes and len(self.joint_nodes) > 4:
            for head_idx in [3, 4]:
                if self._mouse_override and head_idx == 4:
                    yaw = self._mouse_yaw
                    pitch = self._mouse_pitch
                else:
                    yaw = math.sin(self.anim_time) * 0.5
                    pitch = math.sin(self.anim_time * 0.7) * 0.2
                rot_y = np.array([
                    [math.cos(yaw), 0, math.sin(yaw), 0],
                    [0, 1, 0, 0],
                    [-math.sin(yaw), 0, math.cos(yaw), 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                rot_x = np.array([
                    [1, 0, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch), 0],
                    [0, math.sin(pitch), math.cos(pitch), 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                joint_mats_local[head_idx] = rot_y @ rot_x @ joint_mats_local[head_idx]
        # 递归父子层级
        # 获取所有节点的父节点
        node_parents = {i: None for i in range(len(self.gltf.nodes))}
        for idx, node in enumerate(self.gltf.nodes):
            if hasattr(node, 'children') and node.children:
                for c in node.children:
                    node_parents[c] = idx
        # 计算每个joint的全局变换
        for i, node_idx in enumerate(self.joint_nodes):
            parent = node_parents.get(node_idx)
            if parent is not None and node_idx != parent:
                # 找到父joint在joint_nodes中的索引
                try:
                    parent_joint = self.joint_nodes.index(parent)
                    joint_mats_global[i] = joint_mats_global[parent_joint] @ joint_mats_local[i]
                except ValueError:
                    joint_mats_global[i] = joint_mats_local[i]
            else:
                joint_mats_global[i] = joint_mats_local[i]
        # 乘以inverseBindMatrices
        for i in range(len(joint_mats_global)):
            joint_mats_global[i] = joint_mats_global[i] @ self.inverse_bind_matrices[i]
        return joint_mats_global

    def paintGL(self):
        # 优化视角参数
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glLoadIdentity()
        glTranslatef(0, -0.2, -2.8)  # 下移并拉远摄像机
        glScalef(1.5, 1.5, 1.5)
        glRotatef(180, 0, 1, 0)  # 让人物正面朝向你
        glRotatef(20, 1, 0, 0)   # 微调俯仰角
        joint_mats = self._compute_joint_matrices() if self.joint_nodes else None
        for positions, indices, texcoords, tex_id, joints, weights in self.meshes:
            # 输出部分头部顶点的joints/weights信息
            if joints is not None and weights is not None and len(positions) > 0:
                print('[调试] mesh顶点0 joints:', joints[0], 'weights:', weights[0])
                print('[调试] mesh顶点100 joints:', joints[100] if len(joints) > 100 else joints[-1], 'weights:', weights[100] if len(weights) > 100 else weights[-1])
            if joint_mats is not None and joints is not None and weights is not None:
                skinned_pos = []
                for i, v in enumerate(positions):
                    p = np.zeros(4, dtype=np.float32)
                    v4 = np.array([v[0], v[1], v[2], 1.0], dtype=np.float32)
                    for j in range(4):
                        joint_idx = joints[i][j]
                        w = weights[i][j]
                        if joint_idx < len(joint_mats):
                            p += w * (joint_mats[joint_idx] @ v4)
                    skinned_pos.append(p[:3])
                positions_draw = np.array(skinned_pos)
            else:
                positions_draw = positions
            if tex_id:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex_id)
            else:
                glDisable(GL_TEXTURE_2D)
            glBegin(GL_TRIANGLES)
            for i, idx in enumerate(indices):
                if texcoords is not None:
                    uv = texcoords[idx]
                    glTexCoord2f(uv[0], uv[1])
                v = positions_draw[idx]
                glVertex3f(v[0], v[1], v[2])
            glEnd()
            if tex_id:
                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        # TODO: 绘制 VRM 模型

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h != 0 else 1
        # 简单正交投影
        glOrtho(-1.5*aspect, 1.5*aspect, -1.5, 1.5, 0.1, 10)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        if event.button() == 1:  # 左键
            self._mouse_dragging = True
            self._mouse_last_pos = event.position()
            self._mouse_override = True

    def mouseMoveEvent(self, event):
        if self._mouse_dragging and self._mouse_last_pos is not None:
            delta = event.position() - self._mouse_last_pos
            self._mouse_last_pos = event.position()
            self._mouse_yaw += delta.x() * 0.01  # 左右
            self._mouse_pitch += delta.y() * 0.01  # 上下
            self._mouse_pitch = max(-1.0, min(1.0, self._mouse_pitch))  # 限制俯仰
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == 1:
            self._mouse_dragging = False
            self._mouse_last_pos = None
            self._mouse_override = False
            self._mouse_yaw = 0.0
            self._mouse_pitch = 0.0
            self.update() 