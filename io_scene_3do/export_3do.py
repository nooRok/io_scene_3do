# coding: utf-8
from logging import getLogger
from math import degrees
from random import randrange

import bpy
import mathutils
from .icr2model.flavor import *
from .icr2model.flavor.flavor import *
from .icr2model.flavor.value.unit import to_papy_degree
from .icr2model.flavor.value.values import *
from .icr2model.model import Model

logger = getLogger(__name__)

MX_SPACE = {'world': 'matrix_world',
            'basis': 'matrix_basis',
            'local': 'matrix_local',
            'parent': 'matrix_parent_inverse'}
BSP_LENGTH = {5: 1, 6: 2, 7: 3, 8: 3, 9: 4, 10: 2}
BSP_INS_POS = {5: 0, 6: 0, 7: 2, 8: 2, 9: 2, 10: 0}


class _Cache:
    _instances = set()

    def __init__(self, f):
        self.f = f
        self.cache = {}
        self._instances.add(self)

    @classmethod
    def clear(cls):
        for i in cls._instances:  # type: _Cache
            i.cache.clear()

    def __call__(self, obj, *args, **kwargs):
        if obj in self.cache:
            return self.cache[obj]
        return self.cache.setdefault(obj, self.f(obj, *args, **kwargs))


def get_reference_keys(obj, separator):
    """

    :param bpy.types.Object obj:
    :param str separator:
    :return: (obj name, vertex group name)
    :rtype: list[str]
    """
    ref_val = obj.get('ref') or obj.get('reference')  # type: str
    if ref_val:
        obj_name, *grp_name = ref_val.split(separator, 1)
        return [obj_name, ''.join(grp_name)]
    return []


@_Cache
def get_vertex_group_map(obj, mesh):
    """

    :param bpy.types.Object obj:
    :param bpy.types.Mesh mesh:
    :return: {vertex group name: [face index in vertex group, ...], ...}
    :rtype: dict[str, list[int]]
    """
    v_grps = [set(vg_elem.group for vg_elem in v.groups)
              for v in mesh.vertices]  # [[vtx grp idx a vtx is member of (int), ...], ...]
    v_grp_map = {}
    for f in mesh.polygons:  # type: bpy.types.MeshPolygon
        fv_grps = (v_grps[v_idx] for v_idx in f.vertices)  # [[頂点に設定されたvtx grp, ...], ...]
        v_grp_idc = set.intersection(*fv_grps)  # 面の持つ全ての頂点が含まれる(=面を構成できる) vtx grp
        for v_grp_idx in v_grp_idc:
            v_grp = obj.vertex_groups[v_grp_idx]  # type: bpy.types.VertexGroup
            v_grp_map.setdefault(v_grp.name, []).append(f.index)
    return v_grp_map


def get_children(obj, key=None):
    """
    Children are sorted by its names by default

    :param bpy.types.Object obj:
    :param callable key: Function for sorting children
    :return:
    :rtype: list[bpy.types.Object]
    """
    fn = (lambda c: c.name) if key is None else key
    return sorted(obj.children, key=fn)


def get_filename(obj):
    """

    :param bpy.types.Object obj:
    :return:
        Filename from object property 'filename' or object name
        (first 8 characters) if object doesn't have property 'filename'.
    :rtype: str
    """
    return (obj.get('filename') or obj.name.split('.')[0])[:8]


def get_color_index(material_name, alt_color_index=None):
    """
    >>> get_color_index('10')
    10
    >>> get_color_index('11.001')
    11

    :param str|int material_name: -2...255
    :param str|int alt_color_index: -2...255
    :return:
    :rtype: int
    """
    name = str(material_name).split('.')[0]
    try:
        index = int(name)
        if 0 <= index <= 255:
            return index
        elif index == -2:
            return randrange(255)
        elif index == -1:
            return randrange(25, 176)
        elif alt_color_index is not None:
            return get_color_index(alt_color_index)
        raise ValueError('color index is out of range (-2...255): {}'.format(index))
    except Exception:
        raise


def is_face_object(obj):
    """

    :param bpy.types.Object obj:
    :return:
    :rtype: bool
    """
    return obj.get('F') in (1, 2) or obj.type == 'MESH'


def gen_face_vertices(mesh, face_index):
    """

    :param bpy.types.Mesh mesh:
    :param int face_index:
    """
    face = mesh.polygons[face_index]
    for idx in face.loop_indices:
        vtx_idx = mesh.loops[idx].vertex_index
        yield mesh.vertices[vtx_idx].co.copy()


def get_face_material(mesh, face_index):
    """

    :param bpy.types.Mesh mesh:
    :param int face_index:
    :return:
    :rtype: bpy.types.Material
    """
    mtls = mesh.materials
    if mtls:
        idx = mesh.polygons[face_index].material_index
        return mtls[idx]
    return None


def get_texture_slot(material):
    """

    :param bpy.types.Material material:
    :return: first enabled texture slot
    :rtype: bpy.types.MaterialTextureSlot|bpy.types.TextureSlot
    """
    slots = (s for s in material.texture_slots if s and s.use)
    return next(slots, None)


def get_face_texture_image(mesh, face_index, from_material=False):
    """

    :param bpy.types.Mesh mesh:
    :param int face_index:
    :param bool from_material:
    :return:
    :rtype: bpy.types.Image
    """
    if from_material:
        mtl = get_face_material(mesh, face_index)
        if mtl:
            t_slot = get_texture_slot(mtl)
            if t_slot and t_slot.texture.type == 'IMAGE':
                return t_slot.texture.image
    else:
        uvt = mesh.uv_textures.active
        if uvt:
            img = uvt.data[face_index].image
            assert img
            return img


def get_uv_layer(mesh, face_index, from_material=False):
    """

    :param bpy.types.Mesh mesh:
    :param int face_index:
    :param bool from_material:
    :rtype: bpy.types.MeshUVLoopLayer
    """
    if from_material:
        mtl = get_face_material(mesh, face_index)
        if mtl:
            t_slot = get_texture_slot(mtl)
            if t_slot:
                assert t_slot.texture_coords == 'UV'
                assert t_slot.texture.type == 'IMAGE'
                uvm_name = t_slot.uv_layer
                if uvm_name:
                    return mesh.uv_layers[uvm_name]
                return get_uv_layer(mesh, face_index, False)
    else:
        return mesh.uv_layers.active


def gen_face_uv_vertices(mesh, face_index, from_material=False):
    """

    :param bpy.types.Mesh mesh:
    :param int face_index:
    :param bool from_material:
    """
    face = mesh.polygons[face_index]
    uv_layer = get_uv_layer(mesh, face_index, from_material)
    if uv_layer:
        for i in face.loop_indices:  # type: int
            yield uv_layer.data[i].uv.copy()


def get_pixel_coordinate(uv_vertex, image_size, flip_uv=False):
    """
    texel coords (float) to pixel coords (int)

    :param mathutils.Vector uv_vertex:
    :param tuple[int] image_size: image width, height
    :param bool flip_uv:
    """
    if uv_vertex and image_size:
        vtx = uv_vertex.copy()
        vtx.x *= image_size[0]
        vtx.y = (-vtx.y + 1 if flip_uv else vtx.y) * image_size[1]
        return vtx
    return None


class ModelExporter:
    def __init__(self, apply_modifiers, separator,
                 texture_from, texture_flag, flip_uv, alt_color,
                 matrix, f15_rot_space, *args, **kwargs):
        """

        :param bool apply_modifiers:
        :param str separator:
        :param str texture_from: 'active'|'material'
        :param int texture_flag:
            Texture flag for F02 without texture flag value (1|2|4|...|64)
        :param bool flip_uv:
        :param int alt_color:
            Alternate color index for F01|F02 with no color index or invalid color index
            (-2:random 0-255|-1:random 32-175|0-255)
        :param mathutils.Matrix matrix: Global matrix
        :param str f15_rot_space: 'local'|'world'|'basis'|'parent'
        """
        self.model = Model()
        # mesh
        self._apply_modifiers = apply_modifiers
        self._sep = separator
        # material
        self._tex_from = texture_from
        self._tex_flag = texture_flag
        self._flip_uv = flip_uv
        self._alt_color = alt_color
        # matrix
        self._matrix = matrix.to_4x4() or mathutils.Matrix()  # 4x4
        self._f15_rot_space = f15_rot_space
        # maps
        self._files = {'mip': {}, 'pmp': {}, '3do': {}}
        self._offsets = {}  # type: dict[bpy.types.Object, int]  # {blender obj: flavor offset}
        self._meshes = {}  # type: dict[bpy.types.Object, bpy.types.Mesh]

    def _get_reference_object(self, obj, recursive=True):
        """

        :param bpy.types.Object obj:
        :return:
        :rtype: bpy.types.Object
        :raise: Exception (RecursionError)
        """
        ref_keys = get_reference_keys(obj, self._sep)
        if ref_keys:
            ref_obj = bpy.data.objects[ref_keys[0]]
            if recursive:
                return self._get_reference_object(ref_obj) or ref_obj
            return ref_obj
        return None

    def _get_matrix(self, obj, space='world'):
        """

        :param bpy.types.Object obj:
        :param space:
        :return:
        :rtype: mathutils.Matrix
        """
        mx = getattr(obj, MX_SPACE[space]).copy()  # type: mathutils.Matrix
        ref_obj = self._get_reference_object(obj)
        if ref_obj:
            ref_mx = self._get_matrix(ref_obj, space)
            mx.translation = mx.to_translation() - ref_mx.to_translation()
        return mx

    def _get_file_index(self, key, filename):
        """

        :param str key: file type (mip|pmp|3do)
        :param str filename:
        :return:
        :rtype: int
        """
        files = self._files[key]
        idx = files.setdefault(filename, len(files))
        return idx

    def _get_child_offsets(self, obj, order=None):
        """

        :param bpy.types.Object obj:
        :param list[int] order:
        :return:
        :rtype: list[int]
        """
        key = (lambda o: order.index(o['offset'])) if order else None
        return [self._offsets[c] for c in get_children(obj, key)]

    def _get_mesh(self, obj):
        """

        :param bpy.types.Object obj:
        :return: (modifiers applied) mesh
        :rtype: bpy.types.Mesh
        """
        ref_obj = self._get_reference_object(obj)
        if ref_obj:
            return self._get_mesh(ref_obj)
        if self._apply_modifiers and obj.modifiers:
            mesh = (self._meshes.get(obj) or
                    self._meshes.setdefault(
                        obj, obj.to_mesh(bpy.context.scene, True, 'RENDER')))
            logger.info('{} {}'.format(obj, mesh))
            return mesh
        return obj.data

    def _get_faces(self, obj):
        """

        :param bpy.types.Object obj:
        :return:
        :rtype: list[bpy.types.MeshPolygon]
        """
        mesh = self._get_mesh(obj)
        ref_obj = self._get_reference_object(obj)
        if ref_obj:
            logger.info('%s %s', obj, ref_obj)
            v_grp_map = get_vertex_group_map(ref_obj, mesh)
            v_grp_name = get_reference_keys(obj, self._sep)[1]
            try:
                return [mesh.polygons[i] for i in v_grp_map[v_grp_name]]
            except KeyError:
                msg = "'{}{}{}' referred from '{}'".format(
                    ref_obj.name, self._sep, v_grp_name, obj.name)
                raise KeyError(msg)
        if mesh:
            return mesh.polygons
        return []

    def _get_scaled_vertex(self, vertex, matrix=None):
        """

        :param mathutils.Vector vertex:
        :param mathutils.Matrix matrix: extra matrix (obj.matrix_world)
        :return:
        :rtype: mathutils.Vector
        """
        if matrix:
            return self._matrix * matrix * vertex
        return self._matrix * vertex

    def _gen_bsp_values(self, obj):
        """

        :param bpy.types.Object obj:
        """
        mesh = self._get_mesh(obj)
        for f in self._get_faces(obj):
            pts = [self._get_scaled_vertex(vtx, self._get_matrix(obj))
                   for vtx in gen_face_vertices(mesh, f.index)]
            pts_grps = (pts[i:i + 3] for i in range(len(pts) - 2))
            vecs = ((a, b, c) for a, b, c in pts_grps
                    if round(a.angle(b - a) - a.angle(c - a), 4))
            bsp_pts = next(vecs)  # drop invalid normal values
            bv = BspValues.from_coordinates(*bsp_pts)
            yield bv

    def _store_flavor(self, type_, values1=None, values2=None, obj=None):
        """

        :param int type_:
        :param Iterable[int] values1:
        :param Iterable[int] values2:
        :param bpy.types.Object obj:
        :return: Flavor offset
        :rtype: int
        """
        offset = len(self._flavors)
        assert self._flavors.get(offset) is None
        f = build_flavor(type_, offset, None, values1, values2)
        self._flavors[offset] = f
        if isinstance(f, RefFlavor):
            assert all(self._flavors.get(x) for x in f.children)
            for o in f.children:
                self._flavors[o].parents.append(offset)
            if isinstance(f, F16):
                self._flavors[f.next_offset].parents.append(offset)
        if obj:
            self._offsets[obj] = offset
        logger.debug('%s %s', f.offset, f.to_str())
        return offset

    def _store_vertex_flavor(self, vtx, uv=None):
        """

        :param mathutils.Vector vtx: **scaled** vertex
        :param mathutils.Vector uv:
        :return: Vertex flavor offset
        :rtype: int
        """
        offset = self._store_flavor(0, map(round, vtx), map(int, uv or []))
        self._flavors[offset].vtype = 2 if uv else 1
        return offset

    def _store_face_flavor(self, obj, face_index=0):
        """

        :param bpy.types.Object obj:
        :param int face_index:
        :return: Face flavor offset
        :rtype: int
        """
        mesh = self._get_mesh(obj)
        use_mtl = self._tex_from == 'material'
        img = get_face_texture_image(mesh, face_index, use_mtl)
        face_vtc = [self._get_scaled_vertex(vtx, self._get_matrix(obj))
                    for vtx in gen_face_vertices(mesh, face_index)]
        uv_vtc = ([get_pixel_coordinate(vtx, img.size, self._flip_uv)
                   for vtx in gen_face_uv_vertices(mesh, face_index, use_mtl)] or
                  [None for _ in range(len(face_vtc))])
        vf_os = [self._store_vertex_flavor(vtx, uv)
                 for vtx, uv in zip(face_vtc, uv_vtc)]
        mtl = get_face_material(mesh, face_index)
        color_idx = get_color_index(mtl.name if mtl else '', self._alt_color)
        values1 = [color_idx, len(vf_os) - 1]
        if img:  # F02
            tex_flag = (obj.get('values1') or
                        [obj.get('texture_flag') or
                         self._tex_flag])[0]
            values1.insert(0, tex_flag)
        offset = self._store_flavor((2 if img else 1), values1, vf_os, obj)
        return offset

    def _store_face_flavors(self, obj):
        """

        :param bpy.types.Object obj:
        :return:
        :rtype: list[int]
        """
        faces = self._get_faces(obj)
        offsets = [self._store_face_flavor(obj, f.index)
                   for f in faces]
        return offsets

    def _store_bsp_flavor(self, obj, type_):
        """

        :param bpy.types.Object obj:
        :param int type_: 5-10
        :return:
        :rtype: int
        """
        assert 5 <= type_ <= 10, type_
        values2 = self._get_child_offsets(obj, (obj.get('values2') or [])[:])
        mesh = self._get_mesh(obj)
        faces = self._get_faces(obj)
        if mesh and faces:  # face vertices define BSP normal
            values1 = next(self._gen_bsp_values(obj))  # normal of faces[1]
            if len(values2) == BSP_LENGTH[type_] - 1:  # draw itself
                f_os = self._store_face_flavors(obj)
                assert len(f_os) > 0
                o = (f_os[0] if len(f_os) == 1 else
                     self._store_flavor(11, [len(f_os)], f_os))
                values2.insert(BSP_INS_POS[type_], o)
            assert len(values2) == BSP_LENGTH[type_], obj
        else:
            values1 = obj['values1']
        offset = self._store_flavor(type_, values1, values2, obj)
        return offset

    def _build_flavor_from_data(self, obj):
        """

        :param bpy.types.Object obj:
        """
        children = []
        if is_face_object(self._get_reference_object(obj) or obj):
            os_ = self._store_face_flavors(obj)
            bsp = int(obj.get('bsp') or 0)
            children.extend([] if bsp else os_)
            if bsp:
                for o, v1 in zip(os_, self._gen_bsp_values(obj)):
                    bsp_o = self._store_flavor(5, v1, [o], obj)
                    children.append(bsp_o)
        if children or obj.children:
            children.extend(self._get_child_offsets(obj))  # order: [faces, object children]
            self._store_flavor(11, [len(children)], children, obj)
        else:  # F00(null)
            self._store_flavor(0, obj=obj)

    def _build_flavor_from_type(self, obj, type_):
        """
        Required object properties:
            * BspFlavor:
                * ['value1'] (if no specified bsp)
            * F13:
                * ['value2'] (distance1, offset1, distance2, offset2...)
                * Children must have ['offset'] (child w/o ['offset'] is dropped.)
            * F16:
                * ['value1'], ['value2']:
                    * All of children must have ['offset'].
                * No ['values1'] and no ['values2']:
                    * First child as next object, remains are used as children (values2).
            * F12:
                * ['value1']
            * F04, F15, F18:
                * ['filename']

        :param bpy.types.Object obj:
        :param int type_:
        :return:
        """
        if type_ == 0:  # F00, null
            self._store_flavor(0, obj=obj)
        elif type_ in (1, 2):  # FaceFlavor
            # faces[-1] only (other faces are dropped at optimization)
            self._store_face_flavors(obj)
        elif type_ in (5, 6, 7, 8, 9, 10):
            self._store_bsp_flavor(obj, type_)
        elif type_ == 13:  # needs 'values2'
            children = self._get_child_offsets(obj, obj['values2'][:][1::2])
            distances = obj['values2'][:][0::2]
            values2 = [v for vs in zip(distances, children) for v in vs]
            origin = self._get_mesh(obj).vertices[0].co  # type: mathutils.Vector
            vtx = self._get_scaled_vertex(origin, self._get_matrix(obj))
            origin_o = self._store_vertex_flavor(vtx)
            values1 = [origin_o]
            f13_o = self._store_flavor(type_, values1, values2, obj)
            self._flavors[origin_o].parents.append(f13_o)
        elif type_ == 16:
            v1, v2 = obj.get('values1'), obj.get('values2')
            if v1 and v2:
                order = [obj['values1'][0]]
                order.extend(v2)
                values2 = self._get_child_offsets(obj, order)
            elif not (v1 or v2):
                values2 = self._get_child_offsets(obj)
            else:  # v1 or v2
                msg = 'Pair of non-empty and empty values are not allowed'
                raise ValueError(msg, v1[:], v2[:])
            values1 = [values2.pop(0), len(values2)]
            self._store_flavor(type_, values1, values2, obj)
        elif type_ in (4, 11):  # RefFlavor
            values2 = self._get_child_offsets(obj, (obj.get('values2') or [])[:])
            if type_ == 4:
                values1 = obj.get('values1') or [0, 0]  # mip idx, color
                values1[0] = self._get_file_index('mip', get_filename(obj))
            else:  # F11
                assert type_ == 11
                values1 = [len(values2)]
                if 'F17' in obj:
                    mgr_o = self._store_flavor(11, values1, values2, obj)
                    lod_o = self._store_flavor(17, obj['F17'][:])
                    self._flavors[lod_o].parents.append(mgr_o)
                    return
            self._store_flavor(type_, values1, values2, obj)
        elif type_ in (12, 15, 17, 18):  # FixedFlavor
            if type_ == 12:
                values1 = obj['values1']
            elif type_ == 15:
                values1 = obj.get('values1') or []
                if values1:
                    if values1[-1] == 0:
                        pass
                    else:
                        idx = self._get_file_index('3do', get_filename(obj))
                        values1[-1] = ~idx
                else:  # if not values1:  # or some_flag
                    mul = (a * b for a, b in zip(obj.matrix_world.translation,
                                                 self._matrix.to_scale()))
                    tr_vec = mathutils.Vector(mul)
                    tr_vec.rotate(self._matrix)
                    loc = [int(v) for v in tr_vec]
                    eul = self._get_matrix(obj, self._f15_rot_space).to_euler()
                    eul.y *= -1  # y- front and y+ back in blender
                    deg = [degrees(e) for e in reversed(eul)]  # [float] -180...180
                    rot = [to_papy_degree(d) for d in deg]
                    idx = self._get_file_index('3do', get_filename(obj))
                    values1.extend(loc + rot + [~idx])
            elif type_ == 18:
                values1 = obj['values1'][:]
                values1[-1] = self._get_file_index('pmp', get_filename(obj))
            else:  # 17 *F17 never imported*
                raise NotImplementedError(type_, obj)
            self._store_flavor(type_, values1, obj=obj)
        else:  # (3, 14)
            raise NotImplementedError(type_, obj)

    def _build_flavor(self, obj, ignore_property=False):
        """
        Children first

        :param bpy.types.Object obj:
        :param bool ignore_property:
        :return:
        """
        logger.debug(obj)
        ref_keys = get_reference_keys(obj, self._sep)
        if ref_keys and not ref_keys[1]:  # Avoid to convert merged model
            ref_obj = self._get_reference_object(obj, recursive=False)
            assert ref_obj
            self._build_flavor(ref_obj, ignore_property)
            self._offsets[obj] = self._offsets[ref_obj]
            return
        if obj in self._offsets:
            return
        for c_obj in get_children(obj):
            self._build_flavor(c_obj, ignore_property)
        type_ = obj.get('F')
        if type_ is None or type_ == -1 or ignore_property:
            self._build_flavor_from_data(obj)
        else:
            self._build_flavor_from_type(obj, int(type_))

    def build_model(self, root_object):
        """

        :param bpy.types.Object root_object:
        :return:
        :rtype: bool
        """
        try:
            with self._flavors:
                self._build_flavor(root_object)
                root_offset = self._offsets[root_object]
                orphans = [o for o, f in self._flavors.items()
                           if not f.parents and o != root_offset]
                assert len(orphans) == 0, orphans
            for type_, files in self._files.items():
                names = [n for n, _ in sorted(files.items(), key=lambda x: x[1])]
                self.model.header.files.update({type_: names})
            self.model.sort(True)
            return True
        except Exception as err:
            logger.exception('')
            raise
        finally:
            logger.info(self._offsets)
            logger.info(self._meshes)
            for mesh in self._meshes.values():
                logger.info('%s', mesh)
                bpy.data.meshes.remove(mesh)
            self._meshes.clear()
            _Cache.clear()

    @property
    def _flavors(self):
        return self.model.body.flavors


def save(operator, context, filepath, apply_modifiers, separator,
         texture_from, default_texture_flag, flip_uv, alt_color,
         matrix, f15_rot_space, obj=None, **kwargs):
    """

    :param bpy.types.Operator operator:
    :param context:
    :param str filepath:
    :param bool apply_modifiers:
    :param str separator:
    :param str texture_from: 'active'|'material'
    :param int default_texture_flag: 1|2|4|...|64
    :param bool flip_uv:
    :param int alt_color: -2(random 0-255)|-1(random 32-175)|0-255
    :param mathutils.Matrix matrix: global matrix
    :param str f15_rot_space: 'local'|'world'
    :param bpy.types.Object obj:
    :return:
    """
    exporter = ModelExporter(apply_modifiers, separator,
                             texture_from, default_texture_flag, flip_uv, alt_color,
                             matrix, f15_rot_space, **kwargs)
    if exporter.build_model(obj or context.active_object):
        with open(filepath, 'wb') as f:
            f.write(exporter.model.to_bytes())
    return {'FINISHED'}
