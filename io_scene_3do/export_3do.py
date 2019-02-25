# coding: utf-8
from itertools import chain
from logging import getLogger
from math import degrees
from pathlib import Path
from random import randrange

import bpy
import mathutils
from .icr2model import __version__ as icr2model_ver

print(icr2model_ver)
if tuple(map(int, icr2model_ver.split('.'))) < (0, 2, 0):
    from .icr2model.model import Model
    from .icr2model.model.flavor import build_flavor
    from .icr2model.model.flavor.flavor import *
    from .icr2model.model.flavor.values import BspValues
    from .icr2model.model.flavor.values.unit import to_papy_degree
else:
    from .icr2model import Model
    from .icr2model.flavor import *
    from .icr2model.flavor.values import BspValues
    from .icr2model.flavor.values.unit import to_papy_degree

logger = getLogger(__name__)


def is_face_object(obj):
    """

    :param bpy.types.Object obj:
    :return:
    :rtype: bool
    """
    flag = obj.get('F')
    return flag in (1, 2) or obj.type == 'MESH'


def get_face_vertices(mesh, face_index):
    """

    :param bpy.types.Mesh mesh:
    :param int face_index:
    :return:
    :rtype: __generator[mathutils.Vector]
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
    get_face_texture() + get_face_texture_image()

    :param mesh:
    :param face_index:
    :param from_material:
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
            assert img, 'Image for face must be assigned.'
            return img


def get_uv_layer(mesh, face_index, from_material=False):
    """

    :param mesh:
    :param face_index:
    :param from_material:
    :rtype: bpy.types.MeshUVLoopLayer
    """
    if from_material:
        mtl = get_face_material(mesh, face_index)
        if mtl:
            t_slot = get_texture_slot(mtl)
            if t_slot:
                assert t_slot.texture_coords == 'UV', \
                    ('Texture mapping coordinates must be set to UV. '
                     '({})'.format(t_slot.texture_coords))
                assert t_slot.texture.type == 'IMAGE', \
                    ('Texture type must be set to IMAGE. '
                     '({})'.format(t_slot.texture.type))
                uvm_name = t_slot.uv_layer
                if uvm_name:
                    return mesh.uv_layers[uvm_name]
                return get_uv_layer(mesh, face_index, False)
    else:
        return mesh.uv_layers.active


def get_face_uv_vertices(mesh, face_index, from_material=False):
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


def get_pixel_coordinate(vertex, size, flip_uv=False):
    """
    texel coords (float) to pixel coords (int)

    :param tuple[int] size: image width, height
    :param mathutils.Vector vertex:
    :param bool flip_uv:
    """
    if vertex and size:
        vtx = vertex.copy()
        vtx.x *= size[0]
        vtx.y = (-vtx.y + 1 if flip_uv else vtx.y) * size[1]
        return vtx
    return None


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
        if index == -2:
            return randrange(255)
        elif index == -1:
            return randrange(25, 176)
        elif 0 <= index <= 255:
            return index
        raise ValueError('color index out of range (-2...255): {}'.format(index))
    except Exception:
        if alt_color_index is not None and (-2 <= alt_color_index <= 255):
            return get_color_index(alt_color_index)
        raise


def get_object_instance(obj):
    """

    :param bpy.types.Object obj:
    :return:
    :rtype: bpy.types.Object
    """
    if obj.dupli_type == 'GROUP' and obj.dupli_group:
        if obj.dupli_group.objects and obj.dupli_group.objects[0].parent:
            return obj.dupli_group.objects[0]
    return obj


def get_children(obj, key=None):
    """
    default sort order: object.name

    :param bpy.types.Object obj:
    :param key: function for sorting
    :return:
    :rtype: list[bpy.types.Object]
    """
    children = [get_object_instance(c) for c in obj.children]
    if key:
        return sorted(children, key=key)
    return sorted(children, key=lambda c: c.name)


def get_filename(obj):
    """

    :param bpy.types.Object obj:
    :return:
        Filename from object property 'filename' or object name
        (first 8 characters) if object doesn't have property 'filename'.
    :rtype: str
    """
    return (obj.get('filename') or obj.name.split('.')[0])[:8]


def get_reference_source_keys(obj):
    """

    :param bpy.types.Object obj:
    :return: pair of source obj name and vertex group name
    :rtype: tuple[str, str]
    """
    ref_val = obj.get('ref') or obj.get('reference')  # type: str
    if ref_val:
        values = ref_val.split('.')
        src_obj_name = values.pop(0)
        src_grp_name = '.'.join(values) or None
        return src_obj_name, src_grp_name


def get_reference_source_object(obj):
    """

    :param bpy.types.Object obj:
    :return:
    :rtype: bpy.types.Object
    """
    src_keys = get_reference_source_keys(obj)
    if src_keys:
        return bpy.data.objects[src_keys[0]]


def get_matrix(obj):
    """

    :param bpy.types.Object obj:
    :return:
    :rtype: mathutils.Matrix
    """
    src_obj = get_reference_source_object(obj)
    if src_obj:
        tr1 = src_obj.matrix_world.to_translation()
        tr2 = obj.matrix_world.to_translation()
        mx = obj.matrix_world.copy()
        mx.translation = tr1 - tr2
        return mx
    return obj.matrix_world


def get_grouped_faces_map(mesh):
    """

    :param bpy.types.Mesh mesh:
    :return: {vertex group index: [face assigned by vertex group, ...]}
    :rtype: dict[int, list[int]]
    """
    v_grp_map = {}  # type: dict[int, list[int]]  #
    for f in mesh.polygons:  # type: bpy.types.MeshPolygon
        vtc = (mesh.vertices[vi] for vi in f.vertices)  # type: list[bpy.types.MeshVertex]  #
        vgs = (v.groups for v in vtc)
        v_grp_idc = [set(ge.group for ge in vg) for vg in vgs]  # [set(indices vtx assigned), ...]
        f_grp_idc = v_grp_idc.pop().intersection(*v_grp_idc)  # すべての頂点(=面)が属するvtxgrp
        for f_grp_idx in f_grp_idc:  # type: int
            v_grp_val = v_grp_map.setdefault(f_grp_idx, [])
            v_grp_val.append(f.index)
    return v_grp_map


bsp_length = {5: 1, 6: 2, 7: 3, 8: 3, 9: 4, 10: 2}
bsp_pos = {5: 0, 6: 0, 7: 2, 8: 2, 9: 2, 10: 0}


class ModelExporter:
    def __init__(self, root_object, apply_modifiers, matrix,
                 texture_from, default_texture_flag, flip_uv, alt_color,
                 f15_rot_space, *args, **kwargs):
        """

        :param bpy.types.Object root_object:
        :param bool apply_modifiers:
        :param mathutils.Matrix matrix: global matrix
        :param str texture_from: 'active'|'material'
        :param int default_texture_flag: 1|2|4|...|64
        :param bool flip_uv:
        :param int alt_color: -2(random 0-255)|-1(random 32-175)|0-255
        :param str f15_rot_space: 'local'|'world'
        """
        self._model = Model()
        self._root_object = root_object
        self._tex_from = texture_from
        self._matrix = matrix.to_4x4() or mathutils.Matrix()  # 4x4
        self._flip_uv = flip_uv
        self._default_texture_flag = default_texture_flag
        self._alt_color = alt_color
        self._apply_modifiers = apply_modifiers
        self._f15_rot_space = f15_rot_space
        self._args = args
        self._kwargs = kwargs
        # maps
        self._files = {'mip': {}, 'pmp': {}, '3do': {}}
        self._obj_map = {}  # type: dict[bpy.types.Object, int]  #
        """{objID: flavor offset}"""
        self._bsp_ref_map = {}  # type: dict[int, list[int]]  #
        """{bsp flavor offset: vtx flavor offsets for bsp normal}"""
        self._meshes = {}  # type: dict[bpy.types.Object, bpy.types.Mesh]  #
        self._ref_grps = {}  # type: dict[bpy.types.Object, dict[int, list[int]]]  #

    def _get_file_index(self, key, filename):
        """

        :param str key: file type (mip|pmp|3do)
        :param str filename:
        :return:
        :rtype: int
        """
        files = self._files[key]  # type: dict[str, int]  #
        idx = files.setdefault(filename, len(files))
        return idx

    def _get_child_offsets(self, obj, order=None):
        """

        :param bpy.types.Object obj:
        :param list[int] order:
        :return:
        """
        if order:
            children = get_children(
                obj, lambda o: order.index(o['offset']))
        else:
            children = [c for c in get_children(obj)]
        return [self._obj_map[c] for c in children]

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

    def _get_mesh(self, obj):
        """

        :param bpy.types.Object obj:
        :return: (modifiers applied) mesh
        :rtype: bpy.types.Mesh
        """
        src_obj = get_reference_source_object(obj)
        if src_obj:
            src_mesh = self._get_mesh(src_obj)
            if src_obj not in self._ref_grps:  # lazy
                faces_map = get_grouped_faces_map(src_mesh)
                self._ref_grps[src_obj] = faces_map
            return src_mesh
        if self._apply_modifiers and obj.modifiers:
            mesh = (self._meshes.get(obj) or
                    self._meshes.setdefault(
                        obj, obj.to_mesh(bpy.context.scene, True, 'RENDER')))
            return mesh
        return obj.data

    def _get_faces(self, obj):
        """

        :param bpy.types.Object obj:
        :return:
        :rtype: list[bpy.types.MeshPolygon]
        """
        mesh = self._get_mesh(obj)
        src_obj = get_reference_source_object(obj)
        if src_obj:
            _, src_grp_name = (obj.get('ref') or obj.get('reference')).split('.', 1)
            try:
                v_grp_idx = src_obj.vertex_groups[src_grp_name].index  # type: int
                return [mesh.polygons[i] for i in
                        self._ref_grps[src_obj][v_grp_idx]]
            except KeyError as err:
                raise KeyError('{} ({})'.format(err, obj))
            except:
                raise
        if mesh:
            return mesh.polygons
        return []

    def _get_bsp_values(self, obj):
        """

        :param bpy.types.Object obj:
        :rtype: types.GeneratorType[tuple[int]]
        """
        mesh = self._get_mesh(obj)
        for f in self._get_faces(obj):
            pts = [self._get_scaled_vertex(vtx, get_matrix(obj))
                   for vtx in get_face_vertices(mesh, f.index)]
            pts_grps = (pts[i:i + 3] for i in range(len(pts) - 2))
            vecs = ((a, b, c) for a, b, c in pts_grps
                    if round(a.angle(b - a) - a.angle(c - a), 4))
            bsp_pts = next(vecs)  # drop invalid normal values
            bv = BspValues.from_coordinates(*bsp_pts)
            yield bv.i

    def _store_flavor(self, type_, values1=None, values2=None, obj=None):
        """

        :param int type_:
        :param Iterable[int]|None values1:
        :param Iterable[int]|None values2:
        :param bpy.types.Object obj:
        :return: stored flavor offset
        :rtype: int
        """
        offset = len(self._flavors)
        assert self._flavors.get(offset) is None
        flavor = build_flavor(offset, type_, values1=values1, values2=values2)
        self._flavors[offset] = flavor
        if isinstance(flavor, RefFlavor):
            assert all(self._flavors.get(x) for x in flavor.children)
            for o in flavor.children:
                self._flavors[o].parents.append(offset)
            if isinstance(flavor, F16):
                self._flavors[flavor.next_offset].parents.append(offset)
        if obj:
            self._obj_map[obj] = offset
        logger.debug(flavor)
        return offset

    def _store_vertex_flavor(self, vtx, uv=None):
        """

        :param mathutils.Vector vtx: **scaled** vertex
        :param mathutils.Vector uv:
        :return: vertex flavor offset
        :rtype: int
        """
        offset = self._store_flavor(0, map(round, vtx), map(int, uv or []))
        self._flavors[offset].set_vtype(2 if uv else 1)
        return offset

    def _store_face_flavor(self, obj, face_index=0):
        """

        :param bpy.types.Object obj:
        :param int face_index:
        :return:
        """
        mesh = self._get_mesh(obj)  # mesh or self._get_mesh(obj)
        use_mtl = self._tex_from == 'material'
        face_vtc = [self._get_scaled_vertex(vtx, get_matrix(obj))
                    for vtx in get_face_vertices(mesh, face_index)]
        img = get_face_texture_image(mesh, face_index, use_mtl)
        tx_vtc = get_face_uv_vertices(mesh, face_index, use_mtl)
        uv_vtc = ([get_pixel_coordinate(vtx, img.size, self._flip_uv) for vtx in tx_vtc] or
                  [None for _ in range(len(face_vtc))])
        vtx_idc = [self._store_vertex_flavor(vtx, uv)
                   for vtx, uv in zip(face_vtc, uv_vtc)]
        mtl = get_face_material(mesh, face_index)
        color_idx = get_color_index(mtl.name if mtl else '',
                                    self._alt_color)
        values1 = [color_idx, len(vtx_idc) - 1]
        if img:  # F02
            values1.insert(0, (obj.get('values1') or
                               [obj.get('texture_flag') or
                                self._default_texture_flag])[0])
        offset = self._store_flavor((2 if img else 1), values1, vtx_idc, obj)
        return offset

    def _store_face_flavors(self, obj):
        """

        :param bpy.types.Object obj:
        :return: flavor offsets
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
            values1 = next(self._get_bsp_values(obj))  # normal of faces[1]
            if len(values2) == bsp_length[type_] - 1:  # draw
                f_os = self._store_face_flavors(obj)
                o = (f_os[0] if len(f_os) == 1 else
                     self._store_flavor(11, [len(f_os)], f_os))
                values2.insert(bsp_pos[type_], o)
                if '_bsp_ref_map':  # pending deprecation
                    f = self._flavors[f_os[0]]  # type: FaceFlavor
                    bsp_pts = f.children[:3]
            else:
                assert len(values2) == bsp_length[type_], obj
                if '_bsp_ref_map':  # pending deprecation
                    bsp_pts = [self._get_scaled_vertex(vtx, get_matrix(obj))
                               for vtx in get_face_vertices(mesh, 0)][:3]
                    for bsp_pt in bsp_pts:
                        self._store_vertex_flavor(bsp_pt)
            if '_bsp_ref_map':  # pending deprecation
                offset = self._store_flavor(type_, values1, values2, obj)
                self._bsp_ref_map[offset] = bsp_pts
                return offset
        else:
            values1 = obj['values1']
        offset = self._store_flavor(type_, values1, values2, obj)
        return offset

    def _build_flavor_from_type(self, obj, type_):
        """
        required object properties:
            * BspFlavor:
                * ['value1'] (if no specified bsp)
            * F13:
                * ['value2'] (distance1, offset1, distance2, offset2...)
                * children must have ['offset'] (child that has no ['offset'] is
                dropped.)
            * F16:
                * ['value1'], ['value2']
                    * all of children must have ['offset'].
                * no ['values1'] and ['values2']
                    * first child as next object, remains as children.
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
            values2 = chain.from_iterable(zip(distances, children))
            origin = self._get_mesh(obj).vertices[0].co  # type: mathutils.Vector
            vtx = self._get_scaled_vertex(origin, get_matrix(obj))
            values1 = [self._store_vertex_flavor(vtx)]
            self._store_flavor(type_, values1, values2, obj)
        elif type_ == 16:
            v1, v2 = obj.get('values1'), obj.get('values2')
            if (v1 or v2) and not (v1 and v2):
                msg = ('obj must have both of "values1" and "values2" '
                       'if specify "values1" or "values2". '
                       '(values1: {}, values2: {})'.format(v1[:], v2[:]))
                raise ValueError(msg)
            if v1 and v2:
                order = [obj['values1'][0]]
                order.extend(v2)
                values2 = self._get_child_offsets(obj, order)
            else:
                values2 = self._get_child_offsets(obj)
            values1 = [values2.pop(0), len(values2)]
            self._store_flavor(type_, values1, values2, obj)
        elif type_ in (4, 11):  # RefFlavor
            values2 = self._get_child_offsets(obj, (obj.get('values2') or [])[:])
            if type_ == 4:
                values1 = obj.get('values1') or [0, 0]  # mip idx, color
                values1[0] = self._get_file_index('mip', get_filename(obj))
            elif type_ == 11:
                values1 = [len(values2)]
                if 'F17' in obj:
                    mgr_offset = self._store_flavor(11, values1, values2, obj)
                    lod_offset = self._store_flavor(17, obj['F17'][:])
                    self._flavors[lod_offset].parents.append(mgr_offset)
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
                    loc = map(int, tr_vec)
                    eul = (obj.matrix_basis if self._f15_rot_space == 'basis' else
                           obj.matrix_parent_inverse if self._f15_rot_space == 'parent' else
                           obj.matrix_local if self._f15_rot_space == 'local' else
                           obj.matrix_world).to_euler()  # getattr
                    eul.y *= -1  # y- front and y+ back in blender
                    deg = [degrees(e) for e in reversed(eul)]  # [float] -180...180
                    rot = [to_papy_degree(d) for d in deg]
                    idx = self._get_file_index('3do', get_filename(obj))
                    values1.extend(chain(loc, rot, [~idx]))
            elif type_ == 18:
                values1 = list(obj['values1'][:])
                values1[-1] = self._get_file_index('pmp', get_filename(obj))
            else:  # 17 *F17 never imported*
                raise NotImplementedError(type_, obj)
            self._store_flavor(type_, values1, obj=obj)
        else:  # (3, 14)
            raise NotImplementedError(type_, obj)

    def _build_flavor_from_data(self, obj):
        """

        :param bpy.types.Object obj:
        """
        if is_face_object(get_reference_source_object(obj) or obj):
            offsets = self._store_face_flavors(obj)
            children = [] if obj.get('bsp') else offsets
            if obj.get('bsp'):
                for o, v1 in zip(offsets, self._get_bsp_values(obj)):
                    bsp_offset = self._store_flavor(5, v1, [o], obj)
                    children.append(bsp_offset)
                    if '_bsp_ref_map':  # pending deprecation
                        f = self._flavors[o]  # type: FaceFlavor
                        self._bsp_ref_map[bsp_offset] = f.children[:3]
            if len(children) > 1 or obj.children:
                children.extend(self._get_child_offsets(obj))  # order: [faces, object children]
                self._store_flavor(11, [len(children)], children, obj)
        elif len(obj.children):
            children = self._get_child_offsets(obj)
            self._store_flavor(11, [len(children)], children, obj)
        else:  # F00(null)
            self._store_flavor(0, obj=obj)

    def _build_flavors(self, obj, ignore_property=False):
        """
        children first

        :param bpy.types.Object obj:
        :param bool ignore_property:
        :return:
        """
        src_keys = get_reference_source_keys(obj)
        if src_keys and src_keys[1] is None:
            src_obj = get_reference_source_object(obj)
            self._build_flavors(src_obj, ignore_property)
            self._obj_map[obj] = self._obj_map[src_obj]
            return
        if obj in self._obj_map:
            return
        for c_obj in get_children(obj):
            self._build_flavors(c_obj, ignore_property)
        flavor_type = obj.get('F')
        if flavor_type is None or flavor_type == -1 or ignore_property:
            self._build_flavor_from_data(obj)
        else:
            self._build_flavor_from_type(obj, int(flavor_type))

    def build_model(self):
        try:
            self._build_flavors(self._root_object)
            root_offset = self._obj_map[self._root_object]
            orphans = [o for o, f in self._flavors.items()
                       if not f.parents and o != root_offset]
            for o in orphans:
                self._flavors.pop(o)
            for type_, files in self._files.items():
                names = [n for n, _ in sorted(files.items(), key=lambda x: x[1])]
                self._model.header.set_files(**{type_: names})
        finally:
            for mesh in self._meshes.values():
                bpy.data.meshes.remove(mesh)
            self._meshes.clear()

    def get_3do(self):
        """

        :return: 3do model data
        :rtype: bytes
        """
        return self._model.optimized().get_bytes()

    @property
    def _flavors(self):
        return self._model.body.flavors


def save(operator, context, filepath, apply_modifiers=False, matrix=None,
         texture_from='material', default_texture_flag=8, flip_uv=False, alt_color=0,
         f15_rot_space='local', obj=None, **kwargs):
    """

    :param bpy.types.Operator operator:
    :param context:
    :param str filepath:
    :param bool apply_modifiers:
    :param mathutils.Matrix matrix: global matrix
    :param str texture_from: 'active'|'material'
    :param int default_texture_flag: 1|2|4|...|64
    :param bool flip_uv:
    :param int alt_color: -2(random 0-255)|-1(random 32-175)|0-255
    :param str f15_rot_space: 'local'|'world'
    :param bpy.types.Object obj:
    :param kwargs:
    :return:
    """
    exporter = ModelExporter(obj or context.active_object,
                             apply_modifiers,
                             matrix,
                             texture_from,
                             default_texture_flag,
                             flip_uv,
                             alt_color,
                             f15_rot_space,
                             **kwargs)
    exporter.build_model()
    path = Path(filepath)
    data = exporter.get_3do()
    with path.open('wb') as f:
        f.write(data)
    return {'FINISHED'}
