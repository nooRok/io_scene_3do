# coding: utf-8
import os
from itertools import count
from math import radians

import bpy
import mathutils
from .icr2model.flavor.flavor import *
from .icr2model.flavor.value.unit import to_degree
from .icr2model.model import Model

NIL = '__nil__'


def build_id(*elements, delimiter='.'):
    """

    :param elements:
    :param str delimiter:
    :return: Strings joined by delimiter "."
    :rtype: str
    """
    return delimiter.join(map(str, elements))


def build_ref_id(ref_name, *elements, separator=':', delimiter='.'):
    """

    :param str ref_name:
    :param elements:
    :param str separator:
    :param str delimiter:
    :return: Strings joined by delimiter "."
    :rtype: str
    """
    grp_id = build_id(*elements, delimiter=delimiter)
    return separator.join([ref_name, grp_id])


def register_texture(name, width=256, height=256):
    """

    :param str name:
    :param int width: width > 0
    :param int height: height > 0
    :return:
    :rtype: bpy.types.ImageTexture
    """
    assert isinstance(name, str)
    assert width and height, [width, height]
    if name in bpy.data.textures:
        return bpy.data.textures[name]
    tex = bpy.data.textures.new(name=name, type='IMAGE')
    img = bpy.data.images.new(name=build_id(name, 'image'),
                              width=width, height=height)
    img.generated_type = 'UV_GRID'  # ; img.source = 'FILE'
    tex.image = img
    tex.use_fake_user = True
    return tex


def register_material(material_id, texture_id=None):
    """

    :param str material_id:
    :param str texture_id:
    :return:
    :rtype: bpy.types.Material
    """
    assert isinstance(material_id, str)
    if material_id in bpy.data.materials:
        return bpy.data.materials[material_id]
    mtl = bpy.data.materials.new(material_id)
    if texture_id:
        texture = bpy.data.textures[texture_id]
        ts = mtl.texture_slots.create(0)  # addだと重複して登録される
        ts.texture = texture
        ts.texture_coords = 'UV'
    return mtl


def create_object(object_id, object_data=None, parent=None):
    """

    :param str object_id:
    :param object_data:
    :param bpy.types.Object parent:
    :return:
    :rtype: bpy.types.Object
    """
    obj = bpy.data.objects.new(object_id, object_data)
    obj.parent = parent
    if object_data is None:
        obj.empty_draw_type = 'ARROWS'
    return obj


def set_properties(obj, **properties):
    """

    :param bpy.types.Object obj:
    :param properties:
    """
    for key, value in properties.items():
        obj[key] = value


def get_group(name):
    """

    :param str name:
    :return:
    :rtype: bpy.types.Group
    """
    return (bpy.data.groups.get(name) or
            bpy.data.groups.new(name))


def set_dupli_group(obj, group, type_='GROUP'):
    """

    :param bpy.types.Object obj:
    :param bpy.types.Group group:
    :param str type_:
    """
    obj.dupli_group = group
    obj.dupli_type = type_


def get_material(material_id):
    """
    Find non-zero-filled -> zero-filled -> new non-zero-filled

    :param material_id:
    :return:
    :rtype: bpy.types.Material
    """
    mtl_id = str(material_id)
    return (bpy.data.materials.get(mtl_id) or
            bpy.data.materials[mtl_id.zfill(3)])


def get_material_index(material, mesh, append=False):
    """

    :param bpy.types.Material material:
    :param bpy.types.Mesh mesh:
    :param bool append:
    :return:
    :rtype: int
    """
    mtl_idx = mesh.materials.find(material.name)
    if mtl_idx == -1 and append:
        mesh.materials.append(material)
        return get_material_index(material, mesh)
    return mtl_idx


def get_material_texture(material_id, index=0):
    """

    :param str material_id:
    :param int index:
    :return:
    :rtype: bpy.types.Texture|bpy.types.ImageTexture
    """
    mtl = get_material(material_id)
    tex_slot = mtl.texture_slots[index]
    return tex_slot.texture  # bpy.data.textures[mip_name]


def get_uv_map(mesh, name):
    """

    :param bpy.types.Mesh mesh:
    :param str name:
    :return:
    :rtype: bpy.types.MeshTexturePolyLayer
    """
    uv_map = (mesh.uv_textures.get(name) or
              mesh.uv_textures.new(name))
    return uv_map


def set_uv_coordinates(uv_loops, vertex_uvs, size):
    """

    :param list[bpy.types.MeshUVLoop] uv_loops:
    :param list[list[int]] vertex_uvs:
    :param Iterable[int] size: [width, height]
    """
    for uv_loop, vtx_uv in zip(uv_loops, vertex_uvs):
        l_uv = [uv / xy for uv, xy in zip(vtx_uv, size)]
        uv_loop.uv = l_uv


class ModelImporter:
    def __init__(self, filepath, *args, **kwargs):
        self.filepath = filepath
        self._sep = ''
        self._lod_lv = 0b000
        self._merge_uv_maps = False
        self._merged_obj_name = ''  # also used as flag "merge_faces"
        self._model = Model(filepath)
        self._objects = {}  # type: dict[int, bpy.types.Object]  # {flavor offset: blender obj}
        self._dummies = []  # type: list[bpy.types.Object]
        self._ext_objects = []  # type: list[bpy.types.Object]
        self._lod_mgrs = {}  # type: dict[int, list[int]]  # {F11 for lod (F17 parent) offset: F17 values}
        self._tex_names = {}  # type: dict[int, str]  # {F02 offset: texture name}
        # self._args = args
        # self._kwargs = kwargs

    def _get_filename(self, offset):
        """

        :param int offset:
        :return: mip|3do|pmp filename
        :rtype: str
        """
        f = self._flavors[offset]
        if isinstance(f, F15):  # 3do
            return self._model.header.files['3do'][~f.index]
        key = 'mip' if isinstance(f, F04) else 'pmp'
        return self._model.header.files[key][f.index]

    def _find_children(self, offset, fn=None):
        """

        :param offset:
        :param fn: Callable object
        """
        assert callable(fn)
        f = self._flavors[offset]
        if fn(f):
            yield f.offset
        if isinstance(f, RefFlavor):
            for c_o in f.children:
                yield from self._find_children(c_o, fn)

    def _build_flavor_id(self, offset):
        """

        :param int offset:
        :return:
        :rtype: str
        """
        f = self._flavors[offset]  # type: Flavor
        return build_id(f.offset, 'F{:02}'.format(f.type))

    def _get_flavor_properties(self, offset):
        """

        :param int offset:
        :return:
        :rtype: dict[str, int|list[int]]
        """
        f = self._flavors[offset]
        return {'F': f.type, 'offset': f.offset,
                'values1': f.values1, 'values2': f.values2}

    def _create_pydata(self, offset, start_index=0):
        """
        For bpy.types.Mesh.from_pydata

        :param int offset:
        :param int start_index:
        :return: [[co(x,y,z), ...], [vtx index, ...], [uv(u,v), ...]]
        :rtype: list[list]
        """
        f = self._flavors[offset]
        assert isinstance(f, FaceFlavor)
        c = count(start_index)
        coords = [self._flavors[o].co for o in f.children]
        vtx_idc = [next(c) for _ in range(len(f.children))]
        if isinstance(f, F02):
            uvs = [self._flavors[o].uv for o in f.children]
            return coords, vtx_idc, uvs
        return coords, vtx_idc, []

    def _get_face_material(self, offset):
        """

        :param int offset:
        :return:
        :rtype: bpy.types.Material
        """
        f = self._flavors[offset]
        assert isinstance(f, FaceFlavor)
        if isinstance(f, F02):
            mip_name = self._tex_names.get(f.offset, NIL)
            mtl_name = build_id(f.color, mip_name)
            return get_material(mtl_name)
        return get_material(f.color)

    def _set_uv_coordinates(self, mesh, material, vertex_indices, vertex_uvs, face_index=0):
        """

        :param bpy.types.Mesh mesh:
        :param bpy.types.Material material:
        :param list[int] vertex_indices:
        :param list[list[int]] vertex_uvs:
        :param int face_index:
        """
        tex_slot = material.texture_slots[0]
        img = tex_slot.texture.image
        img_name = 'texture' if self.merge_uv_maps else img.name
        uv_map = get_uv_map(mesh, build_id(img_name, 'uv'))
        uv_map.data[face_index].image = img
        tex_slot.uv_layer = uv_map.name
        uv_loops = [mesh.uv_layers[uv_map.name].data[i]
                    for i in vertex_indices]
        size = get_material_texture(material.name).image.size
        set_uv_coordinates(uv_loops, vertex_uvs, size)

    def _create_object_data(self, offset):
        """
        For object.data

        :param int offset:
        :return: Mesh or None (empty)
        :rtype: bpy.types.Mesh|None
        """
        f = self._flavors[offset]
        if isinstance(f, (BspFlavor, F13)):  # 05...10, 13
            data = bpy.data.meshes.new(self._build_flavor_id(offset))
            if isinstance(f, F13):
                ref_vtx = self._flavors[f.origin]  # type: VertexFlavor
                data.from_pydata([ref_vtx.co], [], [])
            return data
        return None

    def _read_flavor(self, offset, parent_offset=None):
        """
        Convert flavors to blender objects

        :param int offset:
        :param int parent_offset:
        """
        f = self._flavors[offset]  # type: Flavor
        f_id = self._build_flavor_id(offset)
        par_obj = self._objects.get(parent_offset)
        org_obj = self._objects.get(offset)
        if org_obj:
            obj = create_object(f_id, None, par_obj)
            props = dict(org_obj.items())
            props['ref'] = org_obj.name
            set_properties(obj, **props)
            group = get_group(org_obj.name)
            if len(group.objects) == 0:  # a new group created this time
                group.objects.link(org_obj)
            set_dupli_group(obj, group)
            self._dummies.append(obj)
        elif isinstance(f, FaceFlavor):  # 01, 02
            if self._merged_obj_name:
                obj = create_object(f_id, None, par_obj)
                ref_name = build_ref_id(
                    self._merged_obj_name, f.offset, 'F{:02}'.format(f.type),
                    separator=self._sep)
                set_properties(obj, ref=ref_name,
                               **self._get_flavor_properties(offset))
            else:
                coords, idc, uvs = self._create_pydata(offset)
                mesh = bpy.data.meshes.new(build_id(offset, 'co'))
                mesh.from_pydata(coords, [], [idc])
                obj = create_object(f_id, mesh, par_obj)
                mtl = self._get_face_material(offset)
                if isinstance(f, F02):
                    self._set_uv_coordinates(mesh, mtl, idc, uvs)
                obj.active_material = mtl
                set_properties(obj, **self._get_flavor_properties(offset))
            self._objects[f.offset] = obj
        else:
            data = self._create_object_data(f.offset)
            obj = create_object(f_id, data, par_obj)
            set_properties(obj, **self._get_flavor_properties(offset))
            self._objects[offset] = obj
            if isinstance(f, RefFlavor):  # 04...11, 13, 16
                if isinstance(f, F04):  # 04 mip
                    obj['filename'] = self._get_filename(f.offset)
                if offset in self._lod_mgrs:
                    obj['F17'] = self._lod_mgrs[offset]
                    lod_offsets = []
                    if self._lod_lv & 1 << 2:  # HI
                        lod_offsets.extend(f.children[:4])
                    if self._lod_lv & 1 << 1:  # MID
                        lod_offsets.extend(f.children[4:6])
                    if self._lod_lv & 1 << 0:  # LO
                        lod_offsets.extend(f.children[6:])
                    for lod_o in lod_offsets:
                        self._read_flavor(lod_o, offset)
                else:
                    if isinstance(f, F13):  # near -> far
                        pairs = zip(f.distances, f.children)
                        children = (c for _, c in reversed(list(pairs)))
                    else:
                        children = f.children
                    for c_o in children:
                        self._read_flavor(c_o, offset)
                    if isinstance(f, F16):
                        self._read_flavor(f.next_offset, offset)
            else:  # 00, 03, 12, 14, 15, 17, 18
                if isinstance(f, (F15, F18)):  # 3do, pmp
                    filename = self._get_filename(f.offset)
                    obj['filename'] = filename
                    if isinstance(f, F15):  # 3do
                        rads = [radians(to_degree(deg))
                                for deg in reversed(f.rotation)]
                        obj.location = f.location
                        obj.rotation_euler = rads
                        obj.show_name = True
                        self._ext_objects.append(obj)
                        group = get_group(filename)
                        set_dupli_group(obj, group)
                    else:  # pmp 18
                        raise NotImplementedError('F18')

    def _create_merged_object(self):
        """

        :return:
        :rtype: bpy.types.Object
        """
        vtc = []  # [(x, y, z), (x, y, z), ...]
        vtx_idc = []  # [[co_idx, co_idx, co_idx, ...], ...] for face
        vtx_uvs = []  # [[[u, v], [u, v], [u, v]], ...]
        offsets = sorted(self._flavors.by_types(1, 2))  # FaceFlavor offsets
        for o in offsets:  # type: int
            coords, idc, uvs = self._create_pydata(o, len(vtc))
            vtc.extend(coords)
            vtx_idc.append(idc)
            vtx_uvs.append(uvs)
        mesh = bpy.data.meshes.new(build_id(self._merged_obj_name, 'mesh'))
        mesh.from_pydata(vtc, [], vtx_idc)
        obj = bpy.data.objects.new(self._merged_obj_name, mesh)
        # Need a merged object creation before doing the following processes.
        for o, idc in zip(offsets, vtx_idc):
            name = self._build_flavor_id(o)
            obj.vertex_groups.new(name).add(idc, 1, 'ADD')
        for o, idc, uvs, poly in zip(offsets, vtx_idc, vtx_uvs, mesh.polygons):
            mtl = self._get_face_material(o)
            if isinstance(self._flavors[o], F02):
                self._set_uv_coordinates(mesh, mtl, idc, uvs, poly.index)
            poly.material_index = get_material_index(mtl, mesh, True)
        return obj

    def read_model(self):
        self._model.read()
        if self._model.is_track():  # for lod
            for f17 in self._flavors.by_types(17).values():  # type: F17
                self._lod_mgrs[f17.parents[0]] = list(f17.values1)
        tex_ref = {}  # {F02 offset: F04 offset} -> ...
        for f04_o in sorted(self._flavors.by_types(4)):  # type: int
            f02_os = self._find_children(f04_o, lambda f: isinstance(f, F02))
            for f02_o in f02_os:
                tex_ref.setdefault(f02_o, f04_o)
        self._tex_names.update({k: self._get_filename(v)  # ... -> {F02 offset: texture name}
                                for k, v in tex_ref.items()})

    def register_textures(self, w=256, h=256):
        """

        :param int w:
        :param int h:
        """
        names = self._model.header.files['mip'][:]
        if set(self._flavors.by_types(2)) - set(self._tex_names):  # __NIL__
            names.append(NIL)
        for name in names:
            register_texture(name, w, h)

    def register_materials(self):
        for _, f in sorted(self._flavors.by_types(1, 2).items()):  # type: int, FaceFlavor
            if isinstance(f, F02):
                mip_name = self._tex_names.get(f.offset, NIL)
                mtl_name = build_id(f.color, mip_name)
                register_material(mtl_name, mip_name)
            else:  # 01
                assert isinstance(f, F01)
                register_material(str(f.color))

    def read_flavors(self, merge_uv_maps, merged_obj_name='', separator=':', lod_level=0b000):
        """

        :param bool merge_uv_maps:
        :param str merged_obj_name:
        :param str separator:
        :param int lod_level: bit field
        """
        self._merge_uv_maps = merge_uv_maps
        self._merged_obj_name = merged_obj_name
        self._sep = separator
        self._lod_lv = lod_level
        self._read_flavor(self._model.header.root_offset)

    def link_objects(self, scale=0, context=None):
        """
        Link objects to current scene

        :param int scale: 0=auto (object=10000, track=1000000)
        :param bpy.types.Context context:
        """
        # scale
        scale = (scale or (1000000 if self._model.is_track() else 10000))
        scale_vec = mathutils.Vector([1, 1, 1]) / scale
        root_obj = self._objects[self._model.header.root_offset]
        root_obj.scale = scale_vec
        root_obj['scale'] = scale
        # for ex_obj in self._external_objects:  # F15 objects
        #     ex_obj.scale = mathutils.Vector.Fill(3, scale)
        # link
        grp = get_group(os.path.basename(self.filepath))
        for obj in (list(self._objects.values()) + self._dummies):
            (context or bpy.context).scene.objects.link(obj)
            grp.objects.link(obj)
        if self._merged_obj_name:
            mrg_obj = self._create_merged_object()
            mrg_obj['ref_source'] = True
            mrg_obj.scale = scale_vec
            (context or bpy.context).scene.objects.link(mrg_obj)

    @property
    def merge_uv_maps(self):
        return self._merge_uv_maps or len(set(self._tex_names.values())) > 8

    @property
    def _flavors(self):
        return self._model.body.flavors


def load_3do(operator, context, filepath, lod_level, scale, tex_w, tex_h,
             merge_faces, merge_uv_maps, merged_obj_name, separator, **_):
    """

    :param bpy.types.Operator operator:
    :param bpy.types.Context context:
    :param str filepath:
    :param int lod_level:
    :param int scale:
    :param int tex_w:
    :param int tex_h:
    :param bool merge_faces:
    :param bool merge_uv_maps:
    :param str merged_obj_name:
    :param str separator:
    :return:
    """
    importer = ModelImporter(filepath)
    importer.read_model()
    importer.register_textures(tex_h, tex_w)
    importer.register_materials()
    importer.read_flavors(merge_uv_maps,
                          merged_obj_name if merge_faces else '',
                          separator,
                          lod_level)
    importer.link_objects(scale, context)
    operator.report({'INFO'}, 'Model imported ({})'.format(filepath))
    return {'FINISHED'}
