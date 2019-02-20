# coding: utf-8
import os
from itertools import chain, count
from math import radians

import bpy
import mathutils
from .icr2model import __version__ as icr2model_ver

print(icr2model_ver)
if tuple(map(int, icr2model_ver.split('.'))) < (0, 2, 0):
    from .icr2model.model import Model
    from .icr2model.model.flavor.flavor import *
    from .icr2model.model.flavor.values.unit import to_degree
else:
    from .icr2model import Model
    from .icr2model.flavor import *
    from .icr2model.flavor.values.unit import to_degree

NIL = '__nil__'


def build_id(*elements, delimiter='.', flavor=None):
    """

    :param elements:
    :param str delimiter:
    :param Flavor flavor:
    :return: joined by delimiter "."
    :rtype: str
    """
    if isinstance(flavor, Flavor):
        elems = (flavor.offset, 'F{:02}'.format(flavor.type))
        return build_id(*elems, delimiter=delimiter)
    return delimiter.join(map(str, elements))


def set_properties(obj, **properties):
    """
    Set properties to object.

    :param bpy.types.Object obj:
    :param properties:
    """
    for key, value in properties.items():
        obj[key] = value


def store_texture(mip_name, width=256, height=256):
    """
    Create empty texture (and image) slot.

    * bpy.data.textures: mip_name
    * bpy.data.images: mip_name.image

    :param str mip_name:
    :param int width: width > 0
    :param int height: height > 0
    :return:
    :rtype: bpy.types.ImageTexture
    """
    if mip_name in bpy.data.textures:
        return bpy.data.textures[mip_name]
    assert width and height, ('width and height must be larger than 0. '
                              '({}, {})'.format(width, height))
    tex = bpy.data.textures.new(name=mip_name, type='IMAGE')
    img = bpy.data.images.new(name=build_id(mip_name, 'image'),
                              width=width, height=height)
    img.generated_type = 'UV_GRID'  # ; img.source = 'FILE'
    tex.image = img
    tex.use_fake_user = True
    return tex


def get_texture(mip_name):
    """

    :param str mip_name:
    :return:
    :rtype: bpy.types.ImageTexture
    """
    return (bpy.data.textures.get(mip_name) or
            store_texture(mip_name))


def register_material(material_id, texture_id=None):
    """

    :param material_id:
    :param str texture_id:
    """
    mtl_id = str(material_id)
    if mtl_id in bpy.data.materials:
        return bpy.data.materials[mtl_id]
    mtl = bpy.data.materials.new(mtl_id)
    if texture_id:
        texture = bpy.data.textures[texture_id]
        ts = mtl.texture_slots.create(0)  # addだと重複して登録される
        ts.texture = texture
        ts.texture_coords = 'UV'
    return mtl


def get_material(material_id):
    """
    finding process:
        non-zero-filled -> zero-filled -> new non-zero-filled

    :param material_id:
    :return:
    :rtype: bpy.types.Material
    """
    mtl_id = str(material_id)
    return (bpy.data.materials.get(mtl_id) or
            bpy.data.materials[mtl_id.zfill(3)])


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


def get_uv_layer(mesh, name):
    """

    :param bpy.types.Mesh mesh:
    :param str name:
    :return:
    :rtype: bpy.types.MeshUVLoopLayer
    """
    return mesh.uv_layers[name]


def set_uv_map_image(uv_map, image):
    """

    :param bpy.types.MeshTexturePolyLayer uv_map:
    :param bpy.types.Image image:
    """
    for uvd in uv_map.data:  # Multitexture
        uvd.image = image


def set_uv_coordinates(uv_loops, vertex_uvs, size):
    """

    :param list[bpy.types.MeshUVLoop] uv_loops:
    :param list[list[int]] vertex_uvs:
    :param Iterable[int] size: [width, height]
    """
    for uv_loop, vtx_uv in zip(uv_loops, vertex_uvs):
        l_uv = [uv / xy for uv, xy in zip(vtx_uv, size)]
        uv_loop.uv = l_uv


def get_group(group_name):
    """

    :param str group_name:
    :return:
    :rtype: bpy.types.Group
    """
    return (bpy.data.groups.get(group_name) or
            bpy.data.groups.new(group_name))


def set_dupli_group(obj, group):
    """

    :param bpy.types.Object obj:
    :param bpy.types.Group group:
    """
    obj.dupli_group = group
    obj.dupli_type = 'GROUP'


def get_flavor_properties(flavor):
    """

    :param Flavor flavor:
    :return: properties from flavor
    :rtype: dict[str, int|list[int]]
    """
    properties = {'values1': flavor.values1.i,
                  'values2': flavor.values2.i,
                  'F': flavor.type,
                  'offset': flavor.offset}
    return properties


def create_object(flavor, object_data=None, parent=None):
    """
    Create object from Flavor

    :param Flavor flavor:
    :param bpy.types.Mesh object_data: Mesh or None (empty)
    :param bpy.types.Object parent:
    :return:
    :rtype: bpy.types.Object
    """
    obj = bpy.data.objects.new(build_id(flavor=flavor), object_data)
    obj.parent = parent
    if object_data is None:
        obj.empty_draw_type = 'ARROWS'
    return obj


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


class ModelImporter:
    def __init__(self, filepath):
        """

        :param str filepath:
        """
        self._filepath = filepath
        self._model = Model(filepath)  # type: Model
        self._objects = {}  # type: dict[int, bpy.types.Object]  #
        self._dummies = []  # type: list[bpy.types.Object]  #
        self._external_objects = []  # type: list[bpy.types.Object]  # F15
        self._lod_mgr_map = {}  # type: dict[int, list[int]]  #
        self._lod_level = 0b111
        self._texture_names = {}  # type: dict[int, str]  # {F02.offset: texture name}
        self._merge_faces = False
        self._merge_uv_maps = False
        self.merged_obj_name = os.path.basename(filepath).replace('.', '_')

    def _create_pydata(self, offset, start_index=0):
        """
        For bpy.types.Mesh.from_pydata

        :param int offset:
        :param int start_index:
        :return: [[co(x,y,z), ...], [vtx index, ...], [uv(u,v), ...]]
        :rtype: (list[[int]], list[int], list[list[int]])
        """
        f = self._flavors[offset]  # type: FaceFlavor
        c = count(start_index)
        coords = [self._flavors[c].co for c in f.children]
        vtx_idc = [next(c) for _ in range(len(f.children))]
        if isinstance(f, F02):
            uvs = [self._flavors[c].uv for c in f.children]
            return coords, vtx_idc, uvs
        return coords, vtx_idc, []

    def _create_object_data(self, offset):
        """
        For object.data

        :param int offset:
        :return: Mesh or None (empty)
        :rtype: bpy.types.Mesh|None
        """
        f = self._flavors[offset]
        if isinstance(f, (BspFlavor, F13)):  # 05...10, 13
            data_name = build_id(flavor=f)
            data = bpy.data.meshes.new(data_name)
            if isinstance(f, F13):
                ref_vtx = self._flavors[f.origin]  # type: VertexFlavor
                data.from_pydata([ref_vtx.co], [], [])
            return data
        return None

    def _get_filename(self, offset):
        """

        :param int offset:
        :return: mip|3do|pmp filename
        :rtype: str
        """
        f = self._flavors.get(offset)
        files = self._model.header.files
        if isinstance(f, F04):  # 04 mip
            return files['mip'][f.mip_index]
        elif isinstance(f, F15):  # 3do
            return files['3do'][~f.object_index]
        elif isinstance(f, F18):  # pmp
            return files['pmp'][f.pmp_index]

    def _get_face_material(self, offset):
        f = self._flavors[offset]  # type: FaceFlavor
        if isinstance(f, F02):
            mip_name = self._texture_names.get(f.offset, NIL)
            mtl_name = build_id(f.color, mip_name)
            return get_material(mtl_name)
        return get_material(f.color)

    def _get_uv_map(self, mtl, mesh, index=0):
        tex_slot = mtl.texture_slots[index]
        img = tex_slot.texture.image
        img_name = 'texture' if self._merge_uv_maps else img.name
        uv_map = get_uv_map(mesh, build_id(img_name, 'uv'))
        # set_uv_map_image(uv_map, img)  # Multitexture
        tex_slot.uv_layer = uv_map.name  # GLSL
        return uv_map

    def _set_uv_coordinates(self, mesh, material, vertex_indices, vertex_uvs):
        """

        :param bpy.types.Mesh mesh:
        :param bpy.types.Material material:
        :param list[int] vertex_indices:
        :param list[list[int]] vertex_uvs:
        """
        uv_map = self._get_uv_map(material, mesh)
        uv_loops = [mesh.uv_layers[uv_map.name].data[i] for i in vertex_indices]
        size = get_material_texture(material.name).image.size
        set_uv_coordinates(uv_loops, vertex_uvs, size)

    def _read_flavor(self, offset, parent_offset=None):
        """
        Convert flavors to blender objects

        :param int offset:
        :param int parent_offset:
        """
        f = self._flavors[offset]
        par_obj = self._objects.get(parent_offset)
        org_obj = self._objects.get(offset)  # origin obj
        if org_obj:
            obj = create_object(f, None, par_obj)
            set_properties(obj, dummy=True, **dict(org_obj.items()))
            group = get_group(org_obj.name)
            if len(group.objects) == 0:  # created new group
                group.objects.link(org_obj)
            set_dupli_group(obj, group)
            self._dummies.append(obj)
        elif isinstance(f, FaceFlavor):  # 01, 02
            if self._merge_faces:
                obj = create_object(f, None, par_obj)
                ref_name = build_id(self.merged_obj_name,
                                    f.offset,
                                    'F{:02}'.format(f.type))
                set_properties(obj, ref=ref_name, **get_flavor_properties(f))
            else:
                coords, idc, uvs = self._create_pydata(f.offset)
                mesh = bpy.data.meshes.new(build_id(f.offset, 'co'))
                mesh.from_pydata(coords, [], [idc])
                obj = create_object(f, mesh, par_obj)
                mtl = self._get_face_material(f.offset)
                if isinstance(f, F02):
                    self._set_uv_coordinates(mesh, mtl, idc, uvs)
                obj.active_material = mtl
                set_properties(obj, **get_flavor_properties(f))
            self._objects[f.offset] = obj
        else:
            data = self._create_object_data(f.offset)
            obj = create_object(f, data, par_obj)
            set_properties(obj, **get_flavor_properties(f))
            self._objects[f.offset] = obj
            if isinstance(f, RefFlavor):  # 04...11, 13, 16
                if isinstance(f, F04):  # 04 mip
                    obj['filename'] = self._get_filename(f.offset)
                if offset in self._lod_mgr_map:
                    obj['F17'] = self._lod_mgr_map[offset]
                    lod_offsets = []
                    if self._lod_level & 1 << 2:  # HI
                        lod_offsets.extend(f.children[:4])
                    if self._lod_level & 1 << 1:  # MID
                        lod_offsets.extend(f.children[4:6])
                    if self._lod_level & 1 << 0:  # LO
                        lod_offsets.extend(f.children[6:])
                    for lod_offset in lod_offsets:
                        self._read_flavor(lod_offset, offset)
                else:
                    if isinstance(f, F13):  # near -> far
                        pairs = zip(f.distances, f.children)
                        children = (c for _, c in reversed(list(pairs)))
                    else:
                        children = f.children
                    for c in children:
                        self._read_flavor(c, offset)
                    if isinstance(f, F16):
                        self._read_flavor(f.next_offset, offset)
            else:  # 00, 03, 12, 14, 15, 17, 18
                if isinstance(f, (F15, F18)):  # 3do,pmp
                    filename = self._get_filename(f.offset)
                    obj['filename'] = filename
                    if isinstance(f, F15):  # 3do
                        rads = [radians(deg) for deg in
                                map(to_degree, reversed(f.rotation))]
                        obj.location = f.location
                        obj.rotation_euler = rads
                        obj.show_name = True
                        self._external_objects.append(obj)
                        group = get_group(filename)
                        set_dupli_group(obj, group)
                    else:  # pmp 18
                        pass

    def _find_children(self, offset, *types):
        """
        find children specified by types

        :param int offset:
        :param types: int[0-18]
        """
        f = self._flavors[offset]
        if f.type in types:
            yield f.offset
        if isinstance(f, RefFlavor):
            for c_offset in f.children:
                yield from self._find_children(c_offset, *types)

    def read_model(self):
        self._model.read()
        if self._model.is_track():
            for f in self._model.body.get_flavors(17).values():  # type: F17
                self._lod_mgr_map[f.parents[0]] = f.values1.i
        # build map: {F02 offset: texture name}
        f02_f04_map = {}  # type: dict[int, int]  #
        for f04_o in sorted(self._model.body.get_flavors(4)):  # type: int
            for f02_o in self._find_children(f04_o, 2):
                f02_f04_map.setdefault(f02_o, f04_o)
        texture_names = {k: self._get_filename(v)
                         for k, v in f02_f04_map.items()}
        self._texture_names.update(texture_names)

    def register_textures(self, w=256, h=256):
        names = self._model.header.files['mip'][:]
        f02_nils = (o for o in self._model.body.get_flavors(2)
                    if o not in self._texture_names)
        if next(f02_nils, None):  # __NIL__
            names.append(NIL)
        for name in names:
            store_texture(name, w, h)

    def register_materials(self):
        f_fs = sorted(self._model.body.get_flavors(1, 2))
        for o in f_fs:
            f = self._flavors[o]  # type: FaceFlavor
            if isinstance(f, F02):
                mip_name = self._texture_names.get(f.offset, NIL)
                mtl_name = build_id(f.color, mip_name)
                register_material(mtl_name, mip_name)
            else:  # 01
                register_material(f.color)

    def _create_merged_object(self):
        """

        :return:
        :rtype: bpy.types.Object
        """
        vtc = []  # type: list[list[int]]  # [(x, y, z), (x, y, z), ...]  #
        vtx_idc = []  # type: list[list[int]]  # [[co_idx, co_idx, co_idx, ...], ...] for face
        vtx_uvs = []  # type: list[list[list[int]]]  # [[[u, v], [u, v], [u, v]], ...]  #
        offsets = sorted(self._model.body.get_flavors(1, 2))  # FaceFlavor offsets
        for o in offsets:  # type: int
            coords, idc, uvs = self._create_pydata(o, len(vtc))
            vtc.extend(coords)
            vtx_idc.append(idc)
            vtx_uvs.append(uvs)
        mesh = bpy.data.meshes.new(build_id(self.merged_obj_name, 'mesh'))
        mesh.from_pydata(vtc, [], vtx_idc)
        obj = bpy.data.objects.new(self.merged_obj_name, mesh)
        # need an object creation for following processes.
        for o, idc in zip(offsets, vtx_idc):
            name = build_id(flavor=self._flavors[o])
            obj.vertex_groups.new(name).add(idc, 1, 'ADD')
        for o, idc, uvs, poly in zip(offsets, vtx_idc, vtx_uvs, mesh.polygons):
            mtl = self._get_face_material(o)
            if isinstance(self._flavors[o], F02):
                self._set_uv_coordinates(mesh, mtl, idc, uvs)
            poly.material_index = get_material_index(mtl, mesh, True)
        return obj

    def read_flavors(self, lod_level=0b111, merge_faces=False, merge_uv_maps=False):
        self._lod_level = lod_level
        self._merge_faces = merge_faces
        # UV maps are merged by forcing because blender not allowed more than 8 UV maps.
        self._merge_uv_maps = merge_uv_maps or len(set(self._texture_names.values())) > 8
        self._read_flavor(self._model.header.root_offset)

    def link_objects(self, scale=0, context=None):
        """
        Link objects to current scene

        :param int scale: 0=auto (object=10000, track=1000000)
        :param bpy.types.Context context:
        """
        # scale
        scale = (scale or (1000000 if self._model.is_track() else 10000))
        scale_vec = mathutils.Vector.Fill(3, 1) / scale
        root_object = self._objects[self._model.header.root_offset]
        root_object.scale = scale_vec
        root_object['scale'] = scale
        for ex_obj in self._external_objects:  # F15 objects
            ex_obj.scale = mathutils.Vector.Fill(3, scale)
        # link
        objects = chain(self._objects.values(), self._dummies)
        group = get_group(os.path.basename(self._filepath))
        for i, obj in enumerate(objects):
            (context or bpy.context).scene.objects.link(obj)
            group.objects.link(obj)
        if self._merge_faces:
            mrg_obj = self._create_merged_object()
            mrg_obj['ref_source'] = True
            mrg_obj.scale = scale_vec
            (context or bpy.context).scene.objects.link(mrg_obj)

    @property
    def _flavors(self):
        return self._model.body.flavors


def load_3do(operator, context, filepath, lod_level, scale, tex_w, tex_h, merge_faces, merge_uv_maps, merged_obj_name,
             **_):
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
    :return:
    """
    importer = ModelImporter(filepath)
    if merged_obj_name:
        importer.merged_obj_name = merged_obj_name.replace('.', '_')
    importer.read_model()
    importer.register_textures(tex_h, tex_w)
    importer.register_materials()
    importer.read_flavors(lod_level, merge_faces, merge_uv_maps)
    importer.link_objects(scale, context)
    operator.report({'INFO'}, 'Model imported ({})'.format(filepath))
    return {'FINISHED'}
