# coding: utf-8
bl_info = {'name': 'Papyrus ICR2 3DO format',
           'author': 'nooRok/BobLogSan',
           'version': (0, 3, 3),
           'blender': (2, 74, 0),
           'location': 'File > Import-Export',
           'description': 'Import/Export 3DO file',
           'warning': '',
           # 'support': 'TESTING',
           'wiki_url': '',
           'tracker_url': '',
           'category': 'Import-Export'}

if 'bpy' in locals():  # for F8
    import importlib

    if 'import_3do' in locals():
        importlib.reload(import_3do)
    if 'export_3do' in locals():
        importlib.reload(export_3do)

if 'blender modules':
    import bpy
    import mathutils
    from bpy.props import (IntProperty,
                           StringProperty,
                           BoolProperty,
                           EnumProperty)
    from bpy_extras.io_utils import (ImportHelper,
                                     ExportHelper,
                                     orientation_helper_factory,
                                     axis_conversion)
    from . import (export_3do,
                   import_3do)

import os
from logging import (getLogger,
                     Formatter,
                     StreamHandler,
                     DEBUG,
                     FileHandler)

logger = getLogger(__name__)
logger.setLevel(DEBUG)
st_hdlr = StreamHandler()

OrientationHelper = orientation_helper_factory('OrientationHelper')

log_lvs = [('50', 'CRITICAL', ''),
           ('40', 'ERROR', ''),
           ('30', 'WARNING', ''),
           ('20', 'INFO', ''),
           ('10', 'DEBUG', ''),
           ('0', 'NOTSET', ''),
           ('-1', 'DISABLE', '')]


# importer/exporter
class Import3DO(bpy.types.Operator, ImportHelper):
    bl_idname = 'import_scene.papyrus_3do'
    bl_label = 'Import 3DO'
    bl_options = {'UNDO'}
    filename_ext = '.3do'
    filter_glob = StringProperty(default='*.3do', options={'HIDDEN'})
    #
    scale = IntProperty(name='Scale',
                        description='3do model scale (1:n) '
                                    '0=auto '
                                    '(object=1:10000 / track=1:1000000)',
                        default=0)
    # trk LOD
    lod_hi = BoolProperty(name='HI', default=True)
    lod_mid = BoolProperty(name='MID', default=True)
    lod_lo = BoolProperty(name='LO', default=True)
    # default MIP size
    tex_w = IntProperty(name='Mip Width', default=256,
                        description='Default texture width')
    tex_h = IntProperty(name='Mip Height', default=256,
                        description='Default texture height')
    # merge
    merge_faces = BoolProperty(name='Merge Faces', default=True)
    merge_uv_maps = BoolProperty(name='Merge UV Maps', default=False)
    merged_obj_name = StringProperty(name='Name', default='',
                                     description="Not allowed to use a character that used for separator.")
    separator = StringProperty(name='Separator', default=':',
                               description='A character for to separate a value of object property "ref"/"reference" to reference object name and its vertex group name.')

    def execute(self, context):
        kw = self.as_keywords()
        kw['lod_level'] = (self.lod_hi << 2 | self.lod_mid << 1 | self.lod_lo)
        kw['merged_obj_name'] = (kw['merged_obj_name'] or
                                 os.path.basename(self.filepath))
        if kw['separator'] in kw['merged_obj_name']:
            self.report({'ERROR'}, "Not allowed to use a character that used for a separator.")
            return {'CANCELLED'}
        return import_3do.load(self, context, **kw)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'scale')
        box_mip = layout.box()
        box_mip.label('Default .mip size:')
        box_mip.prop(self, 'tex_w')
        box_mip.prop(self, 'tex_h')
        box_trk_lod = layout.box()
        box_trk_lod.label('Track details:')
        box_trk_lod_row = box_trk_lod.row()
        box_trk_lod_row.prop(self, 'lod_hi')
        box_trk_lod_row.prop(self, 'lod_mid')
        box_trk_lod_row.prop(self, 'lod_lo')
        box_mrg = layout.box()
        box_mrg.label('Merge options:')
        box_mrg.prop(self, 'merge_faces')
        box_mrg_obj = box_mrg.box()
        box_mrg_obj.prop(self, 'merged_obj_name')
        box_mrg_obj.prop(self, 'separator')
        box_mrg_obj.enabled = self.merge_faces
        box_mrg.prop(self, 'merge_uv_maps')


class Export3DO(bpy.types.Operator, ExportHelper, OrientationHelper):
    bl_idname = 'export_scene.papyrus_3do'
    bl_label = 'Export 3DO'
    bl_options = {'UNDO'}
    filename_ext = '.3do'
    filter_glob = StringProperty(default='*.3do', options={'HIDDEN'})
    #
    scale = IntProperty(name='Scale',
                        description='3do model scale (1*n) '
                                    '0=use object property "scale"',
                        default=0)
    origin = EnumProperty(items=[('world', 'World', ''),
                                 ('object', 'Object', '')],
                          name='Matrix origin',
                          default='world')
    # color, texture
    alt_color = IntProperty(name='Alt color',
                            description=('Color for face with no material '
                                         '-1: Random index in range 32-175 '
                                         '-2: Random index in range 0-255'),
                            default=-1,
                            min=-2,
                            max=255)
    tex_flag_ = EnumProperty(items=[('1', '1: Asphalt/Concrete', ''),
                                    ('2', '2: Grass/Dirt', ''),
                                    ('4', '4: Wall', ''),
                                    ('8', '8: Object', ''),
                                    ('16', '16: Car', ''),
                                    ('32', '32: Horizont', ''),
                                    ('64', '64: Grandstand', '')],
                             name='Texture flag',
                             description='Default texture flag',
                             default='8',
                             update=lambda s, c: setattr(s, 'tex_flag', int(s.tex_flag_)))
    tex_flag = IntProperty(default=8, options={'HIDDEN'})
    flip_uv = BoolProperty(name='Flip UV',
                           description='Flip UV vertically',
                           default=True)
    # modifier
    apply_modifiers = BoolProperty(name='Apply Object Modifiers',
                                   default=True)
    separator = StringProperty(name='Separator', default=':',
                               description='A character for to separate a value of object property "ref" '
                                           'to the referenced object name and its vertex group name.')
    # ex obj
    f15_rot_space = EnumProperty(items=[('basis', 'Basis', 'Local Space'),
                                        ('parent', 'Parent', 'Local Space'),
                                        ('local', 'Local', 'Local Space'),
                                        ('world', 'World', 'World Space')],
                                 name='F15 Matrix',
                                 description='Matrix space for F15 object rotation',
                                 default='basis')
    #
    export_all = BoolProperty(name='Export All Selected Objects',
                              description='Filenames are taken from each selected object names. '
                                          'Input box value is ignored.',
                              default=False)
    # logging
    c_log_lv_ = EnumProperty(items=log_lvs[:-1],
                             name='Console logging level',
                             description='Logging threshold level for console',
                             default='30',
                             update=lambda s, c: setattr(s, 'c_log_lv', int(s.c_log_lv_)))
    c_log_lv = IntProperty(default=30, options={'HIDDEN'})
    f_log_lv_ = EnumProperty(items=log_lvs,
                             name='File logging level',
                             description='Logging threshold level for logfile',
                             default='-1',
                             update=lambda s, c: setattr(
                                 s, 'f_log_lv', int(s.f_log_lv_)))
    f_log_lv = IntProperty(default=-1, options={'HIDDEN'})

    def __enter__(self):
        st_hdlr.setLevel(self.c_log_lv)
        if self.f_log_lv >= 0:
            hdlr = FileHandler(filename='{}.log'.format(self.filepath))
            hdlr.setLevel(self.f_log_lv)
            hdlr.setFormatter(Formatter('%(funcName)s: %(message)s'))
            logger.addHandler(hdlr)

    def __exit__(self, exc_type, exc_val, exc_tb):
        handlers = [h for h in logger.handlers
                    if isinstance(h, FileHandler)]
        for h in handlers:
            h.close()
            logger.removeHandler(h)

    def execute(self, context):
        with self:
            try:
                kw = self.as_keywords()
                kw['default_texture_flag'] = kw.pop('tex_flag')
                rot_mx = axis_conversion(to_forward=self.axis_forward,
                                         to_up=self.axis_up).to_4x4()
                all_ = self.export_all and len(context.selected_objects) > 1
                objs = (context.selected_objects if all_ else
                        [context.active_object])
                for obj in objs:  # type: bpy.types.Object
                    kw['obj'] = obj
                    kw['scale'] = self.scale or int(obj.get('scale', 1))
                    origin = (obj.get('origin') or self.origin).lower()  # type: str  # 'world'|'object'
                    if origin == 'world':
                        kw['matrix'] = rot_mx  # mathutils.Matrix()
                    else:  # object
                        assert self.origin == 'object'
                        obj_mx = context.active_object.matrix_world  # type: mathutils.Matrix
                        tr, rot, sc = obj_mx.decompose()  # type: mathutils.Vector, mathutils.Quaternion, mathutils.Vector
                        kw['matrix'] = mathutils.Matrix.Translation(tr) * rot_mx
                    if all_:
                        dirname, basename = os.path.split(self.filepath)
                        _, ext = os.path.splitext(basename)
                        filename = '{}{}'.format(obj.name.split('.')[0], ext)
                        kw['filepath'] = os.path.join(dirname, filename)
                    logger.info(kw['filepath'])
                    export_3do.save(self, context, **kw)
                logger.info('Finished')
                self.report({'INFO'}, 'Finished: ({})'.format(self.filepath))
                return {'FINISHED'}
            except Exception as err:
                logger.exception(err)
                logger.info('Failed')
                self.report({'ERROR'}, 'Cancelled: ({})'.format(str(err)))
                return {'CANCELLED'}

    def draw(self, context):
        layout = self.layout
        box_mx = layout.box()
        box_mx.label('Matrix:')
        box_mx.prop(self, 'scale')
        box_mx.prop(self, 'axis_forward')
        box_mx.prop(self, 'axis_up')
        box_mx.prop(self, 'origin')
        box_mx.prop(self, 'f15_rot_space')
        box_mtl = layout.box()
        box_mtl.label('Material:')
        box_mtl.prop(self, 'alt_color')
        box_mtl.prop(self, 'tex_flag_')
        box_mtl.prop(self, 'flip_uv')
        box = layout.box()
        box.prop(self, 'separator')
        box.prop(self, 'apply_modifiers')
        if len(context.selected_objects) > 1:
            box.prop(self, 'export_all')
        box_log = layout.box()
        box_log.prop(self, 'c_log_lv_')
        box_log.prop(self, 'f_log_lv_')


# handler
def import_3do_menu_handler(self, context):
    self.layout.operator(Import3DO.bl_idname,
                         text='Papyrus ICR2 (.3do)')


def export_3do_menu_handler(self, context):
    self.layout.operator(Export3DO.bl_idname,
                         text='Papyrus ICR2 (.3do)')


def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(import_3do_menu_handler)
    bpy.types.INFO_MT_file_export.append(export_3do_menu_handler)
    logger.addHandler(st_hdlr)
    print(__name__, 'registered')


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(import_3do_menu_handler)
    bpy.types.INFO_MT_file_export.remove(export_3do_menu_handler)
    logger.removeHandler(st_hdlr)
    print(__name__, 'unregistered')
