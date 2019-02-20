# coding: utf-8
bl_info = {'name': 'Papyrus ICR2 3DO format',
           'author': 'nooRok/BobLogSan',
           'version': (0, 2, 1),
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

from logging import (getLogger,
                     Formatter,
                     StreamHandler,
                     DEBUG,
                     FileHandler,
                     NullHandler)
from pathlib import Path

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
    scale = IntProperty(name='Scale',
                        description='3do model scale (1:n) '
                                    '0=auto '
                                    '(object=1:10000 / track=1:1000000)',
                        default=0)
    # grp = BoolProperty(name='Group data tree', default=True)
    lod_hi = BoolProperty(name='HI', default=True)
    lod_mid = BoolProperty(name='MID', default=True)
    lod_lo = BoolProperty(name='LO', default=True)
    tex_w = IntProperty(name='Mip Width', default=256,
                        description='Default texture width')
    tex_h = IntProperty(name='Mip Height', default=256,
                        description='Default texture height')
    merge_faces = BoolProperty(name='Merge Faces', default=True)
    merge_uv_maps = BoolProperty(name='Merge UV Maps', default=False)
    merged_obj_name = StringProperty(name='Name', default='',
                                     description='"." is replaced to "_" automatically.')

    def execute(self, context):
        lod_lv = (self.lod_hi << 2 | self.lod_mid << 1 | self.lod_lo)
        return import_3do.load_3do(
            self, context, lod_level=lod_lv, **self.as_keywords())

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'scale')
        box_mip_size = layout.box()
        box_mip_size.label('Default .mip size:')
        box_mip_size.prop(self, 'tex_w')
        box_mip_size.prop(self, 'tex_h')
        box_trk_lod = layout.box()
        box_trk_lod.label('Track details:')
        box_trk_lod_row = box_trk_lod.row()
        box_trk_lod_row.prop(self, 'lod_hi')
        box_trk_lod_row.prop(self, 'lod_mid')
        box_trk_lod_row.prop(self, 'lod_lo')
        box_mrg = layout.box()
        box_mrg.label('Merge options:')
        box_mrg.prop(self, 'merge_faces')
        box_mrg.prop(self, 'merge_uv_maps')
        box_mrg.prop(self, 'merged_obj_name')


class Export3DO(bpy.types.Operator, ExportHelper, OrientationHelper):
    bl_idname = 'export_scene.papyrus_3do'
    bl_label = 'Export 3DO'
    bl_options = {'UNDO'}
    filename_ext = '.3do'
    filter_glob = StringProperty(default='*.3do', options={'HIDDEN'})
    scale = IntProperty(name='Scale',
                        description='3do model scale (1*n) '
                                    '0=use object property "scale"',
                        default=0)
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
    tex_from = EnumProperty(items=[('material', 'Material', ''),
                                   ('active', 'Active UV Map', '')],
                            name='Texture Image From',
                            description='Texture image from',
                            default='material')
    flip_uv = BoolProperty(name='Flip UV',
                           description='Flip UV vertically',
                           default=True)
    apply_modifiers = BoolProperty(name='Apply Object Modifiers',
                                   default=True)
    f15_rot_space = EnumProperty(items=[('basis', 'Basis', 'Local Space'),
                                        ('parent', 'Parent', 'Local Space'),
                                        ('local', 'Local', 'Local Space'),
                                        ('world', 'World', 'World Space')],
                                 name='F15 Matrix',
                                 description='Matrix space for F15 object rotation',
                                 default='basis')
    apply_global_mx_f15 = BoolProperty(name='Apply Global Matrix',
                                       description='Apply global orientation to F15 objects',
                                       default=False)
    export_all = BoolProperty(name='Export All Selected Objects',
                              description='Object name is used as filename. '
                                          'Input box value is ignored.',
                              default=False)
    c_log_lv_ = EnumProperty(items=log_lvs[:-1],
                             name='Console logging level',
                             description='Logging threshold level for console',
                             default='20',
                             update=lambda s, c: setattr(s, 'c_log_lv', int(s.c_log_lv_)))
    c_log_lv = IntProperty(default=20, options={'HIDDEN'})
    f_log_lv_ = EnumProperty(items=log_lvs,
                             name='File logging level',
                             description='Logging threshold level for logfile',
                             default='-1',
                             update=lambda s, c: setattr(
                                 s, 'f_log_lv', int(s.f_log_lv_)))
    f_log_lv = IntProperty(default=-1, options={'HIDDEN'})

    def execute(self, context):
        # logger
        path = Path(self.filepath)
        st_hdlr.setLevel(self.c_log_lv)
        hdlr = (NullHandler() if self.f_log_lv < 0 else
                FileHandler(filename=str(path.with_suffix('.log'))))
        hdlr.setLevel(self.f_log_lv)
        hdlr.setFormatter(Formatter('%(funcName)s: %(message)s'))
        logger.addHandler(hdlr)
        try:
            kwargs = self.as_keywords()
            kwargs['texture_from'] = kwargs.pop('tex_from')
            kwargs['default_texture_flag'] = kwargs.pop('tex_flag')
            rot_mx = axis_conversion(to_forward=self.axis_forward, to_up=self.axis_up).to_4x4()
            all_ = len(context.selected_objects) > 1 and self.export_all
            objs = context.selected_objects if all_ else [context.active_object]
            for obj in objs:  # type: bpy.types.Object
                if all_:
                    filename = '{}{}'.format(obj.name.split('.')[0], path.suffix)
                    filepath = str(Path(path.parent, filename))
                    kwargs['filepath'] = filepath
                    kwargs['obj'] = obj
                scl_mx = mathutils.Matrix.Scale(self.scale or obj.get('scale', 1), 4)
                kwargs['matrix'] = scl_mx * rot_mx
                logger.info(kwargs['filepath'])
                export_3do.save(self, context, **kwargs)
            logger.info('Finished')
            self.report({'INFO'}, 'Finished: ({})'.format(self.filepath))
            return {'FINISHED'}
        except Exception as err:
            logger.exception(err)
            logger.info('Failed')
            self.report({'ERROR'}, 'Cancelled: ({})'.format(str(err)))
            return {'CANCELLED'}
        finally:
            hdlr.close()
            logger.removeHandler(hdlr)

    def draw(self, context):
        layout = self.layout
        box_mx = layout.box()
        box_mx.label('Global Matrix:')
        box_mx.prop(self, 'scale')
        box_mx.prop(self, 'axis_forward')
        box_mx.prop(self, 'axis_up')
        box_f15_mx = layout.box()
        box_f15_mx.label('F15 Matrix:')
        box_f15_mx.prop(self, 'f15_rot_space')
        box_f15_mx.prop(self, 'apply_global_mx_f15')
        box_mtl = layout.box()
        box_mtl.prop(self, 'alt_color')
        box_mtl.prop(self, 'tex_flag_')
        box_mtl.prop(self, 'tex_from')
        box_mtl.prop(self, 'flip_uv')
        layout.prop(self, 'apply_modifiers')
        if len(context.selected_objects) > 1:
            layout.prop(self, 'export_all')
        box_log = layout.box()
        box_log.prop(self, 'c_log_lv_')
        box_log.prop(self, 'f_log_lv_')


class Export3D(bpy.types.Operator, ExportHelper, OrientationHelper):
    """ **experimental** """
    bl_idname = 'export_scene.papyrus_3d'
    bl_label = 'Export 3D'
    # bl_options = {'UNDO'}
    # filename_ext = '.3d'
    # filter_glob = StringProperty(default='*.3d', options={'HIDDEN'})
    # scale = Export3DO.scale
    # alt_color = Export3DO.alt_color
    # tex_flag_ = Export3DO.tex_flag_
    # tex_flag = Export3DO.tex_flag
    # tex_from = Export3DO.tex_from
    # flip_uv = Export3DO.flip_uv
    # apply_modifiers = Export3DO.apply_modifiers
    # f15_rot_space = Export3DO.f15_rot_space
    # export_all = Export3DO.export_all
    # c_log_lv_ = Export3DO.c_log_lv_
    # c_log_lv = Export3DO.c_log_lv
    # f_log_lv_ = Export3DO.f_log_lv_
    # f_log_lv = Export3DO.f_log_lv
    #
    # execute = Export3DO.execute
    # draw = Export3DO.draw
    pass


# handler
def import_3do_menu_handler(self, context):
    self.layout.operator(Import3DO.bl_idname,
                         text='Papyrus ICR2 (.3do)')


def export_3do_menu_handler(self, context):
    self.layout.operator(Export3DO.bl_idname,
                         text='Papyrus ICR2 (.3do)')
    # self.layout.operator(Export3D.bl_idname,
    #                      text='N3tools 3D (.3d)')


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
