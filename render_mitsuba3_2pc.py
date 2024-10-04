import argparse
import glob
import os
import os.path
import re

import mitsuba
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from plyfile import PlyData
from render_mitsuba3 import xml_head, xml_ball_segment, xml_tail
from utils import (
    colormap,
    merge_renders,
    write_xml,
    render_xml,
    debug_msg,
    colormap_hsv_value_gradient
)

xml_ball_segment_partial = \
    """
    <shape type="sphere">
        <float name="radius" value="0.007"/>
        <transform name="to_world">
            <translate value="{}, {}, {}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_ball_segment_evaluated = \
    """
    <shape type="sphere">
        <float name="radius" value="0.007"/>
        <transform name="to_world">
            <translate value="{}, {}, {}"/>
        </transform>
        <bsdf type="pplastic">
            <rgb name="diffuse_reflectance" value="{},{},{}"/>
            <float name="alpha" value="0.06"/>
        </bsdf>
    </shape>
"""


def read_ply(path):
    ply = PlyData.read(path)
    vertex = ply['vertex']
    (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
    (r, g, b) = (vertex[t] for t in ('red', 'green', 'blue'))

    pcl_time = np.column_stack((x, y, z))
    pcl_time_rgb = np.column_stack((r, g, b))
    if len(np.shape(pcl_time)) < 3:
        new_size = (1,) + np.shape(pcl_time)
        pcl_time.resize(new_size)
        pcl_time_rgb.resize(new_size)

    return pcl_time, pcl_time_rgb


def read_obj(path):
    import pywavefront
    
    evaluated_vertices = pywavefront.Wavefront(path)
    evaluated_vertices = np.array(evaluated_vertices.vertices)
    
    pcl_time_rgb = evaluated_vertices[:, 3:]
    pcl_time = evaluated_vertices[:, :3]
    
    if len(np.shape(pcl_time)) < 3:
        new_size = (1,) + np.shape(pcl_time)
        pcl_time.resize(new_size)
        if pcl_time_rgb.shape[1] == 3:
            pcl_time_rgb.resize(new_size)
        else:
            # no color info
            pcl_time_rgb = None
    
    return pcl_time, pcl_time_rgb


def standardize_bbox2(pcl1, pcl2, points_per_object):
    """
    Standardizes to [-0.5, 0.5], but considering the union of pcl1 and pcl2, so the standardization is not independent

    :points_per_object: Total points, it will be weighted.
    """
    # weight the points per object (ppo)
    total_n = pcl1.shape[0] + pcl2.shape[0]
    pcl1_ppo = int(pcl1.shape[0] * (points_per_object / total_n))
    pcl2_ppo = points_per_object - pcl1_ppo

    # sample pcl1
    pt1_indices = np.random.choice(pcl1.shape[0], pcl1_ppo, replace=False)
    np.random.shuffle(pt1_indices)
    pcl1 = pcl1[pt1_indices]
    # sample pcl2
    pt2_indices = np.random.choice(pcl2.shape[0], pcl2_ppo, replace=False)
    np.random.shuffle(pt2_indices)
    pcl2 = pcl2[pt2_indices]

    pcl_concat = np.concatenate((pcl1, pcl2), axis=0)
    mins = np.amin(pcl_concat, axis=0)
    maxs = np.amax(pcl_concat, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    debug_msg("Center: {}, Scale: {}".format(center, scale))

    # [-0.5, 0.5]
    pcl1 = ((pcl1 - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    pcl2 = ((pcl2 - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return pcl1, pcl2


def main2(partial_file_path, evaluated_file_path, out_file_path, num_points_per_object, forced=False):
    filename, out_file_extension = os.path.splitext(out_file_path)
    _, in_file_extension = os.path.splitext(partial_file_path)
    # filename = os.path.basename(filename)
    # folder = os.path.dirname(partial_file_path)
    
    if in_file_extension == '.ply':
        partial_pcl_time, _ = read_ply(partial_file_path)
        evaluated_pcl_time, color_evaluated_pcl_time = read_ply(evaluated_file_path)
    elif in_file_extension == '.obj':
        partial_pcl_time, _ = read_obj(partial_file_path)
        evaluated_pcl_time, color_evaluated_pcl_time = read_obj(evaluated_file_path)
    else:
        print(f"unsupported {in_file_extension} file")
        return

    for pcli in range(0, np.shape(partial_pcl_time)[0]):
        partial_pcl = partial_pcl_time[pcli, :, :]
        evaluated_pcl = evaluated_pcl_time[pcli, :, :]
        if color_evaluated_pcl_time is not None:
            # filter black and dark-gray points in evaluated_pcl
            indices = np.where(color_evaluated_pcl_time[pcli][:, 0] >= 10)[0]
            evaluated_pcl = evaluated_pcl[indices]

        # standardize both pcl
        partial_pcl, evaluated_pcl = standardize_bbox2(partial_pcl, evaluated_pcl, num_points_per_object)

        def add_xml_segments(pcl, colormap_fun=colormap, xml_ball_segment=xml_ball_segment):
            pcl = pcl[:, [2, 0, 1]]
            pcl[:, 0] *= -1
            pcl[:, 2] += 0.0125

            for i in range(pcl.shape[0]):
                color = colormap_fun(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
                xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))

        def colormap1(x, y, z):
            return colormap_hsv_value_gradient(x, y, z, 5.0 / 360, 0.9)

        def colormap2(x, y, z):
            return colormap_hsv_value_gradient(x, y, z, 139 / 360, 0.9)

        xml_segments = [xml_head]
        add_xml_segments(partial_pcl, colormap1, xml_ball_segment=xml_ball_segment_partial)
        add_xml_segments(evaluated_pcl, colormap2, xml_ball_segment=xml_ball_segment_evaluated)
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        xml_file = f"{filename}_{pcli:02d}.xml"  # os.path.join(folder, f"{filename}_restored_{pcli:02d}.xml")
        png_file = xml_file.replace('.xml', out_file_extension)
        if forced or (not os.path.exists(png_file)):
            write_xml(xml_file, xml_content)
            render_xml(xml_file, png_file)
        else:
            debug_msg('skipping rendering because the file already exist')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("basename",
                        help="basename of the .ply partial and evaluated files, or name of the folder to process")
    parser.add_argument("-n", "--num_points_per_object", type=int, default=2048)
    parser.add_argument("-v", "--mitsuba_variant", type=str, choices=mitsuba.variants(), default="scalar_rgb")
    parser.add_argument("-j", "--join_renders", type=eval, default=True)
    parser.add_argument('-u', '--up_axis', type=str, choices=['x','y','z'], default='z', help='Axis considered height')
    parser.add_argument('-f', '--force_render', action='store_true', help='overwrite existing renders')
    parser.add_argument('-m', '--multiview_number', type=int, default=1, help='How many views per object')
    parser.add_argument('-d', '--debug', action='store_true')

    parser.add_argument("--partial_suffix", type=str, default="_points.partial.ply")
    parser.add_argument("--evaluated_suffix", type=str, default="_evaluated_pc.ply")
    parser.add_argument("--restored_suffix", type=str, default="_restored_pc.png")

    return parser.parse_args()


def get_pairs_in_folder(partial_suffix, evaluated_suffix, pattern):
    pairs = []

    partial_files = glob.glob(pattern + partial_suffix)
    for partial_ply_path in partial_files:
        evaluated_ply_path = partial_ply_path.rsplit(partial_suffix, 1)[0] + evaluated_suffix
        if os.path.exists(evaluated_ply_path):
            pairs.append([partial_ply_path, evaluated_ply_path])

    return pairs


def get_rendered_views_groups(base_path):
    """
    Using a base_path, we expect to find files with a suffix _(viewX)_Y added.
    This function gets and groups by Y all of these files, Y being the scene number.

    :param base_path:
    :return: list of grouped files
    """
    base_path, base_path_ext = os.path.splitext(base_path)
    # First we get all files related to the rendered object
    pattern = base_path + '_(view*)_*' + base_path_ext
    renders = sorted(glob.glob(pattern))

    grouped_renders = {}
    out_paths = {}
    # Then, we do pattern matching.
    group_pattern = r'([a-zA-Z0-9_\\]+)_\(view(\d+)\)_(\d+)\.png'
    for render in renders:
        match = re.search(group_pattern, render)
        # match.group(1) should be always the same, since we called glob with a base pattern.
        # view_number = int(match.group(2))
        scene_number = match.group(3)
        if scene_number not in grouped_renders:
            grouped_renders[scene_number] = []
            out_paths[scene_number] = base_path + '_' + scene_number + base_path_ext
        grouped_renders[scene_number].append(render)

    return grouped_renders, out_paths


def get_restored_renders(pattern):
    pattern, ext = os.path.splitext(pattern)
    pattern = pattern + "*" + ext  # To look for stuff like restored_pc_01.png, not just restored_pc.png
    # But we need to exclude the (view n) files, so we only used the single/multiview files.
    regex_view = re.compile(r".*\(view\d+\).*")

    restored_files = sorted([file for file in glob.glob(pattern) if not regex_view.search(file)])
    return restored_files


if __name__ == "__main__":
    args = parse_args()
    mitsuba.set_variant(args.mitsuba_variant)
    
    up = args.up_axis
    up_x, up_y, up_z = ('x' in up), ('y' in up), ('z' in up)
    up_vec = f'{1&up_x},{1&up_y},{1&up_z}'
    origin_vec = '3,3,3'
    xml_head_base = xml_head
    xml_tail = xml_tail.format(up_vec)
    if args.debug:
        debug_msg = print
    
    pair_paths = get_pairs_in_folder(args.partial_suffix, args.evaluated_suffix, args.basename)
    for path in tqdm(pair_paths):
        try:
            out_path = path[0].rsplit(args.partial_suffix, 1)[0] + args.restored_suffix
            if args.multiview_number > 1:
                out_path_ext = os.path.splitext(out_path)[1]
                for i_view in range(args.multiview_number):
                    phi = 2*pi*(i_view/args.multiview_number)
                    rotated_origin_vec = Rotation.from_euler(up, -phi).apply(np.array(origin_vec.split(','), dtype=int))
                    xml_head = xml_head_base.format(
                        ','.join([f'{c:.3f}' for c in rotated_origin_vec]),  # array to string 'x,y,z' like
                        up_vec
                    )

                    main2(path[0],
                          path[1],
                          out_path.replace(out_path_ext, f'_(view{i_view:02d})'+out_path_ext),
                          args.num_points_per_object,
                          forced=args.force_render)
                groups, paths = get_rendered_views_groups(out_path)
                for k in groups:
                    merge_renders(
                        groups[k],
                        paths[k],
                        direction='horizontal',
                        caption=False
                    )

            else:
                xml_head = xml_head_base.format(origin_vec, up_vec)
                main2(path[0],
                      path[1],
                      out_path,
                      args.num_points_per_object,
                      forced=args.force_render)
        except Exception as e:
            print(e)

    if args.join_renders:
        merge_renders(get_restored_renders(args.basename + args.restored_suffix), args.basename[:-1] + "combined.png")
