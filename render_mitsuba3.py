import argparse
import numpy as np
import sys, os, subprocess
from PIL import Image
from plyfile import PlyData, PlyElement
import mitsuba

from utils import standardize_bbox, colormap

# replaced by command line arguments
# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load

# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
xml_head = \
    """
<scene version="3.0.0">
    <integrator type="path">
        <integer name="max_depth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="far_clip" value="100"/>
        <float name="near_clip" value="0.1"/>
        <transform name="to_world">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sample_count" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surface_material">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="int_ior" value="1.46"/>
        <rgb name="diffuse_reflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

# I also use a smaller point size
xml_ball_segment = \
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

xml_tail = \
    """
    <shape type="rectangle">
        <ref name="bsdf" id="surface_material"/>
        <transform name="to_world">
            <scale value="10, 10, 1"/>
            <translate value="0, 0, -0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="to_world">
            <scale value="10, 10, 1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def main(args):
    mitsuba.set_variant(args.mitsuba_variant)
    
    pathToFile = args.filename

    filename, file_extension = os.path.splitext(pathToFile)
    folder = os.path.dirname(pathToFile)
    filename = os.path.basename(pathToFile)

    # for the moment supports npy and ply
    if (file_extension == '.npy'):
        pclTime = np.load(pathToFile)
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == '.npz'):
        pclTime = np.load(pathToFile)
        pclTime = pclTime['pred']
        pclTimeSize = np.shape(pclTime)
    elif (file_extension == '.ply'):
        ply = PlyData.read(pathToFile)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pclTime = np.column_stack((x, y, z))
    else:
        print('unsupported file format.')
        return

    if (len(np.shape(pclTime)) < 3):
        pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
        pclTime.resize(pclTimeSize)

    for pcli in range(0, pclTimeSize[0]):
        pcl = pclTime[pcli, :, :]
        
        pcl = standardize_bbox(pcl, args.num_points_per_object)
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125

        xml_segments = [xml_head]
        for i in range(pcl.shape[0]):
            color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        xmlFile = os.path.join(folder, f"{filename}_{pcli:02d}.xml")
        print(['Writing to: ', xmlFile])

        with open(xmlFile, 'w') as f:
            f.write(xml_content)
        f.close()
        
        png_file = os.path.join(folder, f"{filename}_{pcli:02d}.png")
        if (not os.path.exists(png_file)):
            print(['Running Mitsuba, loading: ', xmlFile])
            scene = mitsuba.load_file(xmlFile)
            render = mitsuba.render(scene)
            print(['writing to: ', png_file])
            mitsuba.util.write_bitmap(png_file, render)
        else:
            print('skipping rendering because the file already exists')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename to npy/ply")
    parser.add_argument("-n", "--num_points_per_object", type=int, default=2048)
    parser.add_argument("-v", "--mitsuba_variant", type=str, choices=mitsuba.variants(), default="scalar_rgb")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())