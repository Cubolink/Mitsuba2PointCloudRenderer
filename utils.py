import colorsys
import glob

import mitsuba
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def colormap_gray(x, y, z):
    vec = np.array([x, y, z])
    norm = np.sqrt(np.sum(vec ** 2))  # Compute the Euclidean distance from the origin
    norm = np.clip(norm, 0.0, 1.0)  # Clip the norm to stay within the range [0, 1]

    # Create a grayscale value based on the distance (norm)
    value = norm

    # Return the grayscale color in the RGB format
    return [value, value, value]


def colormap_hsv_value_gradient(x, y, z, hue, saturation):
    vec = np.array([x, y, z])
    norm = np.sqrt(np.sum(vec ** 2))  # Compute the Euclidean distance from the origin
    norm = np.clip(norm, 0.0, 1.0)  # Clip the norm to stay within the range [0, 1]

    value = norm  # Value based on the distance (norm)

    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    return [rgb[0], rgb[1], rgb[2]]


def sample_pcl(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    return pcl[pt_indices]


def estimate_center_scale(pcl):
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    debug_msg("Center: {}, Scale: {}".format(center, scale))
    return [center, scale]


def standardize_pc(pcl, center, scale):
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


def standardize_bbox(pcl, points_per_object):
    pcl = sample_pcl(pcl, points_per_object)
    center, scale = estimate_center_scale(pcl)
    return standardize_pc(pcl, center, scale)


def get_files(pattern):
    return glob.glob(pattern)


def rreplace(s, old, new, occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def get_images(files, replace_dots=False):
    images = []
    for file in files:
        if replace_dots != False:
            replacement = replace_dots if isinstance(replace_dots,str) else '_'
            file = rreplace(file, '.', replacement)
        pattern = f'{file}*.png'
        imgs = sorted(glob.glob(pattern))
        images.extend(imgs)
    return images


def merge_renders(renders, filename, direction='vertical', caption=True):
    if direction not in ['horizontal', 'vertical']:
        raise ValueError(f"Unknown '{direction}' direction when merging renders")

    images = []
    try:
        font = ImageFont.truetype("FreeSans.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    combined_size = {'width': 0, 'height': 0}
    for render in renders:
        image = Image.open(render)
        factor = 540 / image.height
        image = image.resize((int(factor * image.width), 540))
        file = render.split('/')[-1]

        if caption:
            # Draw text
            ImageDraw.Draw(image).text((30, 10), file, fill=(128, 128, 128), font=font)

        images.append(image)

        if direction == 'vertical':
            combined_size['width'] = max(combined_size['width'], images[-1].width)
            combined_size['height'] += images[-1].height
        elif direction == 'horizontal':
            combined_size['width'] += images[-1].width
            combined_size['height'] = max(combined_size['height'], images[-1].height)

    combined_image = Image.new("RGB", (combined_size["width"], combined_size["height"]))

    offset = 0
    for image in images:
        if direction == 'vertical':
            combined_image.paste(image, (0, offset))
            offset += image.height
        elif direction == 'horizontal':
            combined_image.paste(image, (offset, 0))
            offset += image.width
        image.close()

    print("Saving", filename)
    combined_image.save(filename)


def render_xml(xml_file, png_file):
    debug_msg(['Running Mitsuba, loading: ', xml_file])
    scene = mitsuba.load_file(xml_file)
    render = mitsuba.render(scene)
    debug_msg(['writing to: ', png_file])
    mitsuba.util.write_bitmap(png_file, render)


def write_xml(xml_file, content):
    debug_msg(['Writing to: ', xml_file])
    with open(xml_file, 'w') as f:
        f.write(content)


def debug_msg(*args,**kwargs):
    pass
