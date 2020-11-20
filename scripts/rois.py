#!/usr/bin/env python

# Self contained script, based on Simon's script
# https://raw.githubusercontent.com/IDR/idr0052-walther-condensinmap/master/scripts/upload_and_create_rois.py
# and omero-roi package: https://github.com/ome/omero-rois

import os
import numpy as np
import omero
from omero.gateway import BlitzGateway
from omero.gateway import ColorHolder
from omero.model import MaskI
from omero.rtypes import (
    rdouble,
    rint,
    rstring,
)

PROJECT = "idr0100-capar-myelin/experimentA"
RGBA = (255, 255, 0, 128)
DRYRUN = False

def mask_from_binary_image(
      binim, rgba=None, z=None, c=None, t=None, text=None,
      raise_on_no_mask=True):
    """
    Create a mask shape from a binary image (background=0)

    :param numpy.array binim: Binary 2D array, must contain values [0, 1] only
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param z: Optional Z-index for the mask
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: An OMERO mask
    :raises NoMaskFound: If no labels were found
    :raises InvalidBinaryImage: If the maximum labels is greater than 1
    """

    # Find bounding box to minimise size of mask
    xmask = binim.sum(0).nonzero()[0]
    ymask = binim.sum(1).nonzero()[0]
    if any(xmask) and any(ymask):
        x0 = min(xmask)
        w = max(xmask) - x0 + 1
        y0 = min(ymask)
        h = max(ymask) - y0 + 1
        submask = binim[y0:(y0 + h), x0:(x0 + w)]
        if (not np.array_equal(np.unique(submask), [0, 1]) and not
        np.array_equal(np.unique(submask), [1])):
            raise
    else:
        if raise_on_no_mask:
            raise
        x0 = 0
        w = 0
        y0 = 0
        h = 0
        submask = []

    mask = MaskI()
    # BUG in older versions of Numpy:
    # https://github.com/numpy/numpy/issues/5377
    # Need to convert to an int array
    # mask.setBytes(np.packbits(submask))
    mask.setBytes(np.packbits(np.asarray(submask, dtype=int)))
    mask.setWidth(rdouble(w))
    mask.setHeight(rdouble(h))
    mask.setX(rdouble(x0))
    mask.setY(rdouble(y0))

    if rgba is not None:
        ch = ColorHolder.fromRGBA(*rgba)
        mask.setFillColor(rint(ch.getInt()))
    if z is not None:
        mask.setTheZ(rint(z))
    if c is not None:
        mask.setTheC(rint(c))
    if t is not None:
        mask.setTheT(rint(t))
    if text is not None:
        mask.setTextValue(rstring(text))

    return mask


def masks_from_label_image(
      labelim, rgba=None, z=None, c=None, t=None, text=None,
      raise_on_no_mask=False):
    """
    Create mask shapes from a label image (background=0)

    :param numpy.array labelim: 2D label array
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param z: Optional Z-index for the mask
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: A list of OMERO masks in label order ([] if no labels found)

    """
    masks = []
    for i in range(1, labelim.max() + 1):
        mask = mask_from_binary_image(labelim == i, rgba, z, c, t, text,
                                      raise_on_no_mask)
        masks.append(mask)
    return masks


def get_images(conn):
    project = conn.getObject('Project', attributes={'name': PROJECT})
    for dataset in project.listChildren():
        for image in dataset.listChildren():
            if len(image.name) > 8:
                continue
            yield image


def get_segmented_image(conn, image):
    name = image.name+"_Ground_Truth"
    try:
        result = conn.getObject('Image', attributes={'name': name})
        return result
    except Exception as e:
        print("Could not find {} for {}".format(name, image.name))
        return None


def save_rois(conn, im, rois):
    print("Saving {} ROIs for image {}:{}".format(len(rois), im.id, im.name))
    us = conn.getUpdateService()
    for roi in rois:
        im = conn.getObject('Image', im.id)
        roi.setImage(im._obj)
        us.saveAndReturnObject(roi)


def delete_rois(conn, im):
    result = conn.getRoiService().findByImage(im.id, None)
    to_delete = []
    for roi in result.rois:
        to_delete.append(roi.getId().getValue())
    if to_delete:
        print("Deleting existing {} rois".format(len(to_delete)))
        conn.deleteObjects("Roi", to_delete, deleteChildren=True, wait=True)


def create_rois(seg_img):
    zct_list = []
    for z in range(0, seg_img.getSizeZ()):
        zct_list.append((z, 0, 0))
    print("Going through {} planes".format(len(zct_list)))
    planes = seg_img.getPrimaryPixels().getPlanes(zct_list)

    rois = []
    for i, plane in enumerate(planes):
        plane_masks = masks_from_label_image(plane, rgba=RGBA, z=i, c=None,
                                             t=None, text=None,
                                             raise_on_no_mask=False)
        for label, mask in enumerate(plane_masks):
            if mask.getBytes().any():
                roi = omero.model.RoiI()
                roi.addShape(mask)
                rois.append(roi)

    print("{} rois created.".format(len(rois)))
    return rois


def main(conn):
    for im in get_images(conn):
        seg_im = get_segmented_image(conn, im)
        if seg_im is None:
            continue
        try:
            print("Processing {} - {}".format(im.name, seg_im.name))
            delete_rois(conn, im)
            rois = create_rois(seg_im)
            if not DRYRUN and len(rois) > 0:
                save_rois(conn, im, rois)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    host = os.environ.get('OMERO_HOST', 'localhost')
    port = os.environ.get('OMERO_PORT', '4064')
    user = os.environ.get('OMERO_USER', 'NA')
    pw = os.environ.get('OMERO_PASSWORD', 'NA')
    with BlitzGateway(user, pw, host=host, port=port) as conn:
        main(conn)
