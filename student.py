import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # RGB
    rgb = len(images[0].shape) == 3
    if rgb:
            # Save variables
            height, width, depth = images[0].shape  # RGB
            G = np.zeros((height, width, depth, 3))

            # Calculate G from least squares
            for color in range(depth):
                I = np.array([image[:,:,color].flatten() for image in images])  # (NxP)
                L = np.array(lights)  # (Nx3)
                G[:,:,color,:] = (np.linalg.inv(L.T @ L) @ (L.T @ I)).T.reshape((height, width, 3))

            # Calculate albedo and normals
            albedo = np.linalg.norm(G, axis=3)  # (HxWxD)
            normals = np.divide(np.mean(G, axis=2), albedo, out=np.zeros_like(albedo), where=albedo != 0)  # (HxWx3)
            normals /= np.linalg.norm(normals, axis=2)[:,:,np.newaxis]  # (HxWx3)

            return albedo, normals

    # Grayscale
    else:
        # Save variables
        height, width = images[0].shape

        # Calculate G from least squares
        I = np.array([image.flatten() for image in images])  # (NxP)
        L = np.array(lights)  # (Nx3)
        G = np.linalg.inv(L.T @ L) @ (L.T @ I)  # (3xP)

        # Calculate albedo and norms
        albedo = np.linalg.norm(G, axis=0)  # (1xP)
        normals = np.divide(G, albedo, out=np.zeros_like(G), where=albedo != 0)  # (3xP) 

        # Reshape
        normals = normals.reshape((height, width, 3))
        albedo = albedo.reshape((height, width))

        return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    raise NotImplementedError()


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    # Save variables
    import pdb; pdb.set_trace()
    height, width, depth = image.shape

    # Compute patch means, per channel
    # means = np.zeros((height - ncc_size + 1, width - ncc_size + 1, depth))
    # for i in range(means.shape[0]):
    #     for j in range(means.shape[1]):
    #         for k in range(means.shape[2]):
    #             means[i,j,k] = np.mean(image[i:i+ncc_size, j:j+ncc_size, k])

    # Copy image patch and subtract its mean, per channel
    patches = np.zeros((height - ncc_size + 1, width - ncc_size + 1, depth, ncc_size, ncc_size))
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                # Subtract mean from each patch
                patches[i,j,k,:,:] = image[i:i+ncc_size, j:j+ncc_size, k] - np.mean(image[i:i+ncc_size, j:j+ncc_size, k])

            # Divide each patch by its normal
            # patches[i,j,k,:,:] = np.divide(patches[i,j,:,:,:], np.linalg.norm(patches, axis=(0,1)),
            #                                out=np.zeros((height, width, )))
            #
            # albedo = np.linalg.norm(G, axis=3)  # (HxWxD)
            # normals = np.divide(np.mean(G, axis=2), albedo, out=np.zeros_like(albedo), where=albedo != 0)  # (HxWx3)

    print('patches!')
    return

def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    raise NotImplementedError()
