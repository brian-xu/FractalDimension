import numpy as np


def fractal_dimension(image: np.ndarray) -> np.float64:
    """ Calculates the fractal dimension of an image represented by a 2D numpy array.

    The algorithm is a modified box-counting algorithm as described by Wen-Li Lee and Kai-Sheng Hsieh.

    Args:
        image: A 2D array containing a grayscale image. Format should be equivalent to cv2.imread(flags=0).
               The size of the image has no constraints, but it needs to be square (m×m array).
    Returns:
        D: The fractal dimension Df, as estimated by the modified box-counting algorithm.
    """
    M = image.shape[0]  # image shape
    G_min = image.min()  # lowest gray level (0=white)
    G_max = image.max()  # highest gray level (255=black)
    G = G_max - G_min + 1  # number of gray levels, typically 256
    prev = -1  # used to check for plateaus
    r_Nr = []

    for L in range(2, (M // 2) + 1):
        h = max(1, G // (M // L))  # minimum box height is 1
        N_r = 0
        r = L / M
        for i in range(0, M, L):
            boxes = [[]] * ((G + h - 1) // h)  # create enough boxes with height h to fill the fractal space
            for row in image[i:i + L]:  # boxes that exceed bounds are shrunk to fit
                for pixel in row[i:i + L]:
                    height = (pixel - G_min) // h  # lowest box is at G_min and each is h gray levels tall
                    boxes[height].append(pixel)  # assign the pixel intensity to the correct box
            stddev = np.sqrt(np.var(boxes, axis=1))  # calculate the standard deviation of each box
            stddev = stddev[~np.isnan(stddev)]  # remove boxes with NaN standard deviations (empty)
            nBox_r = 2 * (stddev // h) + 1
            N_r += sum(nBox_r)
        if N_r != prev:  # check for plateauing
            r_Nr.append([r, N_r])
            prev = N_r
    x = np.array([np.log(1 / point[0]) for point in r_Nr])  # log(1/r)
    y = np.array([np.log(point[1]) for point in r_Nr])  # log(Nr)
    D = np.polyfit(x, y, 1)[0]  # D = lim r -> 0 log(Nr)/log(1/r)
    return D
