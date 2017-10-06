import numpy as np
from kmeans import Kmeans

class ImageQuantizer:

    def __init__(self, b):
        self.b = b

    def quantize(self, image):
        height = image.shape[0]
        width = image.shape[1]

        flat_image = image.reshape((-1,3))
        model = Kmeans(2**self.b)
        model.fit(flat_image)
        quantize_image = model.predict(flat_image).reshape((height, width, 1))

        return quantize_image, model.means

    def dequantize(self, quantize_image, mean):
        height = quantize_image.shape[0]
        width = quantize_image.shape[1]
        image_area = height*width
        new_image = np.zeros((image_area, 3))

        y_pred = quantize_image.flatten()

        for i in range(image_area):
            new_image[i] = mean[y_pred[i]]

        return new_image.reshape((height, width, 3))