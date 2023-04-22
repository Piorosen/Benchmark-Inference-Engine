import numpy
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader

def _preprocess_images(height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    # image_names = os.listdir(images_folder)
    # if size_limit > 0 and len(image_names) >= size_limit:
    #     batch_filenames = [image_names[i] for i in range(size_limit)]
    # else:
    #     batch_filenames = image_names
    unconcatenated_batch_data = []

    for i in range(5):
        # image_filepath = images_folder + "/" + image_name
        input_data = numpy.random.rand(1, 3, width, height).astype(numpy.float32)
        # nhwc_data = numpy.expand_dims(input_data, axis=0)
        # nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(input_data)
        batch_data = numpy.concatenate(
            numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None