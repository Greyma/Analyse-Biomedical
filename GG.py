import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# global variables
input_shape  = (188, 188, 3)
output_shape = (100, 100, 1)
padding      = [200, 100]
# get the type of cells <rbc, wbc or plt>
cell_type = 'rbc'

output_directory = 'output/rcb'

os.makedirs(output_directory, exist_ok=True)
model_name = cell_type


def conv_bn(filters,
            model,
            kernel=(3, 3),
            activation='relu', 
            strides=(1, 1),
            padding='valid',
            type='normal'):
    '''
    This is a custom convolution function:
    :param filters --> number of filters for each convolution
    :param kernel --> the kernel size
    :param activation --> the general activation function (relu)
    :param strides --> number of strides
    :param padding --> model padding (can be valid or same)
    :param type --> to indicate if it is a transpose or normal convolution

    :return --> returns the output after the convolutions.
    '''
    if type == 'transpose':
        kernel = (2, 2)
        strides = 2
        conv = tf.keras.layers.Conv2DTranspose(filters, kernel, strides, padding)(model)
    else:
        conv = tf.keras.layers.Conv2D(filters, kernel, strides, padding)(model)

    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation(activation)(conv)

    return conv


def max_pool(input):
    '''
    This is a general max pool function with custom parameters.
    '''
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(input)


def concatenate(input1, input2, crop):
    '''
    This is a general concatenation function with custom parameters.
    '''
    return tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(crop)(input1), input2])


def do_unet():
    '''
    This is the dual output U-Net model.
    It is a custom U-Net with optimized number of layers.
    Please read model.summary()
    '''
    inputs = tf.keras.layers.Input((188, 188, 3))

    # encoder
    filters = 32
    encoder1 = conv_bn(3*filters, inputs)
    encoder1 = conv_bn(filters, encoder1, kernel=(1, 1))
    encoder1 = conv_bn(filters, encoder1)
    pool1 = max_pool(encoder1)

    filters *= 2
    encoder2 = conv_bn(filters, pool1)
    encoder2 = conv_bn(filters, encoder2)
    pool2 = max_pool(encoder2)

    filters *= 2
    encoder3 = conv_bn(filters, pool2)
    encoder3 = conv_bn(filters, encoder3)
    pool3 = max_pool(encoder3)

    filters *= 2
    encoder4 = conv_bn(filters, pool3)
    encoder4 = conv_bn(filters, encoder4)

    # decoder
    filters /= 2
    decoder1 = conv_bn(filters, encoder4, type='transpose')
    decoder1 = concatenate(encoder3, decoder1, 4)
    decoder1 = conv_bn(filters, decoder1)
    decoder1 = conv_bn(filters, decoder1)

    filters /= 2
    decoder2 = conv_bn(filters, decoder1, type='transpose')
    decoder2 = concatenate(encoder2, decoder2, 16)
    decoder2 = conv_bn(filters, decoder2)
    decoder2 = conv_bn(filters, decoder2)

    filters /= 2
    decoder3 = conv_bn(filters, decoder2, type='transpose')
    decoder3 = concatenate(encoder1, decoder3, 40)
    decoder3 = conv_bn(filters, decoder3)
    decoder3 = conv_bn(filters, decoder3)

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(decoder3)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(decoder3)
    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    
    model.compile(loss='mse',
                    loss_weights=[0.1, 0.9],
                    optimizer=opt,
                    metrics='accuracy')
    return model


def load_image_list(img_files):
    '''
    This is the load image list function, which loads an enumerate
    of images (param: img_files)
    :param img_files --> the input image files which we want to read

    :return imgs --> the images that we read
    '''
    imgs = []
    for image_file in img_files:
        imgs += [cv2.imread(image_file)]
    return imgs


def clahe_images(img_list):
    '''
    This is the clahe images function, which applies a clahe threshold
    the input image list.
    :param img_files --> the input image files which we want to read

    :return img_list --> the output images
    '''
    for i, img in enumerate(img_list):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img_list[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_list


def preprocess_data(imgs, padding=padding[1]):
    '''
    This is the preprocess data function, which adds a padding to 
    the input images, masks and edges if there are any.
    :param imgs --> the input list of images.
    :param padding --> the input padding which is going to be applied.

    :return imgs --> output images with added padding.
    '''
    imgs = [np.pad(img, ((padding, padding),
                         (padding, padding), (0, 0)), mode='constant') for img in imgs]
    return imgs


def load_data(img_list, padding=padding[1]):
    '''
    This is the load data function, which will handle image loading and preprocessing.
    :param img_list --> list of input images
    :param padding --> padding to be applied on preprocessing

    :return imgs --> the output preprocessed imgs.
    '''
    imgs = load_image_list(img_list)
    imgs = clahe_images(imgs)
    return preprocess_data(imgs, padding=padding)


def slice(imgs,
          padding=padding[1],
          input_size=input_shape[0],
          output_size=output_shape[0]):
    '''
    This is the slice function, which slices each image into image chips.
    :param imgs --> the input images
    :param padding --> the padding which will be applied to each image
    :param input_size --> the input shape
    :param output_size --> the output shape

    :return list tuple (list, list, list) --> the tuple list of output (image, mask and edge chips)
    '''
    img_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size):
            for y in np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
    return np.array(img_chips)


def normalize(img):
    '''
    Normalizes an image
    :param img --> an input image that we want normalized

    :return np.array --> an output image normalized (as a numpy array)
    '''
    return np.array((img - np.min(img)) / (np.max(img) - np.min(img)))


def get_sizes(img,
              padding=padding[1],
              input=input_shape[0],
              output=output_shape[0]):
    '''
    Get full image sizes (x, y) to rebuilt the full image output
    :param img --> an input image we want to get its dimensions
    :param padding --> the default padding used on the test dataset
    :param input --> the input shape of the image (param: img)
    :param output --> the output shape of the image (param: img)

    :return couple --> a couple which contains the image dimensions as in (x, y)
    '''
    offset = padding + (output / 2)
    return [(len(np.arange(offset, img[0].shape[0] - input / 2, output)), len(np.arange(offset, img[0].shape[1] - input / 2, output)))]


def reshape(img,
            size_x,
            size_y):
    '''
    Reshape the full image output using the original sizes (x, y)
    :param img --> an input image we want to reshape
    :param size_x --> the x axis (length) of the input image (param: img)
    :param size_y --> the y axis (length) of the input image (param: img)

    :return img (numpy array) --> the output image reshaped according to the provided dimensions (size_x, size_y)
    '''
    return img.reshape(size_x, size_y, output_shape[0], output_shape[0], 1)


def concat(imgs):
    '''
    Concatenate all the output image chips to rebuild the full image
    :param imgs --> the images that we want to concatenate

    :return full_image --> the concatenation of all the provided images (param: imgs)
    '''
    return cv2.vconcat([cv2.hconcat(im_list) for im_list in imgs[:,:,:,:]])


def denoise(img):
    '''
    Remove noise from an image
    :param img --> the input image that we want to denoise (remove the noise)

    :return image --> the denoised output image
    '''
    # read the image
    img = cv2.imread(img)
    # return the output denoised image
    return cv2.fastNlMeansDenoising(img, 23, 23, 7, 21)


def predict(img):

    test_img = sorted(glob.glob(img))

    # initializing the model
    model = do_unet()

    model.load_weights(f'./models/rbc.h5')

    # load the image (slice it into chips for prediction)
    img = load_data(test_img, padding=padding[0])

    img_chips = slice(
        img,
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0],
    )

    # segment all image chips
    output = model.predict(img_chips)


    new_mask_chips = np.array(output[0])
    new_edge_chips = np.array(output[1])

    # get image dimensions
    dimensions = [get_sizes(img)[0][0], get_sizes(img)[0][1]]

    # reshape chips arrays to be concatenated
    new_mask_chips = reshape(new_mask_chips, dimensions[0], dimensions[1])
    new_edge_chips = reshape(new_edge_chips, dimensions[0], dimensions[1])

    # get rid of none necessary dimension
    new_mask_chips = np.squeeze(new_mask_chips)
    new_edge_chips = np.squeeze(new_edge_chips)

    # concatenate chips into a single image (mask and edge)
    new_mask = concat(new_mask_chips)
    new_edge = concat(new_edge_chips)
    
    # create output directories if it does not exist
    if not os.path.exists('output/'):
        os.makedirs('output/')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.imsave(f'{output_directory}/mask.png', new_mask)
    plt.imsave(f'{output_directory}/edge.png', new_edge)
    return hough_transform(f'{output_directory}/edge.png')

def hough_transform(img):
    # apply hough circles

    # getting the input image
    image = cv2.imread(img)
    # convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=33, maxRadius=55, minRadius=28, param1=30, param2=20)
    output = img.copy()

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        # save the output image
        plt.imsave(f'{output_directory}/hough_transform.png',
                   np.hstack([img, output]))

    # show the hough_transform results
    return len(circles)