from PIL import Image
import numpy as np
import random
import os
import re


# Convert image to matrix
def imageToMatrix(image, threshold=145, size=(100, 100)):
    # Convert image to greyscale
    new_image = Image.open(image).convert(mode='L')
    # Resize image and convert it to array
    image_array = np.asarray(new_image.resize(size))
    result = np.zeros(image_array.shape)
    # pixel <= threshold: -1
    result[image_array <= threshold] = -1
    # pixel > threshold: +1
    result[image_array > threshold] = 1
    return result


# Convert matrix to image
def matrixToImage(arr, output_path=None):
    pixels = np.zeros(arr.shape, dtype=np.uint8)
    # data = 1: pixel = 255
    pixels[arr == 1] = 255
    # data = -1: pixel = 0
    pixels[arr == -1] = 0
    image = Image.fromarray(pixels, mode='L')
    # save image file
    if output_path:
        image.save(output_path)
    return image


# Convert matrix to vector
def matrixToVector(matrix):
    final_vector = []
    row_num = matrix.shape[0]
    column_num = matrix.shape[1]
    for i in range(row_num):
        for j in range(column_num):
            final_vector.append(matrix[i, j])
    return np.array(final_vector)


# Initializing training set image's weight
def initializeImageWeight(vector):
    # Hopfield connection:
    # W[i][j] = W[j][i]: connections are symmetric
    # W[i][i] = 0: no unit has a connection with itself
    size = len(vector)
    weight_matrix = np.zeros([size, size])
    for i in range(size):
        for j in range(i, size):
            if i == j:
                weight_matrix[i][j] = 0
            else:
                weight_matrix[i][j] = vector[i] * vector[j]
                weight_matrix[j][i] = weight_matrix[i][j]
    return weight_matrix


# Update weights
def update(weight, vector, iteration=100):
    for _ in range(iteration):
        size = len(vector)
        # Select a random unit (Asynchronous)
        selected_unit = random.randint(0, size - 1)
        # h = W[i][1] * X[1] + W[i][2] * X[2] + W[i][3] * X[3] + ...
        h = np.dot(weight[selected_unit][:], vector)
        # update vector
        vector[selected_unit] = np.where(h > 0, 1, -1)
    return vector


# save or show result
def output(vector, path, counter):
    if path:
        image_output = path + '/result images/' + str(counter+1) + '.jpg'
        matrixToImage(vector, output_path=image_output)
    else:
        result_image = matrixToImage(vector)
        result_image.show()


def hopfield(train_images_path, test_images_path, iteration=1000, threshold=60, size=(100, 100), current_path=None):
    print('Start learning training images...')
    weights = []
    flag = False
    for image in train_images_path:
        # Convert image to pixel matrix
        train_image_matrix = imageToMatrix(image=image, threshold=threshold, size=size)
        # Convert pixel matrix to vector
        train_image_vector = matrixToVector(train_image_matrix)
        # create weight for first image
        if not flag:
            weights = initializeImageWeight(train_image_vector)
            flag = True
        # Incremental learning rule
        else:
            new_weight = initializeImageWeight(train_image_vector)
            weights += new_weight
    print('Learning is complete.')
    print('Weights are ready!')

    print('Processing test set...')
    counter = 0
    for image in test_images_path:
        # Convert image to matrix
        test_image_matrix = imageToMatrix(image=image, threshold=threshold, size=size)
        # save matrix shape for rebuild image
        image_matrix_dimension = test_image_matrix.shape
        # Convert matrix to image
        print('imported {} test image'.format(counter))

        # Convert matrix to vector
        test_image_vector = matrixToVector(test_image_matrix)
        print('Updating...')
        updated_vector = update(weight=weights, vector=test_image_vector, iteration=iteration)
        # Convert vector to matrix
        updated_vector = updated_vector.reshape(image_matrix_dimension)
        output(vector=updated_vector, path=current_path, counter=counter)
        counter += 1


if __name__ == "__main__":
    current_path = os.getcwd()
    train_path = current_path + "/training images/"
    test_path = current_path + "/test images/"

    # Get 'jpg' file and append to training_path and test_path
    train_images = [train_path + image for image in os.listdir(train_path) if re.match(r'[0-9a-zA-Z]*.jpg', image)]
    test_images = [test_path + image for image in os.listdir(test_path) if re.match(r'[0-9a-zA-Z]*.jpg', image)]
    # Hopfield Network
    hopfield(
        train_images_path=train_images,
        test_images_path=test_images,
        iteration=50000,
        threshold=60,
        size=(100, 100),
        current_path=current_path)
