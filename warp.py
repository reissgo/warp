import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def print_array_info(v):
    print("{} is of type {} with shape {} dtype {}".format(v,
                                                           eval("type({})".format(v)),
                                                           eval("{}.shape".format(v)),
                                                           eval("{}.dtype".format(v))
                                                           ))

def show_samples(array_of_images):
    n = array_of_images.shape[0]
    total_rows = 1+int((n-1)/5)
    total_columns = 5
    fig = plt.figure()
    gridspec_array = fig.add_gridspec(total_rows, total_columns)

    for i, img in enumerate(array_of_images):
        row = int(i/5)
        col = i % 5
        ax = fig.add_subplot(gridspec_array[row, col])
        ax.imshow(img)
        ax.set_title("i={:d}".format(i))

    plt.show()


numpics = 10

piclist = []
for i in range(numpics):
    piclist.append(plt.imread('pic_{:02d}.jpg'.format(i)))

x_train = np.array(piclist)
y_train = np.array([[0]]*numpics)

'''
cifar_data = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar_data.load_data()
'''
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90,zoom_range=0.5)

print_array_info("x_train")
show_samples(x_train)

batches = 0
batch_size=5
print_array_info("y_train")
for x_batch, y_batch in data_generator.flow(x_train, y_train, batch_size=batch_size):
    print_array_info("x_batch")

    batches += 1
    #if batches >= len(x_train) / batch_size:
        #break
    show_samples(x_batch[:batch_size]/255)

print("batches done {:d} batch_size = {:d} so total images = {:d}".format(batches,  batch_size, batches * batch_size))
