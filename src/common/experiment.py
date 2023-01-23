import random
import random as rand

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from utils.bounding_box_iou import bounding_box_intersection_over_union

def print_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def print_results(vit_object_detector, x_test, y_test, box_reader):
    import matplotlib.patches as patches

    i, mean_iou = 0, 0
    N = 10
    for i in range(0, N):
        input_image = x_test[i]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))
        im = input_image

        # Display the image
        ax1.imshow(im, cmap=plt.cm.gray)
        ax2.imshow(im, cmap=plt.cm.gray)

        input_image = np.expand_dims(input_image, axis=0)

        p = vit_object_detector.predict(input_image)

        box_predicted = box_reader(p[0])
        box_truth = box_reader(y_test[i])
        print("--------------------------")
        print(box_predicted)
        print(box_truth)
        print("--------------------------")
        ax3.imshow(im[int(box_truth[0]):int(box_truth[1]), int(box_truth[2]):int(box_truth[3])], cmap=plt.cm.gray)
        ax4.imshow(im[int(box_predicted[0]):int(box_predicted[1]), int(box_predicted[2]):int(box_predicted[3])], cmap=plt.cm.gray)

        # Create the bounding box
        rect = patches.Rectangle(
            (box_predicted[2], box_predicted[0]),
            box_predicted[3] - box_predicted[2],
            box_predicted[1] - box_predicted[0],
            facecolor="none",
            edgecolor="blue",
            linewidth=2,
        )
        # Add the bounding box to the image
        ax1.add_patch(rect)
        ax1.set_xlabel(
            "Predicted: "
            + str(box_predicted[2])
            + ", "
            + str(box_predicted[0])
            + ", "
            + str(box_predicted[3])
            + ", "
            + str(box_predicted[1])
        )

        mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)
        # Create the bounding box
        # y0 y1 x0 x1
        # 0  1  2  3
        rect = patches.Rectangle(
            (box_truth[2], box_truth[0]),
            box_truth[3] - box_truth[2],
            box_truth[1] - box_truth[0],
            facecolor="none",
            edgecolor="green",
            linewidth=2,
        )
        # Add the bounding box to the image
        ax2.add_patch(rect)

        ax2.set_xlabel(
            "Target: "
            + str(box_truth[2])
            + ", "
            + str(box_truth[0])
            + ", "
            + str(box_truth[3])
            + ", "
            + str(box_truth[1])
            + "\n"
            + "IoU"
            + str(bounding_box_intersection_over_union(box_predicted, box_truth))
        )
        # i = i + 1

    print("mean_iou: " + str(mean_iou / N))
    plt.show()


def print_results_resized(image_size, vit_object_detector, x_test, y_test, box_reader):
    import matplotlib.patches as patches

    i, mean_iou = 0, 0
    N = 10
    for i in range(0, N):
        input_image = x_test[i]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))
        im = input_image

        # Display the image
        ax1.imshow(im, cmap=plt.cm.gray)
        ax2.imshow(im, cmap=plt.cm.gray)

        input_image = np.expand_dims(input_image, axis=0)

        p = vit_object_detector.predict(input_image)

        box_predicted = box_reader(p[0])
        box_truth = box_reader(y_test[i])
        box_predicted = (box_predicted[0], box_predicted[1], box_predicted[2], box_predicted[3])
        box_truth = (box_truth[0], box_truth[1], box_truth[2], box_truth[3])

        print("--------------------------")
        print(box_predicted)
        print(box_truth)
        print("--------------------------")
        #(y0 / h, y1 / h, x0 / w, x1 / w)
        #ax3.imshow(im[int(box_truth[0]):int(box_truth[1]), int(box_truth[2]):int(box_truth[3])], cmap=plt.cm.gray)
        #ax4.imshow(im[int(box_predicted[0]):int(box_predicted[1]), int(box_predicted[2]):int(box_predicted[3])], cmap=plt.cm.gray)

        # Create the bounding box
        rect = patches.Rectangle(
            (int(box_predicted[2]), int(box_predicted[0])),
            int(box_predicted[3] - box_predicted[2]),
            int(box_predicted[1] - box_predicted[0]),
            facecolor="none",
            edgecolor="blue",
            linewidth=2,
        )
        # Add the bounding box to the image
        ax1.add_patch(rect)
        ax1.set_xlabel(
            "Predicted: "
            + str(int(box_predicted[2]))
            + ", "
            + str(int(box_predicted[0]))
            + ", "
            + str(int(box_predicted[3]))
            + ", "
            + str(int(box_predicted[1]))
        )

        mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)
        # Create the bounding box
        # y0 y1 x0 x1
        # 0  1  2  3
        rect = patches.Rectangle(
            (int(box_truth[2]), int(box_truth[0])),
            int(box_truth[3] - box_truth[2]),
            int(box_truth[1] - box_truth[0]),
            facecolor="none",
            edgecolor="green",
            linewidth=2,
        )
        # Add the bounding box to the image
        ax2.add_patch(rect)

        ax2.set_xlabel(
            "Target: "
            + str(int(box_truth[2]))
            + ", "
            + str(int(box_truth[0]))
            + ", "
            + str(int(box_truth[3]))
            + ", "
            + str(int(box_truth[1]))
            + "\n"
            + "IoU"
            + str(bounding_box_intersection_over_union(box_predicted, box_truth))
        )
        # i = i + 1

    print("mean_iou: " + str(mean_iou / N))
    plt.show()

def print_results_classification(vit_object_detector, x_test, y_test):

    N = 10
    for i in range(0, N):
        input_image = x_test[i]
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
        im = input_image

        # Display the image
        ax1.imshow(im, cmap=plt.cm.gray)

        input_image = np.expand_dims(input_image, axis=0)
        p = vit_object_detector.predict(input_image)

        predicted = p[0]
        truth = y_test[i]

        ax1.set_xlabel(
            "Predicted: " + str(predicted) + ", Original: " + str(truth)
        )


    plt.show()


def plot_chart():
    print('Plot chart')


def run_experiment_sgd(model, learning_rate,  batch_size, num_epochs, x_train, y_train):

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=learning_rate, nesterov=False, name="SGD"
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    checkpoint_filepath = "../logs/"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )


    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback
        ],
    )

    return history

def run_experiment_cce(model, learning_rate, weight_decay,  batch_size, num_epochs, x_train, y_train):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_filepath = "../logs/"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )


    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback
        ],
    )

    return history

def run_experiment_cce_bin(model, learning_rate, weight_decay,  batch_size, num_epochs, x_train, y_train):

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.001, nesterov=False, name="SGD"
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_filepath = "../logs/"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )


    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback
        ],
    )

    return history

def run_experiment_adamw(model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics=['accuracy'])

    checkpoint_filepath = "../logs/"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback
        ],
    )

    return history


def shuffle_data(images, annotations):
    all = list(zip(images, annotations))
    random.shuffle(all)
    res1, res2 = zip(*all)
    images, annotations = list(res1), list(res2)
    return images, annotations
