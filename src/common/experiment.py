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


def print_results(vit_object_detector, x_test, y_test, save=True, box_reader=lambda bbox: (bbox[0], bbox[1], bbox[2], bbox[3])):
    import matplotlib.patches as patches

    # Saves the model in current path
    if (save):
        vit_object_detector.save("vit_object_detector.h5", save_format="h5")

    i, mean_iou = 0, 0
    N = 10
    # sample = random.sample(x_test, 10)
    # Compare results for 10 images in the test set
    for i in rand.sample(range(0, len(x_test) - 1), N):
        input_image = x_test[i]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
        im = input_image

        # Display the image
        ax1.imshow(im, cmap=plt.cm.gray)
        ax2.imshow(im, cmap=plt.cm.gray)

        input_image = np.expand_dims(input_image, axis=0)
        p = vit_object_detector.predict(input_image)
        preds = p[0]

        #top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)

        #bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

        # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

        box_predicted = preds
        box_truth = y_test[i]
        print(box_truth, box_predicted, box_reader)
        ax3.imshow(im[int(box_truth[0]):int(box_truth[1]), int(box_truth[2]):int(box_truth[3])], cmap=plt.cm.gray)

        # print(box_predicted)

        # Create the bounding box
        rect = patches.Rectangle(
            (box_predicted[0], box_predicted[1]),
            box_predicted[2] - box_predicted[0],
            box_predicted[3] - box_predicted[1],
            facecolor="none",
            edgecolor="red",
            linewidth=1,
        )
        # Add the bounding box to the image
        ax1.add_patch(rect)
        ax1.set_xlabel(
            "Predicted: "
            + str(box_predicted[0])
            + ", "
            + str(box_predicted[1])
            + ", "
            + str(box_predicted[2])
            + ", "
            + str(box_predicted[3])
        )

        #top_left_x, top_left_y = int(y_test[i][0] * w), int(y_test[i][1] * h)

        #bottom_right_x, bottom_right_y = int(y_test[i][2] * w), int(y_test[i][3] * h)

        # top_left_x, top_left_y, bottom_right_x, bottom_right_y



        mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)
        # Create the bounding box
        rect = patches.Rectangle(
            (box_truth[0], box_truth[1]),
            box_truth[2] - box_truth[0],
            box_truth[3] - box_truth[1],
            facecolor="none",
            edgecolor="red",
            linewidth=1,
        )
        # Add the bounding box to the image
        ax2.add_patch(rect)
        ax2.set_xlabel(
            "Target: "
            + str(box_truth[0])
            + ", "
            + str(box_truth[1])
            + ", "
            + str(box_truth[2])
            + ", "
            + str(box_truth[3])
            + "\n"
            + "IoU"
            + str(bounding_box_intersection_over_union(box_predicted, box_truth))
        )
        # i = i + 1

    print("mean_iou: " + str(mean_iou / N))
    plt.show()


def plot_chart():
    print('Plot chart')


def run_experiment_sgd(model, learning_rate, batch_size, num_epochs, x_train, y_train):

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=learning_rate, nesterov=False, name="SGD"
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

def run_experiment_adamw(model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
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