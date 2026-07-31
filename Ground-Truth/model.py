import os
import sys
from pathlib import Path

import numpy as np


# IMPORT PROJECT PREPROCESSING
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import data_preprocessing as dp


# LOAD + PREPROCESS ALL GROUND-TRUTH SCANS

ground_truth_root = "Ground-Truth/NPY-Ground-Truth"

scan_names = ["Scan1", "Scan2", "Scan3",]
dataset = []

for scan_name in scan_names:

    scan_folder = os.path.join(ground_truth_root, scan_name,)
    image_path = os.path.join(scan_folder, "channel_00___0_data.npy",)
    mask_path = os.path.join(scan_folder, "channel_00___0_data_mask.npy",)

    image = np.load(image_path)
    mask = np.load(mask_path)

    image_preprocessed, stages = dp.preprocess_afm(
        image,
        remove_plane=True,
        line_flatten=True,
        align_rows=True,
        remove_2d_background=False,
    )

    image_normalised = dp.robust_normalise(image_preprocessed)

    dataset.append({
        "name": scan_name,
        "image_raw": image,
        "image_preprocessed": image_preprocessed,
        "image_normalised": image_normalised,
        "mask": mask.astype(np.uint8),
        "stages": stages,
    })


# LEAVE-ONE-SCAN-OUT FOLDS

folds = [
    {
        "train_indices": [0, 1],
        "test_index": 2,
    },
    {
        "train_indices": [0, 2],
        "test_index": 1,
    },
    {
        "train_indices": [1, 2],
        "test_index": 0,
    },
]


# PATCH EXTRACTION

def extract_patches(image, mask, patch_size=128,):
    image_patches = []
    mask_patches = []

    height, width = image.shape

    for row in range(0, height, patch_size,):
        for col in range(0, width, patch_size,):

            image_patch = image[row:row + patch_size, col:col + patch_size]
            mask_patch = mask[row:row + patch_size, col:col + patch_size]

            if image_patch.shape != (patch_size, patch_size,):
                continue

            image_patches.append(image_patch)
            mask_patches.append(mask_patch)

    return (np.asarray(image_patches), np.asarray(mask_patches),)


# AUGMENTATION

def augment_dataset(X, Y):
    X_augmented = []
    Y_augmented = []

    for image, mask in zip(X, Y):
        variants = [
            (image, mask,),
            (np.rot90(image, 1), np.rot90(mask, 1),),
            (np.rot90(image, 2), np.rot90(mask, 2),),
            (np.rot90(image, 3), np.rot90(mask, 3),),
            (np.fliplr(image), np.fliplr(mask),),
            (np.flipud(image), np.flipud(mask),),
        ]

        for image_aug, mask_aug in variants:
            X_augmented.append(image_aug)
            Y_augmented.append(mask_aug)

    return (np.asarray(X_augmented, dtype=np.float32,), np.asarray(Y_augmented,dtype=np.float32,),)



# BUILD FIRST FOLD

fold = folds[0]
X_train = []
Y_train = []

for index in fold["train_indices"]:
    item = dataset[index]

    image_patches, mask_patches = extract_patches(
        item["image_normalised"],
        item["mask"],
        patch_size=128,
    )
    X_train.append(image_patches)
    Y_train.append(mask_patches)


X_train = np.concatenate(X_train, axis=0,)
Y_train = np.concatenate(Y_train, axis=0,)


# AUGMENT TRAINING DATA
X_train_aug, Y_train_aug = augment_dataset(X_train, Y_train,)



# ADD CNN CHANNEL DIMENSION
X_train_aug = X_train_aug[
    ...,
    np.newaxis
]

Y_train_aug = Y_train_aug[
    ...,
    np.newaxis
]



# KEEP TEST SCAN SEPARATE
test_item = dataset[fold["test_index"]]
X_test = test_item["image_normalised"]
Y_test = test_item["mask"]

# CHECK EVERYTHING
print("\nTraining scans:")
for index in fold["train_indices"]:
    print(dataset[index]["name"])

print("\nTest scan:", test_item["name"],)

print("\nBefore augmentation:")
print("X_train:", X_train.shape,)
print("Y_train:", Y_train.shape,)

print("\nAfter augmentation:")
print("X_train_aug:", X_train_aug.shape,)
print("Y_train_aug:", Y_train_aug.shape,)

print("\nTest:")
print("X_test:", X_test.shape,)
print("Y_test:", Y_test.shape,)





# CREATING CNN MODEL

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = conv_block(p2, 64)

    # Decoder
    u2 = layers.UpSampling2D(size=(2, 2))(c3)
    u2 = layers.Concatenate()([u2, c2])
    c4 = conv_block(u2, 32)

    u1 = layers.UpSampling2D(size=(2, 2))(c4)
    u1 = layers.Concatenate()([u1, c1])
    c5 = conv_block(u1, 16)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    return Model(inputs, outputs)


model = build_unet()
model.summary()


# LOSS + METRICS
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)

    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=combined_loss,
    metrics=[
        dice_coefficient,
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
        tf.keras.metrics.Precision(thresholds=0.5),
        tf.keras.metrics.Recall(thresholds=0.5),
    ],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )
]

history = model.fit(
    X_train_aug,
    Y_train_aug,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=callbacks,
)

model.save("Ground-Truth/qd_unet_fold1.keras")
print("\nTraining finished. Predicting Scan3...")