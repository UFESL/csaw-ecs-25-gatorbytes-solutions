import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging ; absl.logging.set_verbosity(absl.logging.ERROR)
from random import shuffle
from functools import partial
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import TensorSpec, convert_to_tensor, expand_dims, float32, int8 
from tensorflow.data import AUTOTUNE, Dataset 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, AveragePooling1D, BatchNormalization, \
                                    Conv1D, Dense, Flatten, Input, MaxPooling1D, add
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
print(f"Tensorflow version {tf.__version__} with Keras version {tf.keras.__version__}")

import numpy as np
import argparse


SHIFTS = [2, 7, 9, 13]

def ROTL(a, b):
    if b > 16: 
        return a & 0xFFFF
    if a >= (1 << (16 - b)):
        return (((a << b) & 0xFFFF) | (a >> (16 - b))) & 0xFFFF
    else:
        return ((a << b) & 0xFFFF)

def QR_CAL(a, b, c, d):
    # a,b,c,d int16 values
    b ^= ROTL((a + d) & 0xFFFF, SHIFTS[0])
    c ^= ROTL((b + a) & 0xFFFF, SHIFTS[1])
    d ^= ROTL((c + b) & 0xFFFF, SHIFTS[2])
    a ^= ROTL((d + c) & 0xFFFF, SHIFTS[3])
    return a & 0xFFFF, b & 0xFFFF, c & 0xFFFF, d & 0xFFFF



def generate_data(dataset_path, mode, start_sample, end_sample, qr_round):
    data = torch.load(dataset_path)  # dict with 'traces', 'passwords'
    
    traces = data["traces"]
    passwords = data["passwords"]

    if isinstance(passwords, torch.Tensor):  
        passwords = passwords.cpu().numpy() 

    if isinstance(traces, torch.Tensor):    
        traces = traces.cpu().numpy().astype(np.float32) 

    print(f"Loaded {len(traces)} traces with samples {traces.shape[1]}")

    N = len(traces)

    pw = passwords.astype(np.uint16)
    k_lo = pw[:, 0::2]
    k_hi = pw[:, 1::2]
    keys16 = (k_lo | (k_hi << 8)).astype(np.uint16)   # shape (N, 8), uint16
    
    qr_bytes = np.zeros((N, 8), dtype=np.uint8)
    
    for i in range(N):
        key = keys16[i]  # 8 words: key[0..7]
        
        a1 = 0x4554
        b1 = int(key[3])
        c1 = 0xd9a6
        d1 = int(key[5])
        
        a2 = 0x4332
        b2 = 0xb8f7
        c2 = int(key[6])
        d2 = int(key[0])
        
        a3 = 0x3032
        b3 = int(key[7])
        c3 = int(key[1])
        d3 = 0xeaee
        
        a4 = 0x3520
        b4 = int(key[2])
        c4 = 0x83a0
        d4 = int(key[4])

        # a_1, b_1, c_1, d_1 = QR_CAL(a1, b1, c1, d1)
        # a_2, b_2, c_2, d_2 = QR_CAL(a2, b2, c2, d2)
        # a_3, b_3, c_3, d_3 = QR_CAL(a3, b3, c3, d3)
        # a_4, b_4, c_4, d_4 = QR_CAL(a4, b4, c4, d4)  
        
        b_1 =  ROTL((a1 + d1) & 0xFFFF, SHIFTS[0]) & 0xFFFF
        c_1 =  ROTL((b1 + a1) & 0xFFFF, SHIFTS[1]) & 0xFFFF
        d_1 =  ROTL((c1 + b1) & 0xFFFF, SHIFTS[2]) & 0xFFFF
        a_1 =  ROTL((d1 + c1) & 0xFFFF, SHIFTS[3]) & 0xFFFF
                    
        b_2 =  ROTL((a2 + d2) & 0xFFFF, SHIFTS[0]) & 0xFFFF
        c_2 =  ROTL((b2 + a2) & 0xFFFF, SHIFTS[1]) & 0xFFFF
        d_2 =  ROTL((c2 + b2) & 0xFFFF, SHIFTS[2]) & 0xFFFF
        a_2 =  ROTL((d2 + c2) & 0xFFFF, SHIFTS[3]) & 0xFFFF
        
        b_3 =  ROTL((a3 + d3) & 0xFFFF, SHIFTS[0]) & 0xFFFF
        c_3 =  ROTL((b3 + a3) & 0xFFFF, SHIFTS[1]) & 0xFFFF
        d_3 =  ROTL((c3 + b3) & 0xFFFF, SHIFTS[2]) & 0xFFFF
        a_3 =  ROTL((d3 + c3) & 0xFFFF, SHIFTS[3]) & 0xFFFF
                    
        b_4 =  ROTL((a4 + d4) & 0xFFFF, SHIFTS[0]) & 0xFFFF
        c_4 =  ROTL((b4 + a4) & 0xFFFF, SHIFTS[1]) & 0xFFFF
        d_4 =  ROTL((c4 + b4) & 0xFFFF, SHIFTS[2]) & 0xFFFF
        a_4 =  ROTL((d4 + c4) & 0xFFFF, SHIFTS[3]) & 0xFFFF

        # order requested: b[0], c[0], d[0], a[0]
        out_words_1 = (b_1, c_1, d_1, a_1)
        out_words_2 = (b_2, c_2, d_2, a_2)
        out_words_3 = (b_3, c_3, d_3, a_3)
        out_words_4 = (b_4, c_4, d_4, a_4)
        
        if qr_round == 1:
            out_words = out_words_1
        elif qr_round == 2:
            out_words = out_words_2
        elif qr_round == 3:
            out_words = out_words_3
        elif qr_round == 4:
            out_words = out_words_4
        else:
            out_words = out_words_1
        
        bytes_temp = []
        for w in out_words:
            bytes_temp.extend([(w >> 8) & 0xFF, w & 0xFF])
        qr_bytes[i, :] = bytes_temp  # 8 bytes

    # one-hot 256 for each of the 8 bytes
    targets = [to_categorical(qr_bytes[:, j], num_classes=256).astype(np.float32) for j in range(8)]

    # 80% train, 20% test
    split = int(N * 0.8)
    if mode == "train":
        indices = list(range(split))
    else:
        indices = list(range(split, N))
    shuffle(indices)

    for idx in indices:
        x = convert_to_tensor(traces[idx][start_sample:end_sample], dtype=tf.float32)   
        x = tf.reshape(x, (-1, 1))                         
        y = tuple([convert_to_tensor(targets[i][idx], dtype=tf.float32) for i in range(8)])
        yield x, y
        
# Resnet layer sub-function
def resnet_layer(inputs, num_filters=16, k_size=11, strides=1,
                 activation='relu', batch_norm=True):
    x = inputs
    x = Conv1D(num_filters, kernel_size=k_size, strides=strides, padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


# Output block sub-function for 16 byte outputs (256 classes each)
def output_block(inputs, name, width):
    x = inputs
    x = Dense(1024, activation='relu', name=f'fc1_{name}')(x)
    x = BatchNormalization()(x)

    # name -> f"{name}_output" stays, but now name = "byte_XX"
    x = Dense(width, activation="softmax", name=f'{name}_output')(x)
    return x

def build_model(args):
    num_filters = 16; strides = 1
    
    num_samples = args.end - args.start

    # ✅ variable-length traces instead of fixed 1500
    inputs = Input(shape=(num_samples, 1))

    x = resnet_layer(inputs=inputs)
    for stack in range(9):
        if stack > 0: 
            strides = 2

        y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
        y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)

        if stack > 0:
            x = resnet_layer(inputs=x, num_filters=num_filters, k_size=1,
                             strides=strides, activation=None, batch_norm=False)

        x = add([x, y])
        x = Activation('relu')(x)

        if num_filters < 256: 
            num_filters *= 2

    x = AveragePooling1D(pool_size=4)(x)
    x = Flatten()(x)

    outputs = [output_block(x, f"byte_{i:02d}", 256) for i in range(8)]

    return Model(inputs, outputs, name='ResNetPwBytes')


# Plot the training history
def plot_history(hist, args):
    plt.figure(figsize=(14, 8))
    epochs = range(len(hist.history["loss"]))

    # Total loss
    plt.subplot(2, 1, 1)
    plt.title("Total Loss")
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.legend(["Train Loss", "Val Loss"])
    # plt.ylim(80, 100)
    plt.xlabel("Epoch")
    plt.xticks(epochs)

    # Byte accuracies
    plt.subplot(2, 1, 2)
    plt.title("Byte Accuracy (val)")
    for i in range(8):
        plt.plot(hist.history[f"val_byte_{i:02d}_output_accuracy"])
    plt.xlabel("Epoch")
    plt.xticks(epochs)
    plt.legend([f"byte_{i:02d}" for i in range(8)], ncol=4, fontsize=8)

    plt.tight_layout()
    plt.show()
    
    plt.savefig(args.save_res + str(args.start) + '_' + str(args.end) + '_' + str(args.qr_round) + '_' + "final.png", dpi=300, bbox_inches='tight')



# Build and train the model
def train_model(train_data, valid_data, args):
    model = build_model(args)
    model.compile(
        optimizer=RMSprop(1e-3),
        loss=["categorical_crossentropy"] * 8,   # still one loss per byte
        metrics=["accuracy"] * 8                 # still accuracy metric per head
    )

    callbacks = [ReduceLROnPlateau(factor=0.25, patience=2, verbose=0)]
    callbacks.append(ModelCheckpoint(
        args.save_res + "ghostbloodmodel_checkpoint_" + str(args.start) + '_' + str(args.end) + '_' + str(args.qr_round) + ".weights.h5",
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    ))

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )
    return history, model

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare sliced traces and QR-round labels for first 4 rounds.")
    parser.add_argument("--data", required=True, help="Path to torch dataset (.pt/.pth) with 'traces', 'passwords' (and optional 'nonces').")
    parser.add_argument("--start", type=int, default=None, help="Starting sample index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Ending sample index (exclusive).")
    parser.add_argument("--qr_round", type=int, default=1, choices=[1,2,3,4], help="Target quarter-round in the first 4.")
    parser.add_argument("--save_res", type=str, default=None, help="If set, save results")
    args = parser.parse_args()
    
    print(f"[INFO] data        = {args.data}")
    print(f"[INFO] start       = {args.start}")
    print(f"[INFO] end         = {args.end}")
    print(f"[INFO] qr_round    = {args.qr_round}")
    print(f"[INFO] save_res    = {args.save_res}")
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_data = Dataset.from_generator(
        partial(generate_data, args.data, "train", args.start, args.end, args.qr_round),
        output_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),                     
            tuple([tf.TensorSpec(shape=(256,), dtype=tf.float32) for _ in range(8)])  # CHANGED
        )
    ).cache().batch(64).prefetch(AUTOTUNE)

    valid_data = Dataset.from_generator(
        partial(generate_data, args.data, "valid", args.start, args.end, args.qr_round),
        output_signature=(
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32),                     
            tuple([tf.TensorSpec(shape=(256,), dtype=tf.float32) for _ in range(8)])  # CHANGED
        )
    ).cache().batch(64).prefetch(AUTOTUNE)


    history, model = train_model(train_data, valid_data, args)
    print(f"{np.sum([np.prod(v.shape) for v in model.trainable_variables])} total trainable parameters")
    
    import json

    with open(args.save_res + str(args.start) + '_' + str(args.end) + '_' + str(args.qr_round) + '_' + "history.json", "w") as f:
        json.dump(history.history, f)
    
    plot_history(history, args)


    

if __name__ == "__main__":
    main()

    