"""
Program for training/loading convolutional and recurrent neural network models
for classifying modulation types for IQ samples in the RadioML 2016.10A data set.
Contains 5 CNN models and 2 RNN (LSTM) models.

Created by Magnus Kristoffersen and Markus Bjørklund.
"""

import pickle
import numpy as np
import json 
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams.update({'font.size': 22})


# Load the dataset
file_loc = Path(__file__).parent.absolute()
data_path = (
    file_loc.parent / "data" / "RADIOML_2016.10A" / "RML2016.10a_dict_unix.pkl"
)  # Path to data set. Dos2Unix tool used on .pkl file beforehand for CRLF -> LF EOL encoding

# Unpickling data
print("Unpickling data...")
Xd = pickle.load(open(data_path, "rb"), encoding="latin1")
print("Done with unpickling")

# Stacking into list of modulation labels and SNR, and data.
# The following block of code is copied from:
# https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
            
X = np.vstack(X)
# The RNNs expect a 3D tensor input
X_rnn = np.transpose(X, (0, 2, 1))


# Labels (lbl) is a list of dimensions (n_samples, 2), where each sample entry is (Modulation type, SNR value)
# X is an array of dimensions (220000, 2, 128), where each sample entry is (I / Q, time sample)

# Plotting a sample with the highest SNR value for demonstration:
while True:
    i = np.random.randint(0, len(X))
    if str(lbl[i][1]) == str(18):
        fig, (ax1, ax2) = plt.subplots(nrows=2, tight_layout=True)
        plot_vec = np.linspace(0, 127, 128)
        ax1.plot(plot_vec, X[i, 0, :], label="I", color="blue", linewidth=2)
        ax2.plot(plot_vec, X[i, 1, :], label="Q", color="orange", linewidth=2)
        fig.suptitle("{}, SNR={}dB".format(lbl[i][0], lbl[i][1]), fontsize=32)
        ax2.set_xlabel("Time (arbitrary units)", fontsize=22)
        ax1.legend()
        ax2.legend()

        plt.show()
        plt.close()
        break
    


# Partition the data and corresponding labels into training, validation and test sets (60-20-20 split)
train_idx, temp_idx = train_test_split(np.arange(X.shape[0]), test_size=0.4, random_state=2023)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=12)

X = X.reshape((len(X), 2, 128, 1))
X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

# The RNN models expect a 3D tensor input
X_train_rnn = X_rnn[train_idx]
X_val_rnn = X_rnn[val_idx]
X_test_rnn = X_rnn[test_idx]


# Convert labels to one-hot vectors
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


train_ = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
val_ = list(map(lambda x: mods.index(lbl[x][0]), val_idx))
test_ = list(map(lambda x: mods.index(lbl[x][0]), test_idx))
test_snr = list(map(lambda x: snrs.index(lbl[x][1]), test_idx)) 

y_train = to_onehot(train_)
y_val = to_onehot(val_)
y_test = to_onehot(test_)


nb_epochs = 200  # number of epochs used for training
batch_size = 1024  # training batch size


"""
Creating our neural network models.
Using 50% dropout in all layers except last layer.

Training all models with ADAM as optimizer, categorical cross-entropy as cost function, 
and accuracy as performance metric.
"""
# Build a convolutional neural network with a 1D and a 2D convolution layer, 
# and a single fully connected layer with softmax activation for determining the classification
single_layer_conv_64 = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(2, 128, 1)),
    Dropout(0.5),
    Conv2D(64, kernel_size=(2, 4), activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(11, activation='softmax')
])

single_layer_conv_64.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Build a convolutional neural network with a 1D and a 2D convolution layer, 
# and finally two fully connected layers
two_layer_conv_64 = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(2, 128, 1)),
    Dropout(0.5),
    Conv2D(64, kernel_size=(2, 4), activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax')
])

two_layer_conv_64.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Build a convolutional neural network similar to the one used in original article.
two_layer_conv_origin = Sequential([
    Conv1D(256, kernel_size=3, padding='same', activation='relu', input_shape=(2, 128, 1)),
    Dropout(0.5),
    Conv2D(80, kernel_size=(2, 3), activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax')
])

two_layer_conv_origin.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Build a convolutional neural network with 3 parallel 1D convolutional layers 
# with different kernels sizes, a 2D convolution layer, and finally a fully connected layer
def create_parallel_conv_model(kernel_sizes: list, num_fc_layers: int = 1, conv_filters_1: int = 64, conv_filters_2: int = 64, fc_filters: int = 64):
    input_shape = Input(shape=(2, 128, 1))
    parallel = [0] * len(kernel_sizes)
    for i in range(len(kernel_sizes)):
        parallel[i] = Conv1D(conv_filters_1, kernel_sizes[i], padding='same', activation='relu')(input_shape)
        parallel[i] = Dropout(0.5)(parallel[i])
    
    conc = Concatenate()(parallel)
    
    conv2 = Conv2D(conv_filters_2, kernel_size=(2, 4), padding='same', activation='relu')(conc)
    conv2 = Dropout(0.5)(conv2)
    
    out = Flatten()(conv2)

    for i in range(num_fc_layers - 1):
        out = Dense(fc_filters, activation='relu')(out)
        out = Dropout(0.5)(out)

    out = Dense(11, activation='softmax')(out)

    model = Model(input_shape, out)
    return model


kernels = [3, 5, 8]
parallel_conv_single = create_parallel_conv_model(kernels, num_fc_layers=1, conv_filters_2=80)

parallel_conv_single.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Build a convolutional neural network as above, but with an additional fully connected layer
kernels = [3, 5, 8]
parallel_conv_two = create_parallel_conv_model(kernels, num_fc_layers=2, fc_filters=256, conv_filters_2=80)

parallel_conv_two.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Build a recurrent neural network using an LSTM layer and two fully connected layers
lstm_rnn = Sequential([
    LSTM(64, activation='tanh', input_shape=(128, 2), dropout=0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax'),
])

lstm_rnn.compile(optimizer=Adam(clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])


# Build a recurrent neural network using two LSTMs and two fully connected layers

two_lstm_rnn = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(128, 2), dropout=0.5),
    LSTM(64, activation='tanh', return_sequences=False, dropout=0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax'),
])

# Use gradient norm clipping to handle exploding gradient problems.
two_lstm_rnn.compile(optimizer=Adam(clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])



"""Train the network"""

"""Convolutional network with single FC layer"""

single_layer_weights_path = (
    file_loc.parent / "weights" / "single_fc_layer_CNN_model_weights" / "single_fc_layer_CNN_model_weights"
)
loss_path_single_layer = (
    file_loc.parent / "results" / "Single_FC_layer" / "single_fc_layer_CNN_model_history.json"
)

# Loading the saved weights from file.
single_layer_conv_64.load_weights(single_layer_weights_path)

# Remove """ on both sides of this block to keep training the model.
"""history_single_layer_conv = single_layer_conv_64.fit(X_train, y=y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=nb_epochs)

# Save the weights and loss history
single_layer_conv_64.save_weights(single_layer_weights_path)
json.dump(history_single_layer_conv.history, open(loss_path_single_layer, 'w+'))

single_layer_conv_64.evaluate(X_test, y_test, batch_size=batch_size)"""

# Predict classification labels for the test set. Highest output probability is used as prediction.
y_pred_single_layer_conv = single_layer_conv_64.predict(X_test, batch_size)
y_pred_single_layer_conv = np.array([np.argmax(pred) for pred in y_pred_single_layer_conv])

# Save the network architecture to file
plot_model(single_layer_conv_64, to_file='single_fc_layer_conv_model.png', show_layer_names=False, show_layer_activations=True, show_shapes=True, rankdir='TB')



"""Convolutional network with two FC layers"""

two_layer_weights_path = (
    file_loc.parent / "weights" / "two_fc_layer_CNN_model_weights" / "two_fc_layer_CNN_model_weights"
)
loss_path_two_layer = (
    file_loc.parent / "results" / "Two_FC_layers" / "two_fc_layer_CNN_model_history.json"
)

# Loading the saved weights from file.
two_layer_conv_64.load_weights(two_layer_weights_path)

# Remove """ on both sides of this block to keep training the model.
"""history_two_layer_conv = two_layer_conv_64.fit(X_train, y=y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=nb_epochs)

# Save the weights and loss history
two_layer_conv_64.save_weights(two_layer_weights_path)
json.dump(history_two_layer_conv.history, open(loss_path_two_layer, 'w+'))

two_layer_conv_64.evaluate(X_test, y_test, batch_size=batch_size)"""

# Predict classification labels for the test set. Highest output probability is used as prediction.
y_pred_two_layer_conv = two_layer_conv_64.predict(X_test, batch_size)
y_pred_two_layer_conv = np.array([np.argmax(pred) for pred in y_pred_two_layer_conv])

# Save the network architecture to file
plot_model(two_layer_conv_64, to_file='two_layer_conv_model.png', show_layer_names=False, show_layer_activations=True, show_shapes=True, rankdir='TB')



"""Sequential convolutional network from original article"""

origin_weights_path = (
    file_loc.parent / "weights" / "origin_model_weights" / "origin_model_weights"
)
loss_path_origin = (
    file_loc.parent / "results" / "O'Shea" / "origin_model_history.json"
)

# Loading the saved weights from file.
two_layer_conv_origin.load_weights(origin_weights_path)

# Remove """ on both sides of this block to keep training the model.
"""history_origin = two_layer_conv_origin.fit(X_train, y=y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=nb_epochs)

# Save the weights and loss history
two_layer_conv_origin.save_weights(origin_weights_path)
json.dump(history_origin.history, open(loss_path_origin, 'w+'))

two_layer_conv_origin.evaluate(X_test, y_test, batch_size=batch_size)"""

# Predict classification labels for the test set. Highest output probability is used as prediction.
y_pred_conv_origin = two_layer_conv_origin.predict(X_test, batch_size)
y_pred_conv_origin = np.array([np.argmax(pred) for pred in y_pred_conv_origin])

# Save the network architecture to file
plot_model(two_layer_conv_origin, to_file='original_conv_model.png', show_layer_names=False, show_layer_activations=True, show_shapes=True, rankdir='TB')



"""Parallel convolutional network with a single fully connected layer"""

parallel_single_weights_path = (
    file_loc.parent / "weights" / "parallel_single_fc_model_weights" / "parallel_single_fc_model_weights"
)
loss_path_parallel_single = (
    file_loc.parent / "results" / "Parallel_conv_Single_FC_Layer" / "parallel_single_fc_model_history.json"
)
# Loading the saved weights from file.

parallel_conv_single.load_weights(parallel_single_weights_path)

# Remove """ on both sides of this block to keep training the model.
"""history_parallel_single = parallel_conv_single.fit(X_train, y=y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=nb_epochs)

# Save the weights and loss history
parallel_conv_single.save_weights(parallel_single_weights_path)
json.dump(history_parallel_single.history, open(loss_path_parallel_single, 'w+'))

parallel_conv_single.evaluate(X_test, y_test, batch_size=batch_size)"""

# Predict classification labels for the test set. Highest output probability is used as prediction.
y_pred_parallel_conv_single = parallel_conv_single.predict(X_test, batch_size)
y_pred_parallel_conv_single = np.array([np.argmax(pred) for pred in y_pred_parallel_conv_single])

# Save the network architecture to file
plot_model(parallel_conv_single, to_file='parallel_conv_single_fc_model.png', show_layer_names=False, show_layer_activations=True, show_shapes=True, rankdir='TB')



"""Parallel convolutional network with two fully connected layers"""

parallel_two_weights_path = (
    file_loc.parent / "weights" / "parallel_two_fc_model_weights" / "parallel_two_fc_model_weights"
)
loss_path_parallel_two = (
    file_loc.parent / "results" / "Parallel_conv_Two_FC_Layers" / "parallel_two_model_history.json"
)

# Loading the saved weights from file.
parallel_conv_two.load_weights(parallel_two_weights_path)

# Remove """ on both sides of this block to keep training the model.
"""history_parallel_two = parallel_conv_two.fit(X_train, y=y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=nb_epochs)

# Save the weights and loss history
parallel_conv_two.save_weights(parallel_two_weights_path)
json.dump(history_parallel_two.history, open(loss_path_parallel_two, 'w+'))

parallel_conv_two.evaluate(X_test, y_test, batch_size=batch_size)"""

# Predict classification labels for the test set. Highest output probability is used as prediction.
y_pred_parallel_conv_two = parallel_conv_two.predict(X_test, batch_size)
y_pred_parallel_conv_two = np.array([np.argmax(pred) for pred in y_pred_parallel_conv_two])

# Save the network architecture to file
plot_model(parallel_conv_two, to_file='parallel_conv_two_fc_model.png', show_layer_names=False, show_layer_activations=True, show_shapes=True, rankdir='TB')



"""LSTM recurrent neural network with two fully connected layers"""

lstm_weights_path = (
    file_loc.parent / "weights" / "lstm_rnn_model_weights" / "lstm_rnn_model_weights"
)
loss_path_lstm = (
    file_loc.parent / "results" / "Single_LSTM" / "lstm_rnn_model_history.json"
)

# Loading the saved weights from file.
lstm_rnn.load_weights(lstm_weights_path)

# Remove """ on both sides of this block to keep training the model.
"""history_lstm = lstm_rnn.fit(X_train_rnn, y=y_train, validation_data=(X_val_rnn, y_val), batch_size=batch_size, epochs=nb_epochs)

# Save the weights and loss history
lstm_rnn.save_weights(lstm_weights_path)
json.dump(history_lstm.history, open(loss_path_lstm, 'w+'))

lstm_rnn.evaluate(X_test_rnn, y_test, batch_size=batch_size)"""

# Predict classification labels for the test set. Highest output probability is used as prediction.
y_pred_lstm_rnn = lstm_rnn.predict(X_test_rnn, batch_size)
y_pred_lstm_rnn = np.array([np.argmax(pred) for pred in y_pred_lstm_rnn])



"""Two LSTMs in a recurrent neural network with two fully connected layers"""

two_lstm_weights_path = (
    file_loc.parent / "weights" / "two_lstm_rnn_model_weights" / "two_lstm_rnn_model_weights"
)
loss_path_two_lstm = (
    file_loc.parent / "results" / "Two_LSTMs" / "two_lstm_rnn_model_history.json"
)

# Loading the saved weights from file.
two_lstm_rnn.load_weights(two_lstm_weights_path)

# Remove """ on both sides of this block to keep training the model.
"""history_two_lstm = two_lstm_rnn.fit(X_train_rnn, y=y_train, validation_data=(X_val_rnn, y_val), batch_size=batch_size, epochs=nb_epochs)

# Save the weights and loss history
two_lstm_rnn.save_weights(two_lstm_weights_path)
json.dump(history_two_lstm.history, open(loss_path_two_lstm, 'w+'))

two_lstm_rnn.evaluate(X_test_rnn, y_test, batch_size=batch_size)"""

# Predict classification labels for the test set. Highest output probability is used as prediction.
y_pred_two_lstm_rnn = two_lstm_rnn.predict(X_test_rnn, batch_size)
y_pred_two_lstm_rnn = np.array([np.argmax(pred) for pred in y_pred_two_lstm_rnn])



"""Produce relevant figures 
 (validation accuracy vs epoch, test accuracy vs SNR, confusion matrix - whole data set and specific SNR values)"""
y_test_label = np.array([np.argmax(y_true) for y_true in y_test])


# Plot accuracy vs SNR and confusion matrices for different SNR values
snr_levels, snr_indices = np.unique(np.array(snrs)[test_snr], return_inverse=True)

# 
accuracy_snr_single_layer_conv = np.zeros(len(snr_levels))
accuracy_snr_two_layer_conv = np.zeros(len(snr_levels))
accuracy_snr_origin_conv = np.zeros(len(snr_levels))
accuracy_snr_parallel_conv_single = np.zeros(len(snr_levels))
accuracy_snr_parallel_conv_two = np.zeros(len(snr_levels))
accuracy_snr_lstm= np.zeros(len(snr_levels))
accuracy_snr_two_lstm = np.zeros(len(snr_levels))

for snr_id in range(len(snr_levels)):
    # Find the samples with the given SNR
    samples = np.argwhere(snr_indices == snr_id)[:, 0]

    # Calculate the accuracies for the various models at the given SNR.
    accuracy_snr_single_layer_conv[snr_id] = np.mean(np.where(y_pred_single_layer_conv[samples] == y_test_label[samples], 1, 0))

    accuracy_snr_two_layer_conv[snr_id] = np.mean(np.where(y_pred_two_layer_conv[samples] == y_test_label[samples], 1, 0))

    accuracy_snr_origin_conv[snr_id] = np.mean(np.where(y_pred_conv_origin[samples] == y_test_label[samples], 1, 0))
    
    accuracy_snr_parallel_conv_single[snr_id] = np.mean(np.where(y_pred_parallel_conv_single[samples] == y_test_label[samples], 1, 0))
    
    accuracy_snr_parallel_conv_two[snr_id] = np.mean(np.where(y_pred_parallel_conv_two[samples] == y_test_label[samples], 1, 0))

    accuracy_snr_lstm[snr_id] = np.mean(np.where(y_pred_lstm_rnn[samples] == y_test_label[samples], 1, 0))
    
    accuracy_snr_two_lstm[snr_id] = np.mean(np.where(y_pred_two_lstm_rnn[samples] == y_test_label[samples], 1, 0))


# Create the test accuracy vs SNR plot
fig, ax = plt.subplots()
ax.set_xlabel('SNR')
ax.set_ylabel('Accuracy')
ax.set_xticks(np.arange(len(snr_levels)))
ax.set_xticklabels(snr_levels)
labels = ['Single FC layer (CNN)', 'Two FC layers (CNN)', "O'Shea et. al. (CNN)", 'Parallel conv.\nSingle FC layer (CNN)', 'Parallel conv.\nTwo FC layers (CNN)', 'Single LSTM (RNN)', 'Two LSTMs (RNN)']
for id, accuracy in enumerate([accuracy_snr_single_layer_conv, accuracy_snr_two_layer_conv, accuracy_snr_origin_conv, accuracy_snr_parallel_conv_single, accuracy_snr_parallel_conv_two, accuracy_snr_lstm, accuracy_snr_two_lstm]):
    ax.plot(np.arange(len(snr_levels)), accuracy, label=labels[id])

ax.legend()
plt.show()
plt.close()



# Plot confusion matrices for the all samples with positive SNR from the test set
predictions_models = [y_pred_single_layer_conv, y_pred_two_layer_conv, y_pred_conv_origin, y_pred_parallel_conv_single, y_pred_parallel_conv_two, y_pred_lstm_rnn, y_pred_two_lstm_rnn]
for id, pred in enumerate(predictions_models):
    samples = np.argwhere(snr_indices >= 10)[:, 0]
    fig, ax = plt.subplots(figsize = (12, 12), tight_layout=True)
    confusion = np.round(confusion_matrix(y_test_label[samples], pred[samples], normalize='true'), 2)
    confusion_plot = ConfusionMatrixDisplay(confusion, display_labels=mods)
    confusion_plot.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax.set_title('{} (SNR'.format(labels[id])+r'$\geq$'+'{})'.format(snrs[10]), fontsize=26)
    fig.savefig((
        file_loc.parent / "results" / "{}".format(labels[id][:-6].replace(' ', '_').replace('.', '').replace('_al', '').replace('_et', '').replace('\n', '_')) / 
        "confusion_positive_snr_{}.pdf".format(labels[id][:-6].replace(' ', '_').replace('.', '').replace('_al', '').replace('_et', '').replace('\n', '_')))
    )
    plt.show()
    plt.close()
    


# Plot confusion matrices for individual SNR values
for id, pred in enumerate(predictions_models):
    for snr_id in [0, 5, 10, 15, 19]:
        samples_snr = np.argwhere(snr_indices == snr_id)[:, 0]
        fig_snr, ax_snr = plt.subplots(figsize = (12, 12), tight_layout=True)
        confusion_snr = np.round(confusion_matrix(y_test_label[samples_snr], pred[samples_snr], normalize='true'), 2)
        confusion_snr_plot = ConfusionMatrixDisplay(confusion_snr, display_labels=mods)
        confusion_snr_plot.plot(ax=ax_snr, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax_snr.set_title('{} (SNR {})'.format(labels[id], snrs[snr_id]), fontsize=26)
        fig_snr.savefig((
        file_loc.parent / "results" / "{}".format(labels[id][:-6].replace(' ', '_').replace('.', '').replace('_al', '').replace('_et', '').replace('\n', '_')) / 
        "confusion_{}_snr_{}.pdf".format(snrs[snr_id], labels[id][:-6].replace(' ', '_').replace('.', '').replace('_al', '').replace('_et', '').replace('\n', '_')))
        )
        plt.close()
    
    
    # Create plot showing -8dB and 18dB confusion matrix
    fig, ax = plt.subplots(ncols=2, tight_layout=True)
    
    # Confusion matrix for -8dB SNR
    samples = np.argwhere(snr_indices == 6)[:, 0]
    
    confusion = np.round(confusion_matrix(y_test_label[samples], pred[samples], normalize='true'), 2)
    
    confusion_plot = ConfusionMatrixDisplay(confusion, display_labels=mods)
    confusion_plot.plot(ax=ax[0], cmap='Blues', xticks_rotation=45, colorbar=False)
    ax[0].set_title('{} (SNR={}dB)'.format(labels[id], snrs[6]), fontsize=26)

    # Confusion matrix for 18dB SNR
    samples2 = np.argwhere(snr_indices == 19)[:, 0]
    
    confusion2 = np.round(confusion_matrix(y_test_label[samples2], pred[samples2], normalize='true'), 2)
    
    confusion_plot2 = ConfusionMatrixDisplay(confusion2, display_labels=mods)
    confusion_plot2.plot(ax=ax[1], cmap='Blues', xticks_rotation=45, colorbar=False)
    ax[1].set_title('{} (SNR={}dB)'.format(labels[id], snrs[19]), fontsize=26)
    plt.subplots_adjust(wspace=0.2)

    plt.show()

    plt.close()



# Plot validation accuracy vs epoch
# Load the training history of the models, containing the validation loss after each epoch.
history_single_layer_conv = json.load(open(loss_path_single_layer, 'r'))

history_two_layer_conv = json.load(open(loss_path_two_layer, 'r'))

history_origin = json.load(open(loss_path_origin, 'r'))

history_parallel_single = json.load(open(loss_path_parallel_single, 'r'))

history_parallel_two = json.load(open(loss_path_parallel_two, 'r'))

history_lstm = json.load(open(loss_path_lstm, 'r'))

history_two_lstm = json.load(open(loss_path_two_lstm, 'r'))


fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation accuracy')
for id, hist in enumerate([history_single_layer_conv, history_two_layer_conv, history_origin, history_parallel_single, history_parallel_two, history_lstm, history_two_lstm]):
    ax.plot(np.arange(1, len(hist['accuracy']) + 1), np.array(hist['val_accuracy']), label=labels[id])

ax.legend()
plt.show()
plt.close()