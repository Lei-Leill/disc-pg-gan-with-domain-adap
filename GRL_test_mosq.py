from generator import Generator
import simulation
import global_vars
from slim_iterator import SlimIterator
import discriminator


import keras
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


SLIM_DATA = "../data/1000g/" 
NEUTRAL = "neutral"
SELECTION = ["sel_01", "sel_025", "sel_05", "sel_10"]
TRAIN = "CEU/"
#SELECTION = ["sel_01"]
BATCH_PER_EPOCH = 200 # arbitrary 

class Seq(keras.utils.Sequence):
    def __init__(self, train_neutral_iterator, train_sel_iterator, test_mosq, train, with_GRL, batch_size=None):
        self.train_neutral_iterator = train_neutral_iterator
        self.train_sel_iterator = train_sel_iterator
        self.test_mosq = test_mosq  # Should be a Generator instance
        self.batch_size = batch_size if batch_size is not None else global_vars.BATCH_SIZE
        self.train = train  # Boolean flag for training vs. validation
        self.with_GRL = with_GRL # Boolean flag for having GRL layer or not

    def __len__(self):
        """ Number of batches per epoch """
        if self.train:
            return BATCH_PER_EPOCH
        return 1  # Only one validation batch by default

    def __getitem__(self, idx):
        if self.train:
            quart = self.batch_size // 4
            if self.with_GRL and self.test_mosq:
                size = quart
            else: size = quart * 2

            # Simulated data
            train_neutral_regions = self.train_neutral_iterator.real_batch(batch_size=size)
            train_sel_regions = self.mixed_sel_batch(size, self.train_sel_iterator)

            # Labels
            if self.with_GRL and self.test_mosq: # the case of GBR layer 
                 # Simulated mosquito data
                test_mos_regions = self.test_mosq.simulate_batch(batch_size=quart * 2)
                regions = np.concatenate((train_neutral_regions, train_sel_regions, test_mos_regions), axis=0) # combine together

                class_labels = np.concatenate((
                    np.zeros((quart, 1)),           # Neutral: 0
                    np.ones((quart, 1)),            # Selected: 1
                    -1 * np.ones((quart * 2, 1))     # Simulated mosquito data: unlabeled
                ), axis=0)
            else:
                regions = np.concatenate((train_neutral_regions, train_sel_regions), axis=0) # combine together

                class_labels = np.concatenate((
                    np.zeros((quart*2, 1)),           # Neutral: 0
                    np.ones((quart*2, 1)),            # Selected: 1
                ), axis=0)

            disc_labels = np.concatenate((
                np.zeros((quart * 2, 1)),       # Real data (neutral + selected): 0
                np.ones((quart * 2, 1))         # Simulated mosquito: 1
            ), axis=0)

            # Shuffle
            total = 4 * quart
            idx = np.random.permutation(total)
            regions, class_labels, disc_labels = regions[idx], class_labels[idx], disc_labels[idx]

        else:
            # Validation: use only mosquito simulated data
            regions = self.test_mosq.simulate_batch(batch_size=self.batch_size)

            class_labels = np.zeros((self.batch_size, 1))  # Neutral for all
            disc_labels = np.ones((self.batch_size, 1))    # All are simulated

        if self.with_GRL: 
            return regions, {"classifier": class_labels, "discriminator": disc_labels}
        else: 
            return regions, class_labels

    def mixed_sel_batch(self, num_regions, sel_iterators):
        """ Combines multiple SlimIterator objects with selected regions """
        num_iters = len(sel_iterators)
        base = num_regions // num_iters
        remainder = num_regions - base * num_iters

        sel_regions = [it.real_batch(batch_size=base) for it in sel_iterators]
        sel_regions.append(random.choice(sel_iterators).real_batch(batch_size=remainder))

        return np.concatenate(sel_regions, axis=0)

def plot_metric(train, val, title, ylabel, filename, file_path, y_offset=(0.005, 0.015)):
    plt.clf()
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train, label='Training', marker='o')
    plt.plot(epochs, val, label='Testing', marker='o')
    plt.xlim(left=1)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    # Add labels to each data point
    for i, (tr, va) in enumerate(zip(train, val), start=1):
        plt.text(i, tr + y_offset[0], f'{tr:.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i, va - y_offset[1], f'{va:.2f}', ha='center', va='top', fontsize=8)

    plt.savefig(os.path.join(file_path, filename))

def plotting(history, with_GRL):
    file_path = f'figs/Mosq/{"withGRL" if with_GRL else "withoutGRL"}/'
    os.makedirs(file_path, exist_ok=True)

    if with_GRL:
        plot_metric(
            history.history['classifier_accuracy'],
            history.history['val_classifier_accuracy'],
            title='Classifier Accuracy (Train, Test) over Time',
            ylabel='Accuracy',
            filename='Classifier_Accuracy.pdf',
            file_path=file_path,
            y_offset=(0.005, 0.015)
        )

        plot_metric(
            history.history['classifier_loss'],
            history.history['val_classifier_loss'],
            title='Classifier Loss (Train, Test) over Time',
            ylabel='Loss',
            filename='Classifier_Loss.pdf',
            file_path=file_path,
            y_offset=(0.02, 0.03)
        )

        # Plot discriminator accuracy and loss together
        plt.clf()
        epochs = range(1, len(history.history['discriminator_accuracy']) + 1)
        plt.plot(epochs, history.history['discriminator_accuracy'], label='Training Accuracy', marker='o')
        plt.plot(epochs, history.history['discriminator_loss'], label='Training Loss', marker='o')
        plt.xlim(left=1)
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy / Loss')
        plt.title('Model Accuracy and Loss over Time')
        plt.legend()

        for i, (acc, loss) in enumerate(zip(history.history['discriminator_accuracy'], history.history['discriminator_loss']), start=1):
            plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
            plt.text(i, loss - 0.02, f'{loss:.2f}', ha='center', va='top', fontsize=8)

        plt.savefig(os.path.join(file_path, 'Disc_Acc_Loss.pdf'))

    else:
        plot_metric(
            history.history['accuracy'],
            history.history['val_accuracy'],
            title='Accuracy (Train, Test) over Time',
            ylabel='Accuracy',
            filename='Classifier_Accuracy.pdf',
            file_path=file_path,
            y_offset=(0.005, 0.015)
        )

        plot_metric(
            history.history['loss'],
            history.history['val_loss'],
            title='Classifier Loss (Train, Test) over Time',
            ylabel='Loss',
            filename='Classifier_Loss.pdf',
            file_path=file_path,
            y_offset=(0.02, 0.03)
        )

def main():
    
    train_neutral_iterator = SlimIterator(SLIM_DATA + TRAIN + NEUTRAL)
    train_sel_iterators = [SlimIterator(SLIM_DATA + TRAIN + sel) for sel in SELECTION]

    val_mosq = Generator(
        simulation.exp,  # Simulator function
        ["N1", "T1", "T2", "N2", "growth"],  # Parameter names
        [198],  # Sample sizes
        global_vars.DEFAULT_SEED  # Random seed
    )
    # Override default parameters with the values provided by Sara Mathieson
    val_mosq.update_params([
    5591.367991008426,
    8989.8024202365,
    101.83233386492524,
    25459200.34194682,
    0.017934435132958317
    ])
    # set up CNN model
    print("num haps", train_neutral_iterator.num_samples)
    # training params
    cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    '''
    # With GRL layer
    training_generator = Seq(train_neutral_iterator, train_sel_iterators, val_mosq, True, True, batch_size=global_vars.BATCH_SIZE)
    validation_generator = Seq(None, None, val_mosq, False, True)    
    model = discriminator.create_custom_grl_model()
    history = model.fit(training_generator, validation_data=validation_generator, epochs=20, verbose = 2)
    # Plot accuracy over time (for training and validation)
    plotting(history, True)
    '''
    
    # without GRL Layer
    training_generator = Seq(train_neutral_iterator, train_sel_iterators, None, True, False, batch_size=global_vars.BATCH_SIZE)
    validation_generator = Seq(None, None, val_mosq, False, False)
    model = discriminator.OnePopModel(train_neutral_iterator.num_samples)
    model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])
    history = model.fit(training_generator, validation_data=validation_generator, epochs=10, verbose = 2)
    plotting(history, False)
    
if __name__ == "__main__":
    main()

