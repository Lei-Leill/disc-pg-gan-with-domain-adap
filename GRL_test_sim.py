# python imports
import keras
import numpy as np
import random
import sys, optparse
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# our imports
import discriminator
import global_vars
from slim_iterator import SlimIterator
from real_data_random import RealDataRandomIterator
import param_set
import simulation
import generator

SLIM_DATA = "../data/1000g/" 

NEUTRAL = "neutral"
SELECTION = ["sel_01", "sel_025", "sel_05", "sel_10"]
BATCH_PER_EPOCH = 200 # arbitrary 

def evaluate_model(model, validation_generator):
    print("into the function")
    # Get predictions from the model
    y_true = []
    y_pred = []

    for data, labels in validation_generator:
        # Get the true labels for the classifier
        y_true_batch = labels['classifier']
        y_pred_batch = model.predict(data) # Predict with the model
        
        y_pred_batch = (y_pred_batch[0] > 0.5).astype(int)

        # Append true and predicted labels
        y_true.extend(y_true_batch.flatten())
        y_pred.extend(y_pred_batch.flatten())

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plotting the confusion matrix using matplotlib and seaborn
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral","Selected"])
    disp.plot()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion Matrix for Classifier Matrix")

class TrainingSeq(keras.utils.Sequence):

    def __init__(self, train_neutral_iterator, train_sel_iterator, test_neutral_iterator, test_sel_iterator, train, batch_size=None):
        self.train_neutral_iterator = train_neutral_iterator
        self.train_sel_iterator = train_sel_iterator
        self.test_neutral_iterator = test_neutral_iterator
        self.test_sel_iterator = test_sel_iterator
        self.batch_size = batch_size # not needed for test
        self.train = train # boolean

    def __len__(self):
        """ number of batches per epoch """
        if self.train:
            return BATCH_PER_EPOCH
        return 1

    def __getitem__(self, idx):
        if self.train:
            # not actually "real" data
            half = self.batch_size//4
            train_neutral_regions = self.train_neutral_iterator.real_batch(batch_size=half)
            train_sel_regions = self.mixed_sel_batch(half, self.train_sel_iterator)
            test_neutral_regions = self.test_neutral_iterator.real_batch(batch_size=half)
            test_sel_regions = self.mixed_sel_batch(half, self.test_sel_iterator)
            regions = np.concatenate((train_neutral_regions, train_sel_regions, test_neutral_regions, test_sel_regions), axis=0)
            
            class_labels = np.concatenate((np.zeros((half,1)), np.ones((half,1)), -1*np.ones((half,1)), -1* np.ones((half,1))))
            disc_labels = np.concatenate((np.zeros((half*2,1)), np.ones((half*2,1))))

            self.batch_size = 4* half
            # shuffle
            idx = np.random.permutation(self.batch_size)
            regions, class_labels, disc_labels = regions[idx], class_labels[idx], disc_labels[idx]

        else:
            neutral_regions = self.test_neutral_iterator.test_batch()
            num_neutral = len(neutral_regions)
            sel_regions = [iter.test_batch() for iter in self.test_sel_iterator]
            num_sel = sum([len(x) for x in sel_regions])

            regions = np.concatenate([neutral_regions] + sel_regions, axis=0)
            #labels = np.concatenate((np.zeros((num_neutral,1)), np.ones((num_sel,1))))
            class_labels = np.concatenate((np.zeros((num_neutral,1)), np.ones((num_sel,1))))
            disc_labels = np.ones((num_neutral + num_sel,1))

            # don't need to shuffle
        #print(f"Data shape: {regions.shape}, Class labels shape: {class_labels.shape}, Disc labels shape{disc_labels.shape}")
        return regions, {"classifier": class_labels, "discriminator": disc_labels}

    def mixed_sel_batch(self, num_regions, sel_iterators):
        # just get selected regions
        x = len(sel_iterators) # num iters
        mini = num_regions//x
        excess = num_regions - mini*x 
        sel_regions = [iter.real_batch(batch_size=mini) for iter in sel_iterators]
        excess_regions = random.choice(sel_iterators).real_batch(batch_size=excess)
        sel_regions = np.concatenate(sel_regions + [excess_regions], axis=0)

        assert len(sel_regions) == num_regions
        return sel_regions


def main(): #, loss_filename):#, output_filename=None):
    filename = "/homes/tlei/mathiesonlab/data/HDF5/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5"
    real_iterator = RealDataRandomIterator(filename, global_vars.DEFAULT_SEED) 

    TRAIN = "CEU/"
    VAL = "YRI/"
    train_neutral_iterator = SlimIterator(SLIM_DATA + TRAIN + NEUTRAL)
    train_sel_iterators = [SlimIterator(SLIM_DATA + TRAIN + sel) for sel in SELECTION]

    val_neutral_iterator = SlimIterator(SLIM_DATA + VAL+ NEUTRAL)
    val_sel_iterators = [SlimIterator(SLIM_DATA + VAL + sel) for sel in SELECTION]

    # set up CNN model
    print("num haps", train_neutral_iterator.num_samples)
    print("num haps", val_neutral_iterator.num_samples)
    # training params
    cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    # train and validation data
    training_generator = TrainingSeq(train_neutral_iterator, train_sel_iterators, val_neutral_iterator, val_sel_iterators, True, batch_size=global_vars.BATCH_SIZE)
    validation_generator = TrainingSeq(None, None, val_neutral_iterator, val_sel_iterators, False)
    
    print("Generator done")
    #model = discriminator.OnePopModel(train_neutral_iterator.num_samples)
    #model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])
    model = discriminator.create_custom_grl_model()
    history = model.fit(training_generator, validation_data=validation_generator, epochs=20, verbose = 2)
     # Plot accuracy over time (for training and validation)
    
    #print(history.history)

    plt.plot(history.history['discriminator_accuracy'], label='Training Accuracy')
    plt.plot(history.history['discriminator_loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Model Accuracy over Time')
    plt.legend()
    plt.savefig('Disc_Train_Accuracy_vs_Epoch.pdf')

    plt.clf()
    plt.plot(history.history['classifier_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_classifier_accuracy'], label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy (Train, Test) over Time')
    plt.legend()
    plt.savefig('Class_Train_Test_Accuracy_vs_Epoch.pdf')
    evaluate_model(model, validation_generator)


if __name__ == "__main__":
    main()