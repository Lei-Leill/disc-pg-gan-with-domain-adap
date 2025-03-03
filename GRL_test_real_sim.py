# python imports
import keras
import numpy as np
import random
import sys, optparse
import tensorflow as tf

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

class SlimRealSequence(keras.utils.Sequence):

    def __init__(self, real_iterator, neutral_iterator, sel_iterators, train, batch_size=None):
        self.real_iterator = real_iterator
        self.neutral_iterator = neutral_iterator
        self.sel_iterators = sel_iterators
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
            half = self.batch_size//2
            real_regions = self.real_iterator.real_batch(batch_size = half)
            neutral_regions = self.neutral_iterator.real_batch(batch_size=half//2)
            sel_regions = self.mixed_sel_batch(half//2)
            regions = np.concatenate((real_regions, neutral_regions, sel_regions), axis=0)
            
            # real data -> 0; simulated data -> 1
            labels = np.concatenate((np.zeros((half,1)), np.ones((half,1))))

            # shuffle
            idx = np.random.permutation(self.batch_size)
            regions, labels = regions[idx], labels[idx]

        else:
            neutral_regions = self.neutral_iterator.test_batch()
            num_neutral = len(neutral_regions)
            sel_regions = [iter.test_batch() for iter in self.sel_iterators]
            num_sel = sum([len(x) for x in sel_regions])

            regions = np.concatenate([neutral_regions] + sel_regions, axis=0)
            labels = np.concatenate((np.zeros((num_neutral,1)), np.ones((num_sel,1))))
            # don't need to shuffle
    
        return regions, labels

    def mixed_sel_batch(self, num_regions):
        # just get selected regions
        x = len(self.sel_iterators) # num iters
        mini = num_regions//x
        excess = num_regions - mini*x 
        sel_regions = [iter.real_batch(batch_size=mini) for iter in self.sel_iterators]
        excess_regions = random.choice(self.sel_iterators).real_batch(batch_size=excess)
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
    training_generator = SlimSequence(train_neutral_iterator, train_sel_iterators, True, batch_size=global_vars.BATCH_SIZE)
    validation_generator = SlimSequence(val_neutral_iterator, val_sel_iterators, False)
    
    print("Generator done")
    #model = discriminator.OnePopModel(train_neutral_iterator.num_samples)
    #model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])
    model = discriminator.create_custom_grl_model()
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator, epochs=10, verbose = 0)

    #model.save("models/SLiM_model")


if __name__ == "__main__":
    train()
        


