"""
Loads a trained discriminator (trained with pg-gan) and fine-tunes it using
simulated selection data (SLiM) with label 0 for neutral and 1 for selection.
Edits for case when discriminator is non-trained.
Authors: Sara Mathieson
Date: 8/21/23
"""

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
import param_set
import simulation
import generator
#import feature_extractor

################################################################################
# GLOBALS
################################################################################

#SEL_TYPE = "Aug23" # change for different types of selection (i.e. Aug23, Over)
#MAIN_PATH = "/homes/smathieson/Documents/pg_gan_interpret/"

# globals
#TRAIN_POP = sys.argv[1] # i.e. CEU, ALL for ALL_AI
#SEL_TYPE = sys.argv[2]  # change for different types of selection (i.e. Aug23, Over)

#SLIM_DATA = "/bigdata/smathieson/pg-gan/1000g/SLiM/Aug23/" + TRAIN_POP + "_" + SEL_TYPE + "/"
#SLIM_DATA = "/bigdata/smathieson/1000g-share/SLiM/" + TRAIN_POP + "_" + SEL_TYPE + "/"
SLIM_DATA = "../data/1000g/" 

NEUTRAL = "neutral"
SELECTION = ["sel_01", "sel_025", "sel_05", "sel_10"]
BATCH_PER_EPOCH = 200 # arbitrary 

################################################################################
# HELPERS
################################################################################
def parse_args():
    parser = optparse.OptionParser(description='Dataset and loss function to train cnn model')
    
    parser.add_option('-d', '--dataset', type='choice', choices=["SLiM", "msprime"], help='Enter one dataset (SLiM / msprime)')

    (opts, args) = parser.parse_args()

    mandatories = ['dataset']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()
    return opts

def accuracy(y_true, y_pred): # y_pred is probabilities, use 0.5 threshold
    a = 0
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        logit = y_pred[i].numpy()[0]
        result = 1 if logit >= 0 else 0
        if int(y_true[i][0]) == result:
            a += 1
    return a/len(y_true)

class SlimSequence(keras.utils.Sequence):

    def __init__(self, neutral_iterator, sel_iterators, train, batch_size=None):
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
            neutral_regions = self.neutral_iterator.real_batch(batch_size=half)
            sel_regions = self.mixed_sel_batch(half)
            regions = np.concatenate((neutral_regions, sel_regions), axis=0)
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

class GeneratorSequence(keras.utils.Sequence):

    def __init__(self, train, batch_size = None):
        self.train = train
        self.batch_size = batch_size

    def __len__(self):
        """ number of batches per epoch """
        if self.train:
            return BATCH_PER_EPOCH
        return 1

    def bin_genotype_matrix(self, genotype_matrix, bin_size):
        # Calculate the number of bins
        '''num_bins = (genotype_matrix.shape[1] + bin_size - 1) // bin_size

        # Initialize the binned matrix
        binned_matrix = np.zeros((genotype_matrix.shape[0], num_bins))

        for i in range(num_bins):
            # Sum the values in each bin
            start_index = i * bin_size
            end_index = min((i + 1) * bin_size, genotype_matrix.shape[1])
            binned_matrix[:, i] = np.sum(genotype_matrix[:, start_index:end_index], axis=1)

        return binned_matrix'''
        original_shape = genotype_matrix.shape
    
        # Calculate the number of bins along the specified axis
        num_bins = (original_shape[axis] + bin_size - 1) // bin_size
        
        # Prepare new shape for the binned matrix
        new_shape = list(original_shape)
        new_shape[axis] = num_bins
        
        # Initialize the binned matrix
        binned_matrix = np.zeros(new_shape)
        
        # Perform the binning
        for i in range(num_bins):
            start_index = i * bin_size
            end_index = min((i + 1) * bin_size, original_shape[axis])
            
            if axis == 0:
                binned_matrix[i, :, :, :] = np.sum(genotype_matrix[start_index:end_index, :, :, :], axis=0)
            elif axis == 1:
                binned_matrix[:, i, :, :] = np.sum(genotype_matrix[:, start_index:end_index, :, :], axis=1)
            elif axis == 2:
                binned_matrix[:, :, i, :] = np.sum(genotype_matrix[:, :, start_index:end_index, :], axis=2)
            elif axis == 3:
                binned_matrix[:, :, :, i] = np.sum(genotype_matrix[:, :, :, start_index:end_index], axis=3)
        
        return binned_matrix
    
    def __getitem__(self, idx):
        '''features = BinnedHaplotypeMatrix(
            num_individuals=182,  # 
            num_loci=64,          # Number of bins
            ploidy=2,             # Assuming diploid data
            phased=False,         # Assuming unphased data
            maf_thresh=0.05       # MAF threshold
        )
        '''
        exp_params = param_set.ParamSet(simulation.exp)
        seed = random.randint(0, 100000)
        generat = generator.Generator(simulation.exp, ["N1", "T1"], [182], seed)
        #generat = generator.Generator(simulation.exp, ["N1", "T1"], [20], global_vars.DEFAULT_SEED)
        #generat.update_params([exp_params.N1.value, exp_params.T1.value])
        
        
        generat.update_params([15000, exp_params.T1.value])
        mini_batch_1 = generat.simulate_batch(batch_size = 25)
        y1 = np.zeros(mini_batch_1.shape[0])
        #y1 = tf.one_hot(np.zeros(mini_batch_1.shape[0]), depth = 3)
        
        generat.update_params([30000, exp_params.T1.value])
        mini_batch_2 = generat.simulate_batch(batch_size = 25)
        y2 = np.ones(mini_batch_2.shape[0])
        #y2 = tf.one_hot(np.ones(mini_batch_2.shape[0]), depth = 3)

        generat.update_params([45000, exp_params.T1.value])
        mini_batch_3 = generat.simulate_batch(batch_size = 25)
        y3 = np.array([2 for i in range(25)])
        #y3 = tf.one_hot(np.array([2 for i in range(25)]), depth = 3)
 
        y = np.concatenate((y1, y2, y3), axis = 0)
        regions = np.concatenate((mini_batch_1, mini_batch_2, mini_batch_3), axis = 0)
        print(regions.shape)
        '''binned_matrices = [features.from_ts(ts) for ts in regions]
        # Combine the binned matrices into a single array
        binned_regions = np.concatenate(binned_matrices, axis=0)'''

        binned_regions = self.bin_genotype_matrix(regions, 5)
        print(binned_regions.shape)
        shuffled_index = np.random.permutation(np.arange(len(regions)))

        # Shuffle the binned regions and labels
        #binned_regions = binned_regions[shuffled_index]
        regions = regions[shuffled_index]
        y = y[shuffled_index]

        # Return the shuffled binned regions and labels
        return regions, y

        '''y = []
        regions = np.empty((0,0,0,0))
        for i in range (50):
            rand_num = random.randint(10000, 30000)
            generat.update_params([rand_num, exp_params.T1.value])
            mini_batch = generat.simulate_batch(batch_size = 1)
            if regions.size == 0:
                regions = mini_batch.copy()
            else:
                regions = np.concatenate((regions, mini_batch), axis = 0)
            y.append(rand_num)
        '''
        '''
         # set 5000 (low) and 10000(high)
        generat.update_params([30000, exp_params.T1.value])
        mini_batch_1 = generat.simulate_batch(batch_size = 25)
        y1 = np.zeros(mini_batch_1.shape[0])

        generat.update_params([45000, exp_params.T1.value])
        mini_batch_2 = generat.simulate_batch(batch_size = 25)
        y2 = np.ones(mini_batch_2.shape[0])
        y = np.concatenate((y1, y2), axis = 0)
        regions = np.concatenate((mini_batch_1, mini_batch_2), axis = 0)

        shuffled_index = np.random.permutation(np.arange(75))
        regions = regions[shuffled_index]
        y = np.array(y)[shuffled_index]
        return regions, y'''
        


################################################################################
# TRAINING
################################################################################

def train(): #, loss_filename):#, output_filename=None):
    
    options = parse_args()
    print(options)
    dataset = options.dataset

    if dataset == 'SLiM':
        # SLiM data
        '''
        neutral_iterator = SlimIterator(SLIM_DATA + NEUTRAL)
        sel_iterators = [SlimIterator(SLIM_DATA + sel) for sel in SELECTION]

        # set up CNN model
        print("num haps", neutral_iterator.num_samples)

        # training params
        cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()
        # train and validation data
        training_generator = SlimSequence(neutral_iterator, sel_iterators, True, batch_size=global_vars.BATCH_SIZE)
        validation_generator = SlimSequence(neutral_iterator, sel_iterators, False)'''
        
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
        model = discriminator.OnePopModel(train_neutral_iterator.num_samples)
        model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])
        #model = discriminator.create_custom_grl_model()
        #model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10, verbose = 0)
        model.fit(training_generator, validation_data=validation_generator, epochs=10, verbose = 2)

        #model.save("models/SLiM_model")
    
    elif dataset == 'msprime':
        # set up CNN model
        model = discriminator.OnePopModel(182)
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits= True)
        categori_cross_entro = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True)
        optimizer = tf.keras.optimizers.Adam()
        #model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])
        #model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
        model.compile(optimizer=optimizer, loss=categori_cross_entro, metrics=['accuracy'])

        # train and validation data
        training_generator = GeneratorSequence(True, batch_size=global_vars.BATCH_SIZE)
        validation_generator = GeneratorSequence(False)

        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator, epochs=10)
        model.save("models/5000(0)_10000(1)_model")
    


        

################################################################################
# MAIN
################################################################################

if __name__ == "__main__":

    train()