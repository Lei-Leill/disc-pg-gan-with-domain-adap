"""
CNN-based discriminator models for pg-gan.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date: 2/4/21
"""

# python imports
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, \
    MaxPooling2D, AveragePooling1D, Dropout, Concatenate, Layer
from tensorflow.keras import Model


class GradientReversalLayer(Layer):
    """Custom layer to reverse the gradient."""
    def __init__(self):
        super().__init__()

    @tf.custom_gradient
    def call(self, x):
        y = tf.identity(x)
        def grad(dy):
            return -dy  # Reverse the gradient
        return y, grad

def custom_loss(y_true, y_pred):
    y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def create_custom_grl_model(dropout_rate=0.5):
    input_dims = [40, 36, 2] # channel last
    inputs = Input(shape=input_dims)
    print(f"Input shape: {inputs.shape}")

    # Convolutional layers with pooling
    x = Conv2D(32, (1, 5), activation='relu')(inputs)
    print(f"After Conv2D (32 filters): {x.shape}")
    x = MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)
    print(f"After MaxPooling2D: {x.shape}")

    x = Conv2D(64, (1, 5), activation='relu')(x)
    print(f"After Conv2D (64 filters): {x.shape}")
    x = MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)
    print(f"After MaxPooling2D: {x.shape}")

    # Apply reduction (permutation-invariant function)
    x = tf.reduce_sum(x, axis=1)
    
    # Flatten and fully connected layers for classification
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    class_output = Dense(1, activation='sigmoid', name='classifier')(x)

    '''
    # Discriminator pathway with Gradient Reversal Layer
    disc_x = GradReverse()(x)
    disc_x = Dense(1024, activation='relu')(disc_x)
    disc_x = Dense(512, activation='relu')(disc_x)
    disc_output = Dense(1, activation='sigmoid', name='discriminator')(disc_x)

    
    # Build and compile the model
    model = Model(inputs=inputs, outputs=[class_output, disc_output])
    model.compile(optimizer='adam',
                  loss=[custom_loss, custom_loss],
                  loss_weights=[1, 1],
                  metrics=['accuracy'])
    '''
    # Build and compile the model
    model = Model(inputs=inputs, outputs=[class_output])
    model.compile(optimizer='adam',
                  loss=[custom_loss],
                  metrics=['accuracy'])
    return model
    

class OnePopModel(Model):
    """Single population model - based on defiNETti software."""

    def __init__(self, pop, saved_model=None):
        super(OnePopModel, self).__init__()

        if saved_model is None:
            # it is (1,5) for permutation invariance (shape is n X SNPs)
            self.conv1 = Conv2D(32, (1, 5), activation='relu')
            self.conv2 = Conv2D(64, (1, 5), activation='relu')
            self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))
            #self.grl = GradientReversalLayer()
            self.flatten = Flatten()
            self.dropout = Dropout(rate=0.5)

            # change from 128,128 to 32,32,16 (same # parxams)
            self.fc1 = Dense(128, activation='relu')
            self.fc2 = Dense(128, activation='relu')
            self.dense3 = Dense(1) #3, activation='softmax') # two classes

        else:
            self.conv1 = saved_model.conv1
            self.conv2 = saved_model.conv2
            self.pool = saved_model.pool
            #self.grl = GradientReversalLayer()

            self.flatten = saved_model.flatten
            self.dropout = saved_model.dropout

            self.fc1 = saved_model.fc1
            self.fc2 = saved_model.fc2
            self.dense3 = saved_model.dense3

        self.pop = pop

    def after_perm(self, x):
        """ Note this should mirror call, get data right after
         permutation-invariant function """
        assert x.shape[1] == self.pop

        print("entering after_perm")
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.pool(x) # pool
        x = self.conv2(x)
        x = self.pool(x) # pool

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        #x = tf.math.reduce_max(x, axis=1)
        x = tf.math.reduce_sum(x, axis=1)
        return x

    def last_hidden_layer(self, x):
        """ Note this should mirror call """
        assert x.shape[1] == self.pop
        x = self.conv1(x)
        x = self.pool(x) # pool
        x = self.conv2(x)
        x = self.pool(x) # pool

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        #x = tf.math.reduce_max(x, axis=1)
        x = tf.math.reduce_sum(x, axis=1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=False)

        x = self.fc2(x)
        x = self.dropout(x, training=False)

        return x

    def call(self, x, training=None):
        """x is the genotype matrix + distances"""
        #assert x.shape[1] == self.pop
        x = self.conv1(x)
        x = self.pool(x) # pool
        x = self.conv2(x)
        x = self.pool(x) # pool

        # note axis is 1 b/c first axis is batch
        # can try max or sum as the permutation-invariant function
        #x = tf.math.reduce_max(x, axis=1)
        x = tf.math.reduce_sum(x, axis=1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.dropout(x, training=training)

        return self.dense3(x)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)

class TwoPopModel(Model):
    """Two population model"""

    # integers for num pop1, pop2
    def __init__(self, pop1, pop2):
        super(TwoPopModel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.merge = Concatenate()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(1) # 2, activation='softmax') # two classes

        self.pop1 = pop1
        self.pop2 = pop2

    def call(self, x, training=None):
        """x is the genotype matrix, dist is the SNP distances"""
        assert x.shape[1] == self.pop1 + self.pop2

        # first divide into populations
        x_pop1 = x[:, :self.pop1, :, :]
        x_pop2 = x[:, self.pop1:, :, :]

        # two conv layers for each part
        x_pop1 = self.conv1(x_pop1)
        x_pop2 = self.conv1(x_pop2)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool

        x_pop1 = self.conv2(x_pop1)
        x_pop2 = self.conv2(x_pop2)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool

        # 1 is the dimension of the individuals
        # can try max or sum as the permutation-invariant function
        #x_pop1_max = tf.math.reduce_max(x_pop1, axis=1)
        #x_pop2_max = tf.math.reduce_max(x_pop2, axis=1)
        x_pop1_sum = tf.math.reduce_sum(x_pop1, axis=1)
        x_pop2_sum = tf.math.reduce_sum(x_pop2, axis=1)

        # flatten all
        x_pop1_max = self.flatten(x_pop1_max)
        x_pop2_max = self.flatten(x_pop2_max)
        #x_pop1_sum = self.flatten(x_pop1_sum)
        #x_pop2_sum = self.flatten(x_pop2_sum)

        # concatenate
        m = self.merge([x_pop1_max, x_pop2_max])
        m = self.fc1(m)
        m = self.dropout(m, training=training)
        m = self.fc2(m)
        m = self.dropout(m, training=training)
        return self.dense3(m)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)

class ThreePopModel(Model):
    """Three population model"""

    # integers for num pop1, pop2, pop3
    def __init__(self, pop1, pop2, pop3):
        super(ThreePopModel, self).__init__()

        # it is (1,5) for permutation invariance (shape is n X SNPs)
        self.conv1 = Conv2D(32, (1, 5), activation='relu')
        self.conv2 = Conv2D(64, (1, 5), activation='relu')
        self.pool = MaxPooling2D(pool_size = (1,2), strides = (1,2))

        self.flatten = Flatten()
        self.merge = Concatenate()
        self.dropout = Dropout(rate=0.5)

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.dense3 = Dense(1)#2, activation='softmax') # two classes

        self.pop1 = pop1
        self.pop2 = pop2
        self.pop3 = pop3

    def call(self, x, training=None):
        """x is the genotype matrix, dist is the SNP distances"""
        assert x.shape[1] == self.pop1 + self.pop2 + self.pop3

        # first divide into populations
        x_pop1 = x[:, :self.pop1, :, :]
        x_pop2 = x[:, self.pop1:self.pop1+self.pop2, :, :]
        x_pop3 = x[:, self.pop1+self.pop2:, :, :]

        # two conv layers for each part
        x_pop1 = self.conv1(x_pop1)
        x_pop2 = self.conv1(x_pop2)
        x_pop3 = self.conv1(x_pop3)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool
        x_pop3 = self.pool(x_pop3) # pool

        x_pop1 = self.conv2(x_pop1)
        x_pop2 = self.conv2(x_pop2)
        x_pop3 = self.conv2(x_pop3)
        x_pop1 = self.pool(x_pop1) # pool
        x_pop2 = self.pool(x_pop2) # pool
        x_pop3 = self.pool(x_pop3) # pool

        # 1 is the dimension of the individuals (changing to max)
        x_pop1 = tf.math.reduce_sum(x_pop1, axis=1)
        x_pop2 = tf.math.reduce_sum(x_pop2, axis=1)
        x_pop3 = tf.math.reduce_sum(x_pop3, axis=1)

        # flatten all
        x_pop1 = self.flatten(x_pop1)
        x_pop2 = self.flatten(x_pop2)
        x_pop3 = self.flatten(x_pop3)

        # concatenate
        m = self.merge([x_pop1, x_pop2, x_pop3])
        m = self.fc1(m)
        m = self.dropout(m, training=training)
        m = self.fc2(m)
        m = self.dropout(m, training=training)
        return self.dense3(m)

    def build_graph(self, gt_shape):
        """This is for testing, based on TF tutorials"""
        gt_shape_nobatch = gt_shape[1:]
        self.build(gt_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=gt_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)