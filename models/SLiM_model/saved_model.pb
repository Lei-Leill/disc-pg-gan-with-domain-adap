щб

П
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58нО
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

!Adam/v/one_pop_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/one_pop_model/dense_2/bias

5Adam/v/one_pop_model/dense_2/bias/Read/ReadVariableOpReadVariableOp!Adam/v/one_pop_model/dense_2/bias*
_output_shapes
:*
dtype0

!Adam/m/one_pop_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/one_pop_model/dense_2/bias

5Adam/m/one_pop_model/dense_2/bias/Read/ReadVariableOpReadVariableOp!Adam/m/one_pop_model/dense_2/bias*
_output_shapes
:*
dtype0
Ѓ
#Adam/v/one_pop_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/v/one_pop_model/dense_2/kernel

7Adam/v/one_pop_model/dense_2/kernel/Read/ReadVariableOpReadVariableOp#Adam/v/one_pop_model/dense_2/kernel*
_output_shapes
:	*
dtype0
Ѓ
#Adam/m/one_pop_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#Adam/m/one_pop_model/dense_2/kernel

7Adam/m/one_pop_model/dense_2/kernel/Read/ReadVariableOpReadVariableOp#Adam/m/one_pop_model/dense_2/kernel*
_output_shapes
:	*
dtype0

!Adam/v/one_pop_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/v/one_pop_model/dense_1/bias

5Adam/v/one_pop_model/dense_1/bias/Read/ReadVariableOpReadVariableOp!Adam/v/one_pop_model/dense_1/bias*
_output_shapes	
:*
dtype0

!Adam/m/one_pop_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/m/one_pop_model/dense_1/bias

5Adam/m/one_pop_model/dense_1/bias/Read/ReadVariableOpReadVariableOp!Adam/m/one_pop_model/dense_1/bias*
_output_shapes	
:*
dtype0
Є
#Adam/v/one_pop_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/v/one_pop_model/dense_1/kernel

7Adam/v/one_pop_model/dense_1/kernel/Read/ReadVariableOpReadVariableOp#Adam/v/one_pop_model/dense_1/kernel* 
_output_shapes
:
*
dtype0
Є
#Adam/m/one_pop_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/m/one_pop_model/dense_1/kernel

7Adam/m/one_pop_model/dense_1/kernel/Read/ReadVariableOpReadVariableOp#Adam/m/one_pop_model/dense_1/kernel* 
_output_shapes
:
*
dtype0

Adam/v/one_pop_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/v/one_pop_model/dense/bias

3Adam/v/one_pop_model/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/one_pop_model/dense/bias*
_output_shapes	
:*
dtype0

Adam/m/one_pop_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/m/one_pop_model/dense/bias

3Adam/m/one_pop_model/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/one_pop_model/dense/bias*
_output_shapes	
:*
dtype0
 
!Adam/v/one_pop_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!Adam/v/one_pop_model/dense/kernel

5Adam/v/one_pop_model/dense/kernel/Read/ReadVariableOpReadVariableOp!Adam/v/one_pop_model/dense/kernel* 
_output_shapes
:
*
dtype0
 
!Adam/m/one_pop_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!Adam/m/one_pop_model/dense/kernel

5Adam/m/one_pop_model/dense/kernel/Read/ReadVariableOpReadVariableOp!Adam/m/one_pop_model/dense/kernel* 
_output_shapes
:
*
dtype0

"Adam/v/one_pop_model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/one_pop_model/conv2d_1/bias

6Adam/v/one_pop_model/conv2d_1/bias/Read/ReadVariableOpReadVariableOp"Adam/v/one_pop_model/conv2d_1/bias*
_output_shapes
:@*
dtype0

"Adam/m/one_pop_model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/one_pop_model/conv2d_1/bias

6Adam/m/one_pop_model/conv2d_1/bias/Read/ReadVariableOpReadVariableOp"Adam/m/one_pop_model/conv2d_1/bias*
_output_shapes
:@*
dtype0
Ќ
$Adam/v/one_pop_model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*5
shared_name&$Adam/v/one_pop_model/conv2d_1/kernel
Ѕ
8Adam/v/one_pop_model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/one_pop_model/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
Ќ
$Adam/m/one_pop_model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*5
shared_name&$Adam/m/one_pop_model/conv2d_1/kernel
Ѕ
8Adam/m/one_pop_model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/one_pop_model/conv2d_1/kernel*&
_output_shapes
: @*
dtype0

 Adam/v/one_pop_model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/v/one_pop_model/conv2d/bias

4Adam/v/one_pop_model/conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/v/one_pop_model/conv2d/bias*
_output_shapes
: *
dtype0

 Adam/m/one_pop_model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/m/one_pop_model/conv2d/bias

4Adam/m/one_pop_model/conv2d/bias/Read/ReadVariableOpReadVariableOp Adam/m/one_pop_model/conv2d/bias*
_output_shapes
: *
dtype0
Ј
"Adam/v/one_pop_model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/one_pop_model/conv2d/kernel
Ё
6Adam/v/one_pop_model/conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/one_pop_model/conv2d/kernel*&
_output_shapes
: *
dtype0
Ј
"Adam/m/one_pop_model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/one_pop_model/conv2d/kernel
Ё
6Adam/m/one_pop_model/conv2d/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/one_pop_model/conv2d/kernel*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

one_pop_model/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameone_pop_model/dense_2/bias

.one_pop_model/dense_2/bias/Read/ReadVariableOpReadVariableOpone_pop_model/dense_2/bias*
_output_shapes
:*
dtype0

one_pop_model/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameone_pop_model/dense_2/kernel

0one_pop_model/dense_2/kernel/Read/ReadVariableOpReadVariableOpone_pop_model/dense_2/kernel*
_output_shapes
:	*
dtype0

one_pop_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameone_pop_model/dense_1/bias

.one_pop_model/dense_1/bias/Read/ReadVariableOpReadVariableOpone_pop_model/dense_1/bias*
_output_shapes	
:*
dtype0

one_pop_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameone_pop_model/dense_1/kernel

0one_pop_model/dense_1/kernel/Read/ReadVariableOpReadVariableOpone_pop_model/dense_1/kernel* 
_output_shapes
:
*
dtype0

one_pop_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameone_pop_model/dense/bias

,one_pop_model/dense/bias/Read/ReadVariableOpReadVariableOpone_pop_model/dense/bias*
_output_shapes	
:*
dtype0

one_pop_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_nameone_pop_model/dense/kernel

.one_pop_model/dense/kernel/Read/ReadVariableOpReadVariableOpone_pop_model/dense/kernel* 
_output_shapes
:
*
dtype0

one_pop_model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameone_pop_model/conv2d_1/bias

/one_pop_model/conv2d_1/bias/Read/ReadVariableOpReadVariableOpone_pop_model/conv2d_1/bias*
_output_shapes
:@*
dtype0

one_pop_model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_nameone_pop_model/conv2d_1/kernel

1one_pop_model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpone_pop_model/conv2d_1/kernel*&
_output_shapes
: @*
dtype0

one_pop_model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameone_pop_model/conv2d/bias

-one_pop_model/conv2d/bias/Read/ReadVariableOpReadVariableOpone_pop_model/conv2d/bias*
_output_shapes
: *
dtype0

one_pop_model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameone_pop_model/conv2d/kernel

/one_pop_model/conv2d/kernel/Read/ReadVariableOpReadVariableOpone_pop_model/conv2d/kernel*&
_output_shapes
: *
dtype0

serving_default_input_1Placeholder*0
_output_shapes
:џџџџџџџџџЦ$*
dtype0*%
shape:џџџџџџџџџЦ$
ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1one_pop_model/conv2d/kernelone_pop_model/conv2d/biasone_pop_model/conv2d_1/kernelone_pop_model/conv2d_1/biasone_pop_model/dense/kernelone_pop_model/dense/biasone_pop_model/dense_1/kernelone_pop_model/dense_1/biasone_pop_model/dense_2/kernelone_pop_model/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_10499

NoOpNoOp
O
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*СN
valueЗNBДN B­N
Є
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2

pool
flatten
dropout
fc1
fc2

dense3
	optimizer

signatures*
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
А
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
!trace_0
"trace_1
#trace_2
$trace_3* 
6
%trace_0
&trace_1
'trace_2
(trace_3* 
* 
Ш
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias
 /_jit_compiled_convolution_op*
Ш
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias
 6_jit_compiled_convolution_op*

7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
Ѕ
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator* 
І
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias*
І
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

kernel
bias*
І
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

kernel
bias*

\
_variables
]_iterations
^_learning_rate
__index_dict
`
_momentums
a_velocities
b_update_step_xla*

cserving_default* 
[U
VARIABLE_VALUEone_pop_model/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEone_pop_model/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEone_pop_model/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEone_pop_model/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEone_pop_model/dense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEone_pop_model/dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEone_pop_model/dense_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEone_pop_model/dense_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEone_pop_model/dense_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEone_pop_model/dense_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
0
	1

2
3
4
5
6
7*

d0
e1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
* 

0
1*

0
1*
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

rtrace_0* 

strace_0* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

ytrace_0* 

ztrace_0* 
* 
* 
* 

{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

trace_0* 

trace_0* 
Ж
]0
 1
Ё2
Ђ3
Ѓ4
Є5
Ѕ6
І7
Ї8
Ј9
Љ10
Њ11
Ћ12
Ќ13
­14
Ў15
Џ16
А17
Б18
В19
Г20*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
T
 0
Ђ1
Є2
І3
Ј4
Њ5
Ќ6
Ў7
А8
В9*
T
Ё0
Ѓ1
Ѕ2
Ї3
Љ4
Ћ5
­6
Џ7
Б8
Г9*

Дtrace_0
Еtrace_1
Жtrace_2
Зtrace_3
Иtrace_4
Йtrace_5
Кtrace_6
Лtrace_7
Мtrace_8
Нtrace_9* 
* 
<
О	variables
П	keras_api

Рtotal

Сcount*
M
Т	variables
У	keras_api

Фtotal

Хcount
Ц
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
mg
VARIABLE_VALUE"Adam/m/one_pop_model/conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/one_pop_model/conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/one_pop_model/conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/one_pop_model/conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/one_pop_model/conv2d_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/one_pop_model/conv2d_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/one_pop_model/conv2d_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/one_pop_model/conv2d_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/one_pop_model/dense/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/one_pop_model/dense/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/one_pop_model/dense/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/one_pop_model/dense/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/one_pop_model/dense_1/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/one_pop_model/dense_1/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/one_pop_model/dense_1/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/one_pop_model/dense_1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/one_pop_model/dense_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/one_pop_model/dense_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/one_pop_model/dense_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/one_pop_model/dense_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Р0
С1*

О	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ф0
Х1*

Т	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Б
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/one_pop_model/conv2d/kernel/Read/ReadVariableOp-one_pop_model/conv2d/bias/Read/ReadVariableOp1one_pop_model/conv2d_1/kernel/Read/ReadVariableOp/one_pop_model/conv2d_1/bias/Read/ReadVariableOp.one_pop_model/dense/kernel/Read/ReadVariableOp,one_pop_model/dense/bias/Read/ReadVariableOp0one_pop_model/dense_1/kernel/Read/ReadVariableOp.one_pop_model/dense_1/bias/Read/ReadVariableOp0one_pop_model/dense_2/kernel/Read/ReadVariableOp.one_pop_model/dense_2/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp6Adam/m/one_pop_model/conv2d/kernel/Read/ReadVariableOp6Adam/v/one_pop_model/conv2d/kernel/Read/ReadVariableOp4Adam/m/one_pop_model/conv2d/bias/Read/ReadVariableOp4Adam/v/one_pop_model/conv2d/bias/Read/ReadVariableOp8Adam/m/one_pop_model/conv2d_1/kernel/Read/ReadVariableOp8Adam/v/one_pop_model/conv2d_1/kernel/Read/ReadVariableOp6Adam/m/one_pop_model/conv2d_1/bias/Read/ReadVariableOp6Adam/v/one_pop_model/conv2d_1/bias/Read/ReadVariableOp5Adam/m/one_pop_model/dense/kernel/Read/ReadVariableOp5Adam/v/one_pop_model/dense/kernel/Read/ReadVariableOp3Adam/m/one_pop_model/dense/bias/Read/ReadVariableOp3Adam/v/one_pop_model/dense/bias/Read/ReadVariableOp7Adam/m/one_pop_model/dense_1/kernel/Read/ReadVariableOp7Adam/v/one_pop_model/dense_1/kernel/Read/ReadVariableOp5Adam/m/one_pop_model/dense_1/bias/Read/ReadVariableOp5Adam/v/one_pop_model/dense_1/bias/Read/ReadVariableOp7Adam/m/one_pop_model/dense_2/kernel/Read/ReadVariableOp7Adam/v/one_pop_model/dense_2/kernel/Read/ReadVariableOp5Adam/m/one_pop_model/dense_2/bias/Read/ReadVariableOp5Adam/v/one_pop_model/dense_2/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*1
Tin*
(2&	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_10986
м

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameone_pop_model/conv2d/kernelone_pop_model/conv2d/biasone_pop_model/conv2d_1/kernelone_pop_model/conv2d_1/biasone_pop_model/dense/kernelone_pop_model/dense/biasone_pop_model/dense_1/kernelone_pop_model/dense_1/biasone_pop_model/dense_2/kernelone_pop_model/dense_2/bias	iterationlearning_rate"Adam/m/one_pop_model/conv2d/kernel"Adam/v/one_pop_model/conv2d/kernel Adam/m/one_pop_model/conv2d/bias Adam/v/one_pop_model/conv2d/bias$Adam/m/one_pop_model/conv2d_1/kernel$Adam/v/one_pop_model/conv2d_1/kernel"Adam/m/one_pop_model/conv2d_1/bias"Adam/v/one_pop_model/conv2d_1/bias!Adam/m/one_pop_model/dense/kernel!Adam/v/one_pop_model/dense/kernelAdam/m/one_pop_model/dense/biasAdam/v/one_pop_model/dense/bias#Adam/m/one_pop_model/dense_1/kernel#Adam/v/one_pop_model/dense_1/kernel!Adam/m/one_pop_model/dense_1/bias!Adam/v/one_pop_model/dense_1/bias#Adam/m/one_pop_model/dense_2/kernel#Adam/v/one_pop_model/dense_2/kernel!Adam/m/one_pop_model/dense_2/bias!Adam/v/one_pop_model/dense_2/biastotal_1count_1totalcount*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_11104ќ
D
ы
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10657
x?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	9
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  *
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  g
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  Љ
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*0
_output_shapes
:џџџџџџџџџЦ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Х
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@­
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*0
_output_shapes
:џџџџџџџџџЦ@*
ksize
*
paddingVALID*
strides
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
SumSum max_pooling2d/MaxPool_1:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  s
flatten/ReshapeReshapeSum:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?П
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Д
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ\
dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout/dropout_1/MulMuldense_1/Relu:activations:0 dropout/dropout_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџa
dropout/dropout_1/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:Ё
.dropout/dropout_1/random_uniform/RandomUniformRandomUniform dropout/dropout_1/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0e
 dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Х
dropout/dropout_1/GreaterEqualGreaterEqual7dropout/dropout_1/random_uniform/RandomUniform:output:0)dropout/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ^
dropout/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    М
dropout/dropout_1/SelectV2SelectV2"dropout/dropout_1/GreaterEqual:z:0dropout/dropout_1/Mul:z:0"dropout/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_2/MatMulMatMul#dropout/dropout_1/SelectV2:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
0
_output_shapes
:џџџџџџџџџЦ$

_user_specified_namex
 
C
'__inference_dropout_layer_call_fn_10773

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10147a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
є

!__inference__traced_restore_11104
file_prefixF
,assignvariableop_one_pop_model_conv2d_kernel: :
,assignvariableop_1_one_pop_model_conv2d_bias: J
0assignvariableop_2_one_pop_model_conv2d_1_kernel: @<
.assignvariableop_3_one_pop_model_conv2d_1_bias:@A
-assignvariableop_4_one_pop_model_dense_kernel:
:
+assignvariableop_5_one_pop_model_dense_bias:	C
/assignvariableop_6_one_pop_model_dense_1_kernel:
<
-assignvariableop_7_one_pop_model_dense_1_bias:	B
/assignvariableop_8_one_pop_model_dense_2_kernel:	;
-assignvariableop_9_one_pop_model_dense_2_bias:'
assignvariableop_10_iteration:	 +
!assignvariableop_11_learning_rate: P
6assignvariableop_12_adam_m_one_pop_model_conv2d_kernel: P
6assignvariableop_13_adam_v_one_pop_model_conv2d_kernel: B
4assignvariableop_14_adam_m_one_pop_model_conv2d_bias: B
4assignvariableop_15_adam_v_one_pop_model_conv2d_bias: R
8assignvariableop_16_adam_m_one_pop_model_conv2d_1_kernel: @R
8assignvariableop_17_adam_v_one_pop_model_conv2d_1_kernel: @D
6assignvariableop_18_adam_m_one_pop_model_conv2d_1_bias:@D
6assignvariableop_19_adam_v_one_pop_model_conv2d_1_bias:@I
5assignvariableop_20_adam_m_one_pop_model_dense_kernel:
I
5assignvariableop_21_adam_v_one_pop_model_dense_kernel:
B
3assignvariableop_22_adam_m_one_pop_model_dense_bias:	B
3assignvariableop_23_adam_v_one_pop_model_dense_bias:	K
7assignvariableop_24_adam_m_one_pop_model_dense_1_kernel:
K
7assignvariableop_25_adam_v_one_pop_model_dense_1_kernel:
D
5assignvariableop_26_adam_m_one_pop_model_dense_1_bias:	D
5assignvariableop_27_adam_v_one_pop_model_dense_1_bias:	J
7assignvariableop_28_adam_m_one_pop_model_dense_2_kernel:	J
7assignvariableop_29_adam_v_one_pop_model_dense_2_kernel:	C
5assignvariableop_30_adam_m_one_pop_model_dense_2_bias:C
5assignvariableop_31_adam_v_one_pop_model_dense_2_bias:%
assignvariableop_32_total_1: %
assignvariableop_33_count_1: #
assignvariableop_34_total: #
assignvariableop_35_count: 
identity_37ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9у
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueџBќ%B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHК
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B к
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Њ
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOpAssignVariableOp,assignvariableop_one_pop_model_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_1AssignVariableOp,assignvariableop_1_one_pop_model_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_2AssignVariableOp0assignvariableop_2_one_pop_model_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_3AssignVariableOp.assignvariableop_3_one_pop_model_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_4AssignVariableOp-assignvariableop_4_one_pop_model_dense_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_5AssignVariableOp+assignvariableop_5_one_pop_model_dense_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_6AssignVariableOp/assignvariableop_6_one_pop_model_dense_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_7AssignVariableOp-assignvariableop_7_one_pop_model_dense_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_8AssignVariableOp/assignvariableop_8_one_pop_model_dense_2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_9AssignVariableOp-assignvariableop_9_one_pop_model_dense_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_10AssignVariableOpassignvariableop_10_iterationIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_12AssignVariableOp6assignvariableop_12_adam_m_one_pop_model_conv2d_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_13AssignVariableOp6assignvariableop_13_adam_v_one_pop_model_conv2d_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_m_one_pop_model_conv2d_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_15AssignVariableOp4assignvariableop_15_adam_v_one_pop_model_conv2d_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_16AssignVariableOp8assignvariableop_16_adam_m_one_pop_model_conv2d_1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_v_one_pop_model_conv2d_1_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_m_one_pop_model_conv2d_1_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_v_one_pop_model_conv2d_1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_m_one_pop_model_dense_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_v_one_pop_model_dense_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_m_one_pop_model_dense_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_v_one_pop_model_dense_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_24AssignVariableOp7assignvariableop_24_adam_m_one_pop_model_dense_1_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_v_one_pop_model_dense_1_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_m_one_pop_model_dense_1_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_v_one_pop_model_dense_1_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_28AssignVariableOp7assignvariableop_28_adam_m_one_pop_model_dense_2_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_v_one_pop_model_dense_2_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_m_one_pop_model_dense_2_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_31AssignVariableOp5assignvariableop_31_adam_v_one_pop_model_dense_2_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_34AssignVariableOpassignvariableop_34_totalIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_35AssignVariableOpassignvariableop_35_countIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ч
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: д
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


a
B__inference_dropout_layer_call_and_return_conditional_losses_10238

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_10707
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ђ

є
B__inference_dense_2_layer_call_and_return_conditional_losses_10178

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
C
'__inference_flatten_layer_call_fn_10762

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10123a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ш

'__inference_dense_1_layer_call_fn_10824

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10160p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
K
"__inference__update_step_xla_10697
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
й
`
B__inference_dropout_layer_call_and_return_conditional_losses_10147

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ

є
@__inference_dense_layer_call_and_return_conditional_losses_10815

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ-

H__inference_one_pop_model_layer_call_and_return_conditional_losses_10350
x&
conv2d_10317: 
conv2d_10319: (
conv2d_1_10323: @
conv2d_1_10325:@
dense_10332:

dense_10334:	!
dense_1_10338:

dense_1_10340:	 
dense_2_10344:	
dense_2_10346:
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout/StatefulPartitionedCall_1ь
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_10317conv2d_10319*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_10090ю
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_10323conv2d_1_10325*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10108ђ
max_pooling2d/PartitionedCall_1PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
SumSum(max_pooling2d/PartitionedCall_1:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@П
flatten/PartitionedCallPartitionedCallSum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10123џ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_10332dense_10334*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10136щ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10238
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_10338dense_1_10340*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10160
!dropout/StatefulPartitionedCall_1StatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10238
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_1:output:0dense_2_10344dense_2_10346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_10178w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџД
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout/StatefulPartitionedCall_1!dropout/StatefulPartitionedCall_1:S O
0
_output_shapes
:џџџџџџџџџЦ$

_user_specified_namex
Ў
K
"__inference__update_step_xla_10687
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:E A

_output_shapes	
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
я

(__inference_conv2d_1_layer_call_fn_10736

inputs!
unknown: @
	unknown_0:@
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10108x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЦ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџЦ : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџЦ 
 
_user_specified_nameinputs
Ћ
J
"__inference__update_step_xla_10667
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ћ
J
"__inference__update_step_xla_10677
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ЫB
с	
 __inference__wrapped_model_10060
input_1M
3one_pop_model_conv2d_conv2d_readvariableop_resource: B
4one_pop_model_conv2d_biasadd_readvariableop_resource: O
5one_pop_model_conv2d_1_conv2d_readvariableop_resource: @D
6one_pop_model_conv2d_1_biasadd_readvariableop_resource:@F
2one_pop_model_dense_matmul_readvariableop_resource:
B
3one_pop_model_dense_biasadd_readvariableop_resource:	H
4one_pop_model_dense_1_matmul_readvariableop_resource:
D
5one_pop_model_dense_1_biasadd_readvariableop_resource:	G
4one_pop_model_dense_2_matmul_readvariableop_resource:	C
5one_pop_model_dense_2_biasadd_readvariableop_resource:
identityЂ+one_pop_model/conv2d/BiasAdd/ReadVariableOpЂ*one_pop_model/conv2d/Conv2D/ReadVariableOpЂ-one_pop_model/conv2d_1/BiasAdd/ReadVariableOpЂ,one_pop_model/conv2d_1/Conv2D/ReadVariableOpЂ*one_pop_model/dense/BiasAdd/ReadVariableOpЂ)one_pop_model/dense/MatMul/ReadVariableOpЂ,one_pop_model/dense_1/BiasAdd/ReadVariableOpЂ+one_pop_model/dense_1/MatMul/ReadVariableOpЂ,one_pop_model/dense_2/BiasAdd/ReadVariableOpЂ+one_pop_model/dense_2/MatMul/ReadVariableOpІ
*one_pop_model/conv2d/Conv2D/ReadVariableOpReadVariableOp3one_pop_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ц
one_pop_model/conv2d/Conv2DConv2Dinput_12one_pop_model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  *
paddingVALID*
strides

+one_pop_model/conv2d/BiasAdd/ReadVariableOpReadVariableOp4one_pop_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Н
one_pop_model/conv2d/BiasAddBiasAdd$one_pop_model/conv2d/Conv2D:output:03one_pop_model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  
one_pop_model/conv2d/ReluRelu%one_pop_model/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  Х
#one_pop_model/max_pooling2d/MaxPoolMaxPool'one_pop_model/conv2d/Relu:activations:0*0
_output_shapes
:џџџџџџџџџЦ *
ksize
*
paddingVALID*
strides
Њ
,one_pop_model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5one_pop_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0я
one_pop_model/conv2d_1/Conv2DConv2D,one_pop_model/max_pooling2d/MaxPool:output:04one_pop_model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@*
paddingVALID*
strides
 
-one_pop_model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6one_pop_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
one_pop_model/conv2d_1/BiasAddBiasAdd&one_pop_model/conv2d_1/Conv2D:output:05one_pop_model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@
one_pop_model/conv2d_1/ReluRelu'one_pop_model/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@Щ
%one_pop_model/max_pooling2d/MaxPool_1MaxPool)one_pop_model/conv2d_1/Relu:activations:0*0
_output_shapes
:џџџџџџџџџЦ@*
ksize
*
paddingVALID*
strides
e
#one_pop_model/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ќ
one_pop_model/SumSum.one_pop_model/max_pooling2d/MaxPool_1:output:0,one_pop_model/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@l
one_pop_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
one_pop_model/flatten/ReshapeReshapeone_pop_model/Sum:output:0$one_pop_model/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
)one_pop_model/dense/MatMul/ReadVariableOpReadVariableOp2one_pop_model_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0В
one_pop_model/dense/MatMulMatMul&one_pop_model/flatten/Reshape:output:01one_pop_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
*one_pop_model/dense/BiasAdd/ReadVariableOpReadVariableOp3one_pop_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Г
one_pop_model/dense/BiasAddBiasAdd$one_pop_model/dense/MatMul:product:02one_pop_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџy
one_pop_model/dense/ReluRelu$one_pop_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
one_pop_model/dropout/IdentityIdentity&one_pop_model/dense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЂ
+one_pop_model/dense_1/MatMul/ReadVariableOpReadVariableOp4one_pop_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0З
one_pop_model/dense_1/MatMulMatMul'one_pop_model/dropout/Identity:output:03one_pop_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
,one_pop_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp5one_pop_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Й
one_pop_model/dense_1/BiasAddBiasAdd&one_pop_model/dense_1/MatMul:product:04one_pop_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ}
one_pop_model/dense_1/ReluRelu&one_pop_model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
 one_pop_model/dropout/Identity_1Identity(one_pop_model/dense_1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
+one_pop_model/dense_2/MatMul/ReadVariableOpReadVariableOp4one_pop_model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0И
one_pop_model/dense_2/MatMulMatMul)one_pop_model/dropout/Identity_1:output:03one_pop_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
,one_pop_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp5one_pop_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
one_pop_model/dense_2/BiasAddBiasAdd&one_pop_model/dense_2/MatMul:product:04one_pop_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
one_pop_model/dense_2/SoftmaxSoftmax&one_pop_model/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
IdentityIdentity'one_pop_model/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp,^one_pop_model/conv2d/BiasAdd/ReadVariableOp+^one_pop_model/conv2d/Conv2D/ReadVariableOp.^one_pop_model/conv2d_1/BiasAdd/ReadVariableOp-^one_pop_model/conv2d_1/Conv2D/ReadVariableOp+^one_pop_model/dense/BiasAdd/ReadVariableOp*^one_pop_model/dense/MatMul/ReadVariableOp-^one_pop_model/dense_1/BiasAdd/ReadVariableOp,^one_pop_model/dense_1/MatMul/ReadVariableOp-^one_pop_model/dense_2/BiasAdd/ReadVariableOp,^one_pop_model/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 2Z
+one_pop_model/conv2d/BiasAdd/ReadVariableOp+one_pop_model/conv2d/BiasAdd/ReadVariableOp2X
*one_pop_model/conv2d/Conv2D/ReadVariableOp*one_pop_model/conv2d/Conv2D/ReadVariableOp2^
-one_pop_model/conv2d_1/BiasAdd/ReadVariableOp-one_pop_model/conv2d_1/BiasAdd/ReadVariableOp2\
,one_pop_model/conv2d_1/Conv2D/ReadVariableOp,one_pop_model/conv2d_1/Conv2D/ReadVariableOp2X
*one_pop_model/dense/BiasAdd/ReadVariableOp*one_pop_model/dense/BiasAdd/ReadVariableOp2V
)one_pop_model/dense/MatMul/ReadVariableOp)one_pop_model/dense/MatMul/ReadVariableOp2\
,one_pop_model/dense_1/BiasAdd/ReadVariableOp,one_pop_model/dense_1/BiasAdd/ReadVariableOp2Z
+one_pop_model/dense_1/MatMul/ReadVariableOp+one_pop_model/dense_1/MatMul/ReadVariableOp2\
,one_pop_model/dense_2/BiasAdd/ReadVariableOp,one_pop_model/dense_2/BiasAdd/ReadVariableOp2Z
+one_pop_model/dense_2/MatMul/ReadVariableOp+one_pop_model/dense_2/MatMul/ReadVariableOp:Y U
0
_output_shapes
:џџџџџџџџџЦ$
!
_user_specified_name	input_1
Е
I
-__inference_max_pooling2d_layer_call_fn_10752

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_10672
gradient"
variable: @*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: @: *
	_noinline(:P L
&
_output_shapes
: @
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10757

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10108

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЦ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџЦ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџЦ 
 
_user_specified_nameinputs
К
O
"__inference__update_step_xla_10702
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	: *
	_noinline(:I E

_output_shapes
:	
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Э
P
"__inference__update_step_xla_10682
gradient
variable:
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:џџџџџџџџџ: *
	_noinline(:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
і4
ы
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10596
x?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	:
&dense_1_matmul_readvariableop_resource:
6
'dense_1_biasadd_readvariableop_resource:	9
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:
identityЂconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Є
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  *
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  g
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  Љ
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*0
_output_shapes
:џџџџџџџџџЦ *
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Х
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@­
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*0
_output_shapes
:џџџџџџџџџЦ@*
ksize
*
paddingVALID*
strides
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
SumSum max_pooling2d/MaxPool_1:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  s
flatten/ReshapeReshapeSum:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџi
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџa
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџm
dropout/Identity_1Identitydense_1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_2/MatMulMatMuldropout/Identity_1:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
0
_output_shapes
:џџџџџџџџџЦ$

_user_specified_namex
З


-__inference_one_pop_model_layer_call_fn_10549
x!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:џџџџџџџџџЦ$

_user_specified_namex
ы

&__inference_conv2d_layer_call_fn_10716

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_10090x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЦ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџЦ$: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџЦ$
 
_user_specified_nameinputs

ќ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10747

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ@j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЦ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџЦ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџЦ 
 
_user_specified_nameinputs
М
^
B__inference_flatten_layer_call_and_return_conditional_losses_10123

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
М
^
B__inference_flatten_layer_call_and_return_conditional_losses_10768

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ђ

є
B__inference_dense_2_layer_call_and_return_conditional_losses_10855

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф

'__inference_dense_2_layer_call_fn_10844

inputs
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_10178o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѕ

і
B__inference_dense_1_layer_call_and_return_conditional_losses_10160

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я
V
"__inference__update_step_xla_10662
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Щ


-__inference_one_pop_model_layer_call_fn_10208
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10185o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџЦ$
!
_user_specified_name	input_1

њ
A__inference_conv2d_layer_call_and_return_conditional_losses_10090

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЦ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџЦ$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџЦ$
 
_user_specified_nameinputs
Ф

%__inference_dense_layer_call_fn_10804

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10136p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё*
й
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10434
input_1&
conv2d_10401: 
conv2d_10403: (
conv2d_1_10407: @
conv2d_1_10409:@
dense_10416:

dense_10418:	!
dense_1_10422:

dense_1_10424:	 
dense_2_10428:	
dense_2_10430:
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallђ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_10401conv2d_10403*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_10090ю
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_10407conv2d_1_10409*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10108ђ
max_pooling2d/PartitionedCall_1PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
SumSum(max_pooling2d/PartitionedCall_1:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@П
flatten/PartitionedCallPartitionedCallSum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10123џ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_10416dense_10418*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10136й
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10147
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_10422dense_1_10424*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10160н
dropout/PartitionedCall_1PartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10147
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_1:output:0dense_2_10428dense_2_10430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_10178w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџю
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџЦ$
!
_user_specified_name	input_1
Н
P
"__inference__update_step_xla_10692
gradient
variable:
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: *
	_noinline(:J F
 
_output_shapes
:

"
_user_specified_name
gradient:($
"
_user_specified_name
variable
п*
г
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10185
x&
conv2d_10091: 
conv2d_10093: (
conv2d_1_10109: @
conv2d_1_10111:@
dense_10137:

dense_10139:	!
dense_1_10161:

dense_1_10163:	 
dense_2_10179:	
dense_2_10181:
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallь
conv2d/StatefulPartitionedCallStatefulPartitionedCallxconv2d_10091conv2d_10093*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_10090ю
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_10109conv2d_1_10111*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10108ђ
max_pooling2d/PartitionedCall_1PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
SumSum(max_pooling2d/PartitionedCall_1:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@П
flatten/PartitionedCallPartitionedCallSum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10123џ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_10137dense_10139*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10136й
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10147
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_10161dense_1_10163*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10160н
dropout/PartitionedCall_1PartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10147
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout/PartitionedCall_1:output:0dense_2_10179dense_2_10181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_10178w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџю
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
0
_output_shapes
:џџџџџџџџџЦ$

_user_specified_namex
Щ


-__inference_one_pop_model_layer_call_fn_10398
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10350o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџЦ$
!
_user_specified_name	input_1

њ
A__inference_conv2d_layer_call_and_return_conditional_losses_10727

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЦ  j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЦ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџЦ$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџЦ$
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_10795

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
N
ч
__inference__traced_save_10986
file_prefix:
6savev2_one_pop_model_conv2d_kernel_read_readvariableop8
4savev2_one_pop_model_conv2d_bias_read_readvariableop<
8savev2_one_pop_model_conv2d_1_kernel_read_readvariableop:
6savev2_one_pop_model_conv2d_1_bias_read_readvariableop9
5savev2_one_pop_model_dense_kernel_read_readvariableop7
3savev2_one_pop_model_dense_bias_read_readvariableop;
7savev2_one_pop_model_dense_1_kernel_read_readvariableop9
5savev2_one_pop_model_dense_1_bias_read_readvariableop;
7savev2_one_pop_model_dense_2_kernel_read_readvariableop9
5savev2_one_pop_model_dense_2_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopA
=savev2_adam_m_one_pop_model_conv2d_kernel_read_readvariableopA
=savev2_adam_v_one_pop_model_conv2d_kernel_read_readvariableop?
;savev2_adam_m_one_pop_model_conv2d_bias_read_readvariableop?
;savev2_adam_v_one_pop_model_conv2d_bias_read_readvariableopC
?savev2_adam_m_one_pop_model_conv2d_1_kernel_read_readvariableopC
?savev2_adam_v_one_pop_model_conv2d_1_kernel_read_readvariableopA
=savev2_adam_m_one_pop_model_conv2d_1_bias_read_readvariableopA
=savev2_adam_v_one_pop_model_conv2d_1_bias_read_readvariableop@
<savev2_adam_m_one_pop_model_dense_kernel_read_readvariableop@
<savev2_adam_v_one_pop_model_dense_kernel_read_readvariableop>
:savev2_adam_m_one_pop_model_dense_bias_read_readvariableop>
:savev2_adam_v_one_pop_model_dense_bias_read_readvariableopB
>savev2_adam_m_one_pop_model_dense_1_kernel_read_readvariableopB
>savev2_adam_v_one_pop_model_dense_1_kernel_read_readvariableop@
<savev2_adam_m_one_pop_model_dense_1_bias_read_readvariableop@
<savev2_adam_v_one_pop_model_dense_1_bias_read_readvariableopB
>savev2_adam_m_one_pop_model_dense_2_kernel_read_readvariableopB
>savev2_adam_v_one_pop_model_dense_2_kernel_read_readvariableop@
<savev2_adam_m_one_pop_model_dense_2_bias_read_readvariableop@
<savev2_adam_v_one_pop_model_dense_2_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: р
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*
valueџBќ%B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЗ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ш
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_one_pop_model_conv2d_kernel_read_readvariableop4savev2_one_pop_model_conv2d_bias_read_readvariableop8savev2_one_pop_model_conv2d_1_kernel_read_readvariableop6savev2_one_pop_model_conv2d_1_bias_read_readvariableop5savev2_one_pop_model_dense_kernel_read_readvariableop3savev2_one_pop_model_dense_bias_read_readvariableop7savev2_one_pop_model_dense_1_kernel_read_readvariableop5savev2_one_pop_model_dense_1_bias_read_readvariableop7savev2_one_pop_model_dense_2_kernel_read_readvariableop5savev2_one_pop_model_dense_2_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop=savev2_adam_m_one_pop_model_conv2d_kernel_read_readvariableop=savev2_adam_v_one_pop_model_conv2d_kernel_read_readvariableop;savev2_adam_m_one_pop_model_conv2d_bias_read_readvariableop;savev2_adam_v_one_pop_model_conv2d_bias_read_readvariableop?savev2_adam_m_one_pop_model_conv2d_1_kernel_read_readvariableop?savev2_adam_v_one_pop_model_conv2d_1_kernel_read_readvariableop=savev2_adam_m_one_pop_model_conv2d_1_bias_read_readvariableop=savev2_adam_v_one_pop_model_conv2d_1_bias_read_readvariableop<savev2_adam_m_one_pop_model_dense_kernel_read_readvariableop<savev2_adam_v_one_pop_model_dense_kernel_read_readvariableop:savev2_adam_m_one_pop_model_dense_bias_read_readvariableop:savev2_adam_v_one_pop_model_dense_bias_read_readvariableop>savev2_adam_m_one_pop_model_dense_1_kernel_read_readvariableop>savev2_adam_v_one_pop_model_dense_1_kernel_read_readvariableop<savev2_adam_m_one_pop_model_dense_1_bias_read_readvariableop<savev2_adam_v_one_pop_model_dense_1_bias_read_readvariableop>savev2_adam_m_one_pop_model_dense_2_kernel_read_readvariableop>savev2_adam_v_one_pop_model_dense_2_kernel_read_readvariableop<savev2_adam_m_one_pop_model_dense_2_bias_read_readvariableop<savev2_adam_v_one_pop_model_dense_2_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *3
dtypes)
'2%	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*к
_input_shapesШ
Х: : : : @:@:
::
::	:: : : : : : : @: @:@:@:
:
:::
:
:::	:	::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%	!

_output_shapes
:	: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @:,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	:%!

_output_shapes
:	: 

_output_shapes
::  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: 
й
`
B__inference_dropout_layer_call_and_return_conditional_losses_10783

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З


-__inference_one_pop_model_layer_call_fn_10524
x!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10185o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:џџџџџџџџџЦ$

_user_specified_namex



#__inference_signature_wrapper_10499
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:

	unknown_4:	
	unknown_5:

	unknown_6:	
	unknown_7:	
	unknown_8:
identityЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_10060o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџЦ$
!
_user_specified_name	input_1
л-

H__inference_one_pop_model_layer_call_and_return_conditional_losses_10470
input_1&
conv2d_10437: 
conv2d_10439: (
conv2d_1_10443: @
conv2d_1_10445:@
dense_10452:

dense_10454:	!
dense_1_10458:

dense_1_10460:	 
dense_2_10464:	
dense_2_10466:
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout/StatefulPartitionedCall_1ђ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_10437conv2d_10439*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_10090ю
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_10443conv2d_1_10445*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10108ђ
max_pooling2d/PartitionedCall_1PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЦ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10069W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
SumSum(max_pooling2d/PartitionedCall_1:output:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@П
flatten/PartitionedCallPartitionedCallSum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10123џ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_10452dense_10454*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10136щ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10238
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_10458dense_1_10460*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_10160
!dropout/StatefulPartitionedCall_1StatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10238
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout/StatefulPartitionedCall_1:output:0dense_2_10464dense_2_10466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_10178w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџД
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџЦ$: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout/StatefulPartitionedCall_1!dropout/StatefulPartitionedCall_1:Y U
0
_output_shapes
:џџџџџџџџџЦ$
!
_user_specified_name	input_1
Ѕ

і
B__inference_dense_1_layer_call_and_return_conditional_losses_10835

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ

є
@__inference_dense_layer_call_and_return_conditional_losses_10136

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
`
'__inference_dropout_layer_call_fn_10778

inputs
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_10238p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_default 
D
input_19
serving_default_input_1:0џџџџџџџџџЦ$<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:мћ
Й
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2

pool
flatten
dropout
fc1
fc2

dense3
	optimizer

signatures"
_tf_keras_model
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
и
!trace_0
"trace_1
#trace_2
$trace_32э
-__inference_one_pop_model_layer_call_fn_10208
-__inference_one_pop_model_layer_call_fn_10524
-__inference_one_pop_model_layer_call_fn_10549
-__inference_one_pop_model_layer_call_fn_10398Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z!trace_0z"trace_1z#trace_2z$trace_3
Ф
%trace_0
&trace_1
'trace_2
(trace_32й
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10596
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10657
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10434
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10470Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z%trace_0z&trace_1z'trace_2z(trace_3
ЫBШ
 __inference__wrapped_model_10060input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
н
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

kernel
bias
 /_jit_compiled_convolution_op"
_tf_keras_layer
н
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias
 6_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
М
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator"
_tf_keras_layer
Л
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer

\
_variables
]_iterations
^_learning_rate
__index_dict
`
_momentums
a_velocities
b_update_step_xla"
experimentalOptimizer
,
cserving_default"
signature_map
5:3 2one_pop_model/conv2d/kernel
':% 2one_pop_model/conv2d/bias
7:5 @2one_pop_model/conv2d_1/kernel
):'@2one_pop_model/conv2d_1/bias
.:,
2one_pop_model/dense/kernel
':%2one_pop_model/dense/bias
0:.
2one_pop_model/dense_1/kernel
):'2one_pop_model/dense_1/bias
/:-	2one_pop_model/dense_2/kernel
(:&2one_pop_model/dense_2/bias
 "
trackable_list_wrapper
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
-__inference_one_pop_model_layer_call_fn_10208input_1"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
шBх
-__inference_one_pop_model_layer_call_fn_10524x"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
шBх
-__inference_one_pop_model_layer_call_fn_10549x"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
-__inference_one_pop_model_layer_call_fn_10398input_1"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10596x"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10657x"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10434input_1"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10470input_1"Ў
ЅВЁ
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ъ
ktrace_02Э
&__inference_conv2d_layer_call_fn_10716Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zktrace_0

ltrace_02ш
A__inference_conv2d_layer_call_and_return_conditional_losses_10727Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zltrace_0
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ь
rtrace_02Я
(__inference_conv2d_1_layer_call_fn_10736Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zrtrace_0

strace_02ъ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10747Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zstrace_0
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
ё
ytrace_02д
-__inference_max_pooling2d_layer_call_fn_10752Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zytrace_0

ztrace_02я
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10757Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_flatten_layer_call_fn_10762Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_flatten_layer_call_and_return_conditional_losses_10768Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
У
trace_0
trace_12
'__inference_dropout_layer_call_fn_10773
'__inference_dropout_layer_call_fn_10778Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
љ
trace_0
trace_12О
B__inference_dropout_layer_call_and_return_conditional_losses_10783
B__inference_dropout_layer_call_and_return_conditional_losses_10795Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
%__inference_dense_layer_call_fn_10804Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ч
@__inference_dense_layer_call_and_return_conditional_losses_10815Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_1_layer_call_fn_10824Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_dense_1_layer_call_and_return_conditional_losses_10835Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
э
trace_02Ю
'__inference_dense_2_layer_call_fn_10844Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02щ
B__inference_dense_2_layer_call_and_return_conditional_losses_10855Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
в
]0
 1
Ё2
Ђ3
Ѓ4
Є5
Ѕ6
І7
Ї8
Ј9
Љ10
Њ11
Ћ12
Ќ13
­14
Ў15
Џ16
А17
Б18
В19
Г20"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
p
 0
Ђ1
Є2
І3
Ј4
Њ5
Ќ6
Ў7
А8
В9"
trackable_list_wrapper
p
Ё0
Ѓ1
Ѕ2
Ї3
Љ4
Ћ5
­6
Џ7
Б8
Г9"
trackable_list_wrapper
П
Дtrace_0
Еtrace_1
Жtrace_2
Зtrace_3
Иtrace_4
Йtrace_5
Кtrace_6
Лtrace_7
Мtrace_8
Нtrace_92Є
"__inference__update_step_xla_10662
"__inference__update_step_xla_10667
"__inference__update_step_xla_10672
"__inference__update_step_xla_10677
"__inference__update_step_xla_10682
"__inference__update_step_xla_10687
"__inference__update_step_xla_10692
"__inference__update_step_xla_10697
"__inference__update_step_xla_10702
"__inference__update_step_xla_10707Й
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zДtrace_0zЕtrace_1zЖtrace_2zЗtrace_3zИtrace_4zЙtrace_5zКtrace_6zЛtrace_7zМtrace_8zНtrace_9
ЪBЧ
#__inference_signature_wrapper_10499input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
О	variables
П	keras_api

Рtotal

Сcount"
_tf_keras_metric
c
Т	variables
У	keras_api

Фtotal

Хcount
Ц
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
кBз
&__inference_conv2d_layer_call_fn_10716inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
A__inference_conv2d_layer_call_and_return_conditional_losses_10727inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_conv2d_1_layer_call_fn_10736inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10747inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сBо
-__inference_max_pooling2d_layer_call_fn_10752inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќBљ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10757inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_flatten_layer_call_fn_10762inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_flatten_layer_call_and_return_conditional_losses_10768inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ьBщ
'__inference_dropout_layer_call_fn_10773inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ьBщ
'__inference_dropout_layer_call_fn_10778inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_10783inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_10795inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
йBж
%__inference_dense_layer_call_fn_10804inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
єBё
@__inference_dense_layer_call_and_return_conditional_losses_10815inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_1_layer_call_fn_10824inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_1_layer_call_and_return_conditional_losses_10835inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
лBи
'__inference_dense_2_layer_call_fn_10844inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
іBѓ
B__inference_dense_2_layer_call_and_return_conditional_losses_10855inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
::8 2"Adam/m/one_pop_model/conv2d/kernel
::8 2"Adam/v/one_pop_model/conv2d/kernel
,:* 2 Adam/m/one_pop_model/conv2d/bias
,:* 2 Adam/v/one_pop_model/conv2d/bias
<:: @2$Adam/m/one_pop_model/conv2d_1/kernel
<:: @2$Adam/v/one_pop_model/conv2d_1/kernel
.:,@2"Adam/m/one_pop_model/conv2d_1/bias
.:,@2"Adam/v/one_pop_model/conv2d_1/bias
3:1
2!Adam/m/one_pop_model/dense/kernel
3:1
2!Adam/v/one_pop_model/dense/kernel
,:*2Adam/m/one_pop_model/dense/bias
,:*2Adam/v/one_pop_model/dense/bias
5:3
2#Adam/m/one_pop_model/dense_1/kernel
5:3
2#Adam/v/one_pop_model/dense_1/kernel
.:,2!Adam/m/one_pop_model/dense_1/bias
.:,2!Adam/v/one_pop_model/dense_1/bias
4:2	2#Adam/m/one_pop_model/dense_2/kernel
4:2	2#Adam/v/one_pop_model/dense_2/kernel
-:+2!Adam/m/one_pop_model/dense_2/bias
-:+2!Adam/v/one_pop_model/dense_2/bias
їBє
"__inference__update_step_xla_10662gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10667gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10672gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10677gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10682gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10687gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10692gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10697gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10702gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
"__inference__update_step_xla_10707gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Р0
С1"
trackable_list_wrapper
.
О	variables"
_generic_user_object
:  (2total
:  (2count
0
Ф0
Х1"
trackable_list_wrapper
.
Т	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЄ
"__inference__update_step_xla_10662~xЂu
nЂk
!
gradient 
<9	%Ђ"
њ 

p
` VariableSpec 
`РчуШє?
Њ "
 
"__inference__update_step_xla_10667f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
` эуШє?
Њ "
 Є
"__inference__update_step_xla_10672~xЂu
nЂk
!
gradient @
<9	%Ђ"
њ @

p
` VariableSpec 
` ЖуШє?
Њ "
 
"__inference__update_step_xla_10677f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
` ЭЖуШє?
Њ "
  
"__inference__update_step_xla_10682ztЂq
jЂg
# 
gradientџџџџџџџџџ
63	Ђ
њ


p
` VariableSpec 
`РћЖуШє?
Њ "
 
"__inference__update_step_xla_10687hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`ИуШє?
Њ "
 
"__inference__update_step_xla_10692rlЂi
bЂ_

gradient

63	Ђ
њ


p
` VariableSpec 
`рЙЙуШє?
Њ "
 
"__inference__update_step_xla_10697hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`РКуШє?
Њ "
 
"__inference__update_step_xla_10702pjЂg
`Ђ]

gradient	
52	Ђ
њ	

p
` VariableSpec 
`рЛуШє?
Њ "
 
"__inference__update_step_xla_10707f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рМКуШє?
Њ "
  
 __inference__wrapped_model_10060|
9Ђ6
/Ђ,
*'
input_1џџџџџџџџџЦ$
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџМ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10747u8Ђ5
.Ђ+
)&
inputsџџџџџџџџџЦ 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџЦ@
 
(__inference_conv2d_1_layer_call_fn_10736j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџЦ 
Њ "*'
unknownџџџџџџџџџЦ@К
A__inference_conv2d_layer_call_and_return_conditional_losses_10727u8Ђ5
.Ђ+
)&
inputsџџџџџџџџџЦ$
Њ "5Ђ2
+(
tensor_0џџџџџџџџџЦ  
 
&__inference_conv2d_layer_call_fn_10716j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџЦ$
Њ "*'
unknownџџџџџџџџџЦ  Ћ
B__inference_dense_1_layer_call_and_return_conditional_losses_10835e0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
'__inference_dense_1_layer_call_fn_10824Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџЊ
B__inference_dense_2_layer_call_and_return_conditional_losses_10855d0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
'__inference_dense_2_layer_call_fn_10844Y0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЉ
@__inference_dense_layer_call_and_return_conditional_losses_10815e0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
%__inference_dense_layer_call_fn_10804Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџЋ
B__inference_dropout_layer_call_and_return_conditional_losses_10783e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 Ћ
B__inference_dropout_layer_call_and_return_conditional_losses_10795e4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
'__inference_dropout_layer_call_fn_10773Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ ""
unknownџџџџџџџџџ
'__inference_dropout_layer_call_fn_10778Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ ""
unknownџџџџџџџџџЊ
B__inference_flatten_layer_call_and_return_conditional_losses_10768d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
'__inference_flatten_layer_call_fn_10762Y3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ ""
unknownџџџџџџџџџђ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10757ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ь
-__inference_max_pooling2d_layer_call_fn_10752RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџХ
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10434y
=Ђ:
3Ђ0
*'
input_1џџџџџџџџџЦ$
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Х
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10470y
=Ђ:
3Ђ0
*'
input_1џџџџџџџџџЦ$
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 П
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10596s
7Ђ4
-Ђ*
$!
xџџџџџџџџџЦ$
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 П
H__inference_one_pop_model_layer_call_and_return_conditional_losses_10657s
7Ђ4
-Ђ*
$!
xџџџџџџџџџЦ$
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_one_pop_model_layer_call_fn_10208n
=Ђ:
3Ђ0
*'
input_1џџџџџџџџџЦ$
p 
Њ "!
unknownџџџџџџџџџ
-__inference_one_pop_model_layer_call_fn_10398n
=Ђ:
3Ђ0
*'
input_1џџџџџџџџџЦ$
p
Њ "!
unknownџџџџџџџџџ
-__inference_one_pop_model_layer_call_fn_10524h
7Ђ4
-Ђ*
$!
xџџџџџџџџџЦ$
p 
Њ "!
unknownџџџџџџџџџ
-__inference_one_pop_model_layer_call_fn_10549h
7Ђ4
-Ђ*
$!
xџџџџџџџџџЦ$
p
Њ "!
unknownџџџџџџџџџЏ
#__inference_signature_wrapper_10499
DЂA
Ђ 
:Њ7
5
input_1*'
input_1џџџџџџџџџЦ$"3Њ0
.
output_1"
output_1џџџџџџџџџ