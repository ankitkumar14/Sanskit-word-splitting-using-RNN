¡î4
NèM
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignSub
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
ì
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
2
L2Loss
t"T
output"T"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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

ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
¥

ScatterAdd
ref"T
indices"Tindices
updates"T

output_ref"T" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring 
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
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
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Þ
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
Á
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.8.02b'v1.8.0-0-g93bc2e2072'ì1
k
inputsPlaceholder*
dtype0	*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape:ÿÿÿÿÿÿÿÿÿ
l
targetsPlaceholder*
dtype0	*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape:ÿÿÿÿÿÿÿÿÿ
w

split_cntsPlaceholder*
dtype0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*!
shape:ÿÿÿÿÿÿÿÿÿ

V
dropout_keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
b
seqlensPlaceholder*
dtype0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape:ÿÿÿÿÿÿÿÿÿ
R
learning_ratePlaceholder*
dtype0*
_output_shapes
:*
shape:

+embeddings/Initializer/random_uniform/shapeConst*
valueB"7      *
dtype0*
_output_shapes
:*
_class
loc:@embeddings

)embeddings/Initializer/random_uniform/minConst*
valueB
 *  ¿*
dtype0*
_output_shapes
: *
_class
loc:@embeddings

)embeddings/Initializer/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: *
_class
loc:@embeddings
à
3embeddings/Initializer/random_uniform/RandomUniformRandomUniform+embeddings/Initializer/random_uniform/shape*
seed2 *
_class
loc:@embeddings*
T0*
_output_shapes
:	7*
dtype0*

seed 
Æ
)embeddings/Initializer/random_uniform/subSub)embeddings/Initializer/random_uniform/max)embeddings/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@embeddings
Ù
)embeddings/Initializer/random_uniform/mulMul3embeddings/Initializer/random_uniform/RandomUniform)embeddings/Initializer/random_uniform/sub*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
Ë
%embeddings/Initializer/random_uniformAdd)embeddings/Initializer/random_uniform/mul)embeddings/Initializer/random_uniform/min*
T0*
_output_shapes
:	7*
_class
loc:@embeddings


embeddings
VariableV2*
shared_name *
_class
loc:@embeddings*
_output_shapes
:	7*
dtype0*
	container *
shape:	7
À
embeddings/AssignAssign
embeddings%embeddings/Initializer/random_uniform*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
p
embeddings/readIdentity
embeddings*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
u
embedded_inputs/axisConst*
value	B : *
dtype0*
_output_shapes
: *
_class
loc:@embeddings
Ä
embedded_inputsGatherV2embeddings/readinputsembedded_inputs/axis*
Taxis0*
Tparams0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tindices0	*
_class
loc:@embeddings
M
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 

concatConcatV2embedded_inputs
split_cntsconcat/axis*
T0*
N*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0
c
bidi_/DropoutWrapperInit/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
e
 bidi_/DropoutWrapperInit/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
e
 bidi_/DropoutWrapperInit_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
g
"bidi_/DropoutWrapperInit_1/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
"bidi_/bidirectional_rnn/fw/fw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
k
)bidi_/bidirectional_rnn/fw/fw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
k
)bidi_/bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Î
#bidi_/bidirectional_rnn/fw/fw/rangeRange)bidi_/bidirectional_rnn/fw/fw/range/start"bidi_/bidirectional_rnn/fw/fw/Rank)bidi_/bidirectional_rnn/fw/fw/range/delta*
_output_shapes
:*

Tidx0
~
-bidi_/bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
k
)bidi_/bidirectional_rnn/fw/fw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
é
$bidi_/bidirectional_rnn/fw/fw/concatConcatV2-bidi_/bidirectional_rnn/fw/fw/concat/values_0#bidi_/bidirectional_rnn/fw/fw/range)bidi_/bidirectional_rnn/fw/fw/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
§
'bidi_/bidirectional_rnn/fw/fw/transpose	Transposeconcat$bidi_/bidirectional_rnn/fw/fw/concat*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
-bidi_/bidirectional_rnn/fw/fw/sequence_lengthIdentityseqlens*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#bidi_/bidirectional_rnn/fw/fw/ShapeShape'bidi_/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
{
1bidi_/bidirectional_rnn/fw/fw/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

+bidi_/bidirectional_rnn/fw/fw/strided_sliceStridedSlice#bidi_/bidirectional_rnn/fw/fw/Shape1bidi_/bidirectional_rnn/fw/fw/strided_slice/stack3bidi_/bidirectional_rnn/fw/fw/strided_slice/stack_13bidi_/bidirectional_rnn/fw/fw/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 

Vbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Rbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims+bidi_/bidirectional_rnn/fw/fw/strided_sliceVbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

Mbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:È*
dtype0*
_output_shapes
:

Sbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Nbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Rbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsMbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ConstSbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

Sbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¿
Mbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillNbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concatSbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0

Xbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Tbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims+bidi_/bidirectional_rnn/fw/fw/strided_sliceXbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
T0*

Tdim0*
_output_shapes
:

Obidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:È*
dtype0*
_output_shapes
:

Xbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Tbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims+bidi_/bidirectional_rnn/fw/fw/strided_sliceXbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
T0*

Tdim0*
_output_shapes
:

Obidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:È*
dtype0*
_output_shapes
:

Ubidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Pbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Tbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Obidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Ubidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

Ubidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
Obidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillPbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Ubidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0

Xbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Tbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims+bidi_/bidirectional_rnn/fw/fw/strided_sliceXbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0*
_output_shapes
:

Obidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:È*
dtype0*
_output_shapes
:

%bidi_/bidirectional_rnn/fw/fw/Shape_1Shape-bidi_/bidirectional_rnn/fw/fw/sequence_length*
T0*
out_type0*
_output_shapes
:

#bidi_/bidirectional_rnn/fw/fw/stackPack+bidi_/bidirectional_rnn/fw/fw/strided_slice*

axis *
T0*
N*
_output_shapes
:

#bidi_/bidirectional_rnn/fw/fw/EqualEqual%bidi_/bidirectional_rnn/fw/fw/Shape_1#bidi_/bidirectional_rnn/fw/fw/stack*
T0*
_output_shapes
:
m
#bidi_/bidirectional_rnn/fw/fw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
§
!bidi_/bidirectional_rnn/fw/fw/AllAll#bidi_/bidirectional_rnn/fw/fw/Equal#bidi_/bidirectional_rnn/fw/fw/Const*
	keep_dims( *
_output_shapes
: *

Tidx0
¸
*bidi_/bidirectional_rnn/fw/fw/Assert/ConstConst*^
valueUBS BMExpected shape for Tensor bidi_/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 
}
,bidi_/bidirectional_rnn/fw/fw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
À
2bidi_/bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*^
valueUBS BMExpected shape for Tensor bidi_/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 

2bidi_/bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
¬
+bidi_/bidirectional_rnn/fw/fw/Assert/AssertAssert!bidi_/bidirectional_rnn/fw/fw/All2bidi_/bidirectional_rnn/fw/fw/Assert/Assert/data_0#bidi_/bidirectional_rnn/fw/fw/stack2bidi_/bidirectional_rnn/fw/fw/Assert/Assert/data_2%bidi_/bidirectional_rnn/fw/fw/Shape_1*
T
2*
	summarize
À
)bidi_/bidirectional_rnn/fw/fw/CheckSeqLenIdentity-bidi_/bidirectional_rnn/fw/fw/sequence_length,^bidi_/bidirectional_rnn/fw/fw/Assert/Assert*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%bidi_/bidirectional_rnn/fw/fw/Shape_2Shape'bidi_/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/fw/fw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/fw/fw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/fw/fw/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

-bidi_/bidirectional_rnn/fw/fw/strided_slice_1StridedSlice%bidi_/bidirectional_rnn/fw/fw/Shape_23bidi_/bidirectional_rnn/fw/fw/strided_slice_1/stack5bidi_/bidirectional_rnn/fw/fw/strided_slice_1/stack_15bidi_/bidirectional_rnn/fw/fw/strided_slice_1/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 

%bidi_/bidirectional_rnn/fw/fw/Shape_3Shape'bidi_/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/fw/fw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/fw/fw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/fw/fw/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

-bidi_/bidirectional_rnn/fw/fw/strided_slice_2StridedSlice%bidi_/bidirectional_rnn/fw/fw/Shape_33bidi_/bidirectional_rnn/fw/fw/strided_slice_2/stack5bidi_/bidirectional_rnn/fw/fw/strided_slice_2/stack_15bidi_/bidirectional_rnn/fw/fw/strided_slice_2/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
n
,bidi_/bidirectional_rnn/fw/fw/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ä
(bidi_/bidirectional_rnn/fw/fw/ExpandDims
ExpandDims-bidi_/bidirectional_rnn/fw/fw/strided_slice_2,bidi_/bidirectional_rnn/fw/fw/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
p
%bidi_/bidirectional_rnn/fw/fw/Const_1Const*
valueB:È*
dtype0*
_output_shapes
:
m
+bidi_/bidirectional_rnn/fw/fw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ê
&bidi_/bidirectional_rnn/fw/fw/concat_1ConcatV2(bidi_/bidirectional_rnn/fw/fw/ExpandDims%bidi_/bidirectional_rnn/fw/fw/Const_1+bidi_/bidirectional_rnn/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
n
)bidi_/bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ã
#bidi_/bidirectional_rnn/fw/fw/zerosFill&bidi_/bidirectional_rnn/fw/fw/concat_1)bidi_/bidirectional_rnn/fw/fw/zeros/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0
o
%bidi_/bidirectional_rnn/fw/fw/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
¸
!bidi_/bidirectional_rnn/fw/fw/MinMin)bidi_/bidirectional_rnn/fw/fw/CheckSeqLen%bidi_/bidirectional_rnn/fw/fw/Const_2*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
o
%bidi_/bidirectional_rnn/fw/fw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
¸
!bidi_/bidirectional_rnn/fw/fw/MaxMax)bidi_/bidirectional_rnn/fw/fw/CheckSeqLen%bidi_/bidirectional_rnn/fw/fw/Const_3*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
d
"bidi_/bidirectional_rnn/fw/fw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Ò
)bidi_/bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3-bidi_/bidirectional_rnn/fw/fw/strided_slice_1*
dynamic_size( *
identical_element_shapes(*%
element_shape:ÿÿÿÿÿÿÿÿÿÈ*
_output_shapes

:: *
dtype0*
clear_after_read(*I
tensor_array_name42bidi_/bidirectional_rnn/fw/fw/dynamic_rnn/output_0
Ó
+bidi_/bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3-bidi_/bidirectional_rnn/fw/fw/strided_slice_1*
dynamic_size( *
identical_element_shapes(*%
element_shape:ÿÿÿÿÿÿÿÿÿ*
_output_shapes

:: *
dtype0*
clear_after_read(*H
tensor_array_name31bidi_/bidirectional_rnn/fw/fw/dynamic_rnn/input_0

6bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape'bidi_/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:

Dbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Fbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Fbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
î
>bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice6bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeDbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackFbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Fbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
~
<bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
~
<bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
¬
6bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRange<bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/start>bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice<bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0

Xbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3+bidi_/bidirectional_rnn/fw/fw/TensorArray_16bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/range'bidi_/bidirectional_rnn/fw/fw/transpose-bidi_/bidirectional_rnn/fw/fw/TensorArray_1:1*
T0*
_output_shapes
: *:
_class0
.,loc:@bidi_/bidirectional_rnn/fw/fw/transpose
i
'bidi_/bidirectional_rnn/fw/fw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 

%bidi_/bidirectional_rnn/fw/fw/MaximumMaximum'bidi_/bidirectional_rnn/fw/fw/Maximum/x!bidi_/bidirectional_rnn/fw/fw/Max*
T0*
_output_shapes
: 
§
%bidi_/bidirectional_rnn/fw/fw/MinimumMinimum-bidi_/bidirectional_rnn/fw/fw/strided_slice_1%bidi_/bidirectional_rnn/fw/fw/Maximum*
T0*
_output_shapes
: 
w
5bidi_/bidirectional_rnn/fw/fw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
û
)bidi_/bidirectional_rnn/fw/fw/while/EnterEnter5bidi_/bidirectional_rnn/fw/fw/while/iteration_counter*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes
: 
ê
+bidi_/bidirectional_rnn/fw/fw/while/Enter_1Enter"bidi_/bidirectional_rnn/fw/fw/time*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes
: 
ó
+bidi_/bidirectional_rnn/fw/fw/while/Enter_2Enter+bidi_/bidirectional_rnn/fw/fw/TensorArray:1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes
: 
§
+bidi_/bidirectional_rnn/fw/fw/while/Enter_3EnterMbidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
©
+bidi_/bidirectional_rnn/fw/fw/while/Enter_4EnterObidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¼
)bidi_/bidirectional_rnn/fw/fw/while/MergeMerge)bidi_/bidirectional_rnn/fw/fw/while/Enter1bidi_/bidirectional_rnn/fw/fw/while/NextIteration*
T0*
N*
_output_shapes
: : 
Â
+bidi_/bidirectional_rnn/fw/fw/while/Merge_1Merge+bidi_/bidirectional_rnn/fw/fw/while/Enter_13bidi_/bidirectional_rnn/fw/fw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
Â
+bidi_/bidirectional_rnn/fw/fw/while/Merge_2Merge+bidi_/bidirectional_rnn/fw/fw/while/Enter_23bidi_/bidirectional_rnn/fw/fw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
Ô
+bidi_/bidirectional_rnn/fw/fw/while/Merge_3Merge+bidi_/bidirectional_rnn/fw/fw/while/Enter_33bidi_/bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
Ô
+bidi_/bidirectional_rnn/fw/fw/while/Merge_4Merge+bidi_/bidirectional_rnn/fw/fw/while/Enter_43bidi_/bidirectional_rnn/fw/fw/while/NextIteration_4*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
¬
(bidi_/bidirectional_rnn/fw/fw/while/LessLess)bidi_/bidirectional_rnn/fw/fw/while/Merge.bidi_/bidirectional_rnn/fw/fw/while/Less/Enter*
T0*
_output_shapes
: 
ø
.bidi_/bidirectional_rnn/fw/fw/while/Less/EnterEnter-bidi_/bidirectional_rnn/fw/fw/strided_slice_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
: 
²
*bidi_/bidirectional_rnn/fw/fw/while/Less_1Less+bidi_/bidirectional_rnn/fw/fw/while/Merge_10bidi_/bidirectional_rnn/fw/fw/while/Less_1/Enter*
T0*
_output_shapes
: 
ò
0bidi_/bidirectional_rnn/fw/fw/while/Less_1/EnterEnter%bidi_/bidirectional_rnn/fw/fw/Minimum*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
: 
ª
.bidi_/bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd(bidi_/bidirectional_rnn/fw/fw/while/Less*bidi_/bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 

,bidi_/bidirectional_rnn/fw/fw/while/LoopCondLoopCond.bidi_/bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
î
*bidi_/bidirectional_rnn/fw/fw/while/SwitchSwitch)bidi_/bidirectional_rnn/fw/fw/while/Merge,bidi_/bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/while/Merge
ô
,bidi_/bidirectional_rnn/fw/fw/while/Switch_1Switch+bidi_/bidirectional_rnn/fw/fw/while/Merge_1,bidi_/bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/fw/while/Merge_1
ô
,bidi_/bidirectional_rnn/fw/fw/while/Switch_2Switch+bidi_/bidirectional_rnn/fw/fw/while/Merge_2,bidi_/bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/fw/while/Merge_2

,bidi_/bidirectional_rnn/fw/fw/while/Switch_3Switch+bidi_/bidirectional_rnn/fw/fw/while/Merge_3,bidi_/bidirectional_rnn/fw/fw/while/LoopCond*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/fw/while/Merge_3

,bidi_/bidirectional_rnn/fw/fw/while/Switch_4Switch+bidi_/bidirectional_rnn/fw/fw/while/Merge_4,bidi_/bidirectional_rnn/fw/fw/while/LoopCond*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/fw/while/Merge_4

,bidi_/bidirectional_rnn/fw/fw/while/IdentityIdentity,bidi_/bidirectional_rnn/fw/fw/while/Switch:1*
T0*
_output_shapes
: 

.bidi_/bidirectional_rnn/fw/fw/while/Identity_1Identity.bidi_/bidirectional_rnn/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 

.bidi_/bidirectional_rnn/fw/fw/while/Identity_2Identity.bidi_/bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 

.bidi_/bidirectional_rnn/fw/fw/while/Identity_3Identity.bidi_/bidirectional_rnn/fw/fw/while/Switch_3:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

.bidi_/bidirectional_rnn/fw/fw/while/Identity_4Identity.bidi_/bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

)bidi_/bidirectional_rnn/fw/fw/while/add/yConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
¨
'bidi_/bidirectional_rnn/fw/fw/while/addAdd,bidi_/bidirectional_rnn/fw/fw/while/Identity)bidi_/bidirectional_rnn/fw/fw/while/add/y*
T0*
_output_shapes
: 
­
5bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV3;bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter.bidi_/bidirectional_rnn/fw/fw/while/Identity_1=bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter+bidi_/bidirectional_rnn/fw/fw/TensorArray_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
²
=bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1EnterXbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
: 
Ö
0bidi_/bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqual.bidi_/bidirectional_rnn/fw/fw/while/Identity_16bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

6bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter)bidi_/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
Lbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"R     *
dtype0*
_output_shapes
:*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ï
Jbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *Bµ½*
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ï
Jbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *Bµ=*
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ä
Tbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformLbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel*
T0* 
_output_shapes
:
Ò *
dtype0*

seed 
Ê
Jbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/subSubJbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxJbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Þ
Jbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulMulTbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformJbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ð
Fbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniformAddJbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulJbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ã
+bidi_/bidirectional_rnn/fw/lstm_cell/kernel
VariableV2*
shared_name *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
Ò *
dtype0*
	container *
shape:
Ò 
Å
2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AssignAssign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelFbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel

0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/readIdentity+bidi_/bidirectional_rnn/fw/lstm_cell/kernel*
T0* 
_output_shapes
:
Ò 
È
;bidi_/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
Õ
)bidi_/bidirectional_rnn/fw/lstm_cell/bias
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
: *
dtype0*
	container *
shape: 
¯
0bidi_/bidirectional_rnn/fw/lstm_cell/bias/AssignAssign)bidi_/bidirectional_rnn/fw/lstm_cell/bias;bidi_/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias

.bidi_/bidirectional_rnn/fw/lstm_cell/bias/readIdentity)bidi_/bidirectional_rnn/fw/lstm_cell/bias*
T0*
_output_shapes	
: 
ª
9bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axisConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ª
4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concatConcatV25bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3.bidi_/bidirectional_rnn/fw/fw/while/Identity_49bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ*

Tidx0

4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMulMatMul4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat:bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

:bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/EnterEnter0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/read*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(* 
_output_shapes
:
Ò 
ý
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAddBiasAdd4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul;bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

;bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/EnterEnter.bidi_/bidirectional_rnn/fw/lstm_cell/bias/read*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes	
: 
¤
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/ConstConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
®
=bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dimConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
²
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/splitSplit=bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split
§
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add/yConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
×
1bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/addAdd5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:23bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add/y*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¦
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/SigmoidSigmoid1bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ò
1bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mulMul5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid.bidi_/bidirectional_rnn/fw/fw/while/Identity_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ª
7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1Sigmoid3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
2bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/TanhTanh5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ú
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1Mul7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_12bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Õ
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1Add1bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¬
7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2Sigmoid5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1Tanh3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ü
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2Mul7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_24bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
1bidi_/bidirectional_rnn/fw/fw/while/dropout/ShapeShape3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
²
>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/minConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
²
>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/maxConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
å
Hbidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniformRandomUniform1bidi_/bidirectional_rnn/fw/fw/while/dropout/Shape*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
seed2 *

seed 
æ
>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/subSub>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/max>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*
_output_shapes
: 

>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mulMulHbidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ô
:bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniformAdd>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul>bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ì
/bidi_/bidirectional_rnn/fw/fw/while/dropout/addAdd5bidi_/bidirectional_rnn/fw/fw/while/dropout/add/Enter:bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform*
T0*
_output_shapes
:
å
5bidi_/bidirectional_rnn/fw/fw/while/dropout/add/EnterEnterdropout_keep_prob*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

1bidi_/bidirectional_rnn/fw/fw/while/dropout/FloorFloor/bidi_/bidirectional_rnn/fw/fw/while/dropout/add*
T0*
_output_shapes
:
É
/bidi_/bidirectional_rnn/fw/fw/while/dropout/divRealDiv3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_25bidi_/bidirectional_rnn/fw/fw/while/dropout/add/Enter*
T0*
_output_shapes
:
Í
/bidi_/bidirectional_rnn/fw/fw/while/dropout/mulMul/bidi_/bidirectional_rnn/fw/fw/while/dropout/div1bidi_/bidirectional_rnn/fw/fw/while/dropout/Floor*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
À
*bidi_/bidirectional_rnn/fw/fw/while/SelectSelect0bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual0bidi_/bidirectional_rnn/fw/fw/while/Select/Enter/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/mul
Æ
0bidi_/bidirectional_rnn/fw/fw/while/Select/EnterEnter#bidi_/bidirectional_rnn/fw/fw/zeros*
is_constant(*B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/mul*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
È
,bidi_/bidirectional_rnn/fw/fw/while/Select_1Select0bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual.bidi_/bidirectional_rnn/fw/fw/while/Identity_33bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*F
_class<
:8loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1
È
,bidi_/bidirectional_rnn/fw/fw/while/Select_2Select0bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual.bidi_/bidirectional_rnn/fw/fw/while/Identity_43bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*F
_class<
:8loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2

Gbidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Mbidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter.bidi_/bidirectional_rnn/fw/fw/while/Identity_1*bidi_/bidirectional_rnn/fw/fw/while/Select.bidi_/bidirectional_rnn/fw/fw/while/Identity_2*
T0*
_output_shapes
: *B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/mul
Û
Mbidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter)bidi_/bidirectional_rnn/fw/fw/TensorArray*
is_constant(*B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/mul*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:

+bidi_/bidirectional_rnn/fw/fw/while/add_1/yConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
®
)bidi_/bidirectional_rnn/fw/fw/while/add_1Add.bidi_/bidirectional_rnn/fw/fw/while/Identity_1+bidi_/bidirectional_rnn/fw/fw/while/add_1/y*
T0*
_output_shapes
: 

1bidi_/bidirectional_rnn/fw/fw/while/NextIterationNextIteration'bidi_/bidirectional_rnn/fw/fw/while/add*
T0*
_output_shapes
: 

3bidi_/bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration)bidi_/bidirectional_rnn/fw/fw/while/add_1*
T0*
_output_shapes
: 
®
3bidi_/bidirectional_rnn/fw/fw/while/NextIteration_2NextIterationGbidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
¥
3bidi_/bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration,bidi_/bidirectional_rnn/fw/fw/while/Select_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¥
3bidi_/bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration,bidi_/bidirectional_rnn/fw/fw/while/Select_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
}
(bidi_/bidirectional_rnn/fw/fw/while/ExitExit*bidi_/bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 

*bidi_/bidirectional_rnn/fw/fw/while/Exit_1Exit,bidi_/bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 

*bidi_/bidirectional_rnn/fw/fw/while/Exit_2Exit,bidi_/bidirectional_rnn/fw/fw/while/Switch_2*
T0*
_output_shapes
: 

*bidi_/bidirectional_rnn/fw/fw/while/Exit_3Exit,bidi_/bidirectional_rnn/fw/fw/while/Switch_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

*bidi_/bidirectional_rnn/fw/fw/while/Exit_4Exit,bidi_/bidirectional_rnn/fw/fw/while/Switch_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

@bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3)bidi_/bidirectional_rnn/fw/fw/TensorArray*bidi_/bidirectional_rnn/fw/fw/while/Exit_2*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray
º
:bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
value	B : *
dtype0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray
º
:bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray
æ
4bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/rangeRange:bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/range/start@bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3:bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray

Bbidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3)bidi_/bidirectional_rnn/fw/fw/TensorArray4bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/range*bidi_/bidirectional_rnn/fw/fw/while/Exit_2*%
element_shape:ÿÿÿÿÿÿÿÿÿÈ*
dtype0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray
p
%bidi_/bidirectional_rnn/fw/fw/Const_4Const*
valueB:È*
dtype0*
_output_shapes
:
f
$bidi_/bidirectional_rnn/fw/fw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
m
+bidi_/bidirectional_rnn/fw/fw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
m
+bidi_/bidirectional_rnn/fw/fw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ö
%bidi_/bidirectional_rnn/fw/fw/range_1Range+bidi_/bidirectional_rnn/fw/fw/range_1/start$bidi_/bidirectional_rnn/fw/fw/Rank_1+bidi_/bidirectional_rnn/fw/fw/range_1/delta*
_output_shapes
:*

Tidx0

/bidi_/bidirectional_rnn/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
m
+bidi_/bidirectional_rnn/fw/fw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
&bidi_/bidirectional_rnn/fw/fw/concat_2ConcatV2/bidi_/bidirectional_rnn/fw/fw/concat_2/values_0%bidi_/bidirectional_rnn/fw/fw/range_1+bidi_/bidirectional_rnn/fw/fw/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
ç
)bidi_/bidirectional_rnn/fw/fw/transpose_1	TransposeBbidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3&bidi_/bidirectional_rnn/fw/fw/concat_2*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
²
*bidi_/bidirectional_rnn/bw/ReverseSequenceReverseSequenceconcatseqlens*
seq_dim*
T0*

Tlen0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	batch_dim 
d
"bidi_/bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
k
)bidi_/bidirectional_rnn/bw/bw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
k
)bidi_/bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Î
#bidi_/bidirectional_rnn/bw/bw/rangeRange)bidi_/bidirectional_rnn/bw/bw/range/start"bidi_/bidirectional_rnn/bw/bw/Rank)bidi_/bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:*

Tidx0
~
-bidi_/bidirectional_rnn/bw/bw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
k
)bidi_/bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
é
$bidi_/bidirectional_rnn/bw/bw/concatConcatV2-bidi_/bidirectional_rnn/bw/bw/concat/values_0#bidi_/bidirectional_rnn/bw/bw/range)bidi_/bidirectional_rnn/bw/bw/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ë
'bidi_/bidirectional_rnn/bw/bw/transpose	Transpose*bidi_/bidirectional_rnn/bw/ReverseSequence$bidi_/bidirectional_rnn/bw/bw/concat*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
-bidi_/bidirectional_rnn/bw/bw/sequence_lengthIdentityseqlens*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#bidi_/bidirectional_rnn/bw/bw/ShapeShape'bidi_/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
{
1bidi_/bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

+bidi_/bidirectional_rnn/bw/bw/strided_sliceStridedSlice#bidi_/bidirectional_rnn/bw/bw/Shape1bidi_/bidirectional_rnn/bw/bw/strided_slice/stack3bidi_/bidirectional_rnn/bw/bw/strided_slice/stack_13bidi_/bidirectional_rnn/bw/bw/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 

Vbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Rbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims+bidi_/bidirectional_rnn/bw/bw/strided_sliceVbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

Mbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
valueB:È*
dtype0*
_output_shapes
:

Sbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Nbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Rbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsMbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ConstSbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

Sbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¿
Mbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillNbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concatSbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0

Xbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Tbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims+bidi_/bidirectional_rnn/bw/bw/strided_sliceXbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
T0*

Tdim0*
_output_shapes
:

Obidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:È*
dtype0*
_output_shapes
:

Xbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Tbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims+bidi_/bidirectional_rnn/bw/bw/strided_sliceXbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
T0*

Tdim0*
_output_shapes
:

Obidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
valueB:È*
dtype0*
_output_shapes
:

Ubidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Pbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Tbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Obidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Ubidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

Ubidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
Obidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillPbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Ubidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0

Xbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 

Tbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims+bidi_/bidirectional_rnn/bw/bw/strided_sliceXbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0*
_output_shapes
:

Obidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:È*
dtype0*
_output_shapes
:

%bidi_/bidirectional_rnn/bw/bw/Shape_1Shape-bidi_/bidirectional_rnn/bw/bw/sequence_length*
T0*
out_type0*
_output_shapes
:

#bidi_/bidirectional_rnn/bw/bw/stackPack+bidi_/bidirectional_rnn/bw/bw/strided_slice*

axis *
T0*
N*
_output_shapes
:

#bidi_/bidirectional_rnn/bw/bw/EqualEqual%bidi_/bidirectional_rnn/bw/bw/Shape_1#bidi_/bidirectional_rnn/bw/bw/stack*
T0*
_output_shapes
:
m
#bidi_/bidirectional_rnn/bw/bw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
§
!bidi_/bidirectional_rnn/bw/bw/AllAll#bidi_/bidirectional_rnn/bw/bw/Equal#bidi_/bidirectional_rnn/bw/bw/Const*
	keep_dims( *
_output_shapes
: *

Tidx0
¸
*bidi_/bidirectional_rnn/bw/bw/Assert/ConstConst*^
valueUBS BMExpected shape for Tensor bidi_/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 
}
,bidi_/bidirectional_rnn/bw/bw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
À
2bidi_/bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*^
valueUBS BMExpected shape for Tensor bidi_/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

2bidi_/bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
¬
+bidi_/bidirectional_rnn/bw/bw/Assert/AssertAssert!bidi_/bidirectional_rnn/bw/bw/All2bidi_/bidirectional_rnn/bw/bw/Assert/Assert/data_0#bidi_/bidirectional_rnn/bw/bw/stack2bidi_/bidirectional_rnn/bw/bw/Assert/Assert/data_2%bidi_/bidirectional_rnn/bw/bw/Shape_1*
T
2*
	summarize
À
)bidi_/bidirectional_rnn/bw/bw/CheckSeqLenIdentity-bidi_/bidirectional_rnn/bw/bw/sequence_length,^bidi_/bidirectional_rnn/bw/bw/Assert/Assert*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

%bidi_/bidirectional_rnn/bw/bw/Shape_2Shape'bidi_/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/bw/bw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/bw/bw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/bw/bw/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

-bidi_/bidirectional_rnn/bw/bw/strided_slice_1StridedSlice%bidi_/bidirectional_rnn/bw/bw/Shape_23bidi_/bidirectional_rnn/bw/bw/strided_slice_1/stack5bidi_/bidirectional_rnn/bw/bw/strided_slice_1/stack_15bidi_/bidirectional_rnn/bw/bw/strided_slice_1/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 

%bidi_/bidirectional_rnn/bw/bw/Shape_3Shape'bidi_/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
}
3bidi_/bidirectional_rnn/bw/bw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/bw/bw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

5bidi_/bidirectional_rnn/bw/bw/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

-bidi_/bidirectional_rnn/bw/bw/strided_slice_2StridedSlice%bidi_/bidirectional_rnn/bw/bw/Shape_33bidi_/bidirectional_rnn/bw/bw/strided_slice_2/stack5bidi_/bidirectional_rnn/bw/bw/strided_slice_2/stack_15bidi_/bidirectional_rnn/bw/bw/strided_slice_2/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
n
,bidi_/bidirectional_rnn/bw/bw/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ä
(bidi_/bidirectional_rnn/bw/bw/ExpandDims
ExpandDims-bidi_/bidirectional_rnn/bw/bw/strided_slice_2,bidi_/bidirectional_rnn/bw/bw/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
p
%bidi_/bidirectional_rnn/bw/bw/Const_1Const*
valueB:È*
dtype0*
_output_shapes
:
m
+bidi_/bidirectional_rnn/bw/bw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ê
&bidi_/bidirectional_rnn/bw/bw/concat_1ConcatV2(bidi_/bidirectional_rnn/bw/bw/ExpandDims%bidi_/bidirectional_rnn/bw/bw/Const_1+bidi_/bidirectional_rnn/bw/bw/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
n
)bidi_/bidirectional_rnn/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ã
#bidi_/bidirectional_rnn/bw/bw/zerosFill&bidi_/bidirectional_rnn/bw/bw/concat_1)bidi_/bidirectional_rnn/bw/bw/zeros/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0
o
%bidi_/bidirectional_rnn/bw/bw/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
¸
!bidi_/bidirectional_rnn/bw/bw/MinMin)bidi_/bidirectional_rnn/bw/bw/CheckSeqLen%bidi_/bidirectional_rnn/bw/bw/Const_2*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
o
%bidi_/bidirectional_rnn/bw/bw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
¸
!bidi_/bidirectional_rnn/bw/bw/MaxMax)bidi_/bidirectional_rnn/bw/bw/CheckSeqLen%bidi_/bidirectional_rnn/bw/bw/Const_3*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
d
"bidi_/bidirectional_rnn/bw/bw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Ò
)bidi_/bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3-bidi_/bidirectional_rnn/bw/bw/strided_slice_1*
dynamic_size( *
identical_element_shapes(*%
element_shape:ÿÿÿÿÿÿÿÿÿÈ*
_output_shapes

:: *
dtype0*
clear_after_read(*I
tensor_array_name42bidi_/bidirectional_rnn/bw/bw/dynamic_rnn/output_0
Ó
+bidi_/bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3-bidi_/bidirectional_rnn/bw/bw/strided_slice_1*
dynamic_size( *
identical_element_shapes(*%
element_shape:ÿÿÿÿÿÿÿÿÿ*
_output_shapes

:: *
dtype0*
clear_after_read(*H
tensor_array_name31bidi_/bidirectional_rnn/bw/bw/dynamic_rnn/input_0

6bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape'bidi_/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:

Dbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Fbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Fbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
î
>bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice6bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeDbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackFbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Fbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
~
<bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
~
<bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
¬
6bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRange<bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/start>bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice<bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0

Xbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3+bidi_/bidirectional_rnn/bw/bw/TensorArray_16bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/range'bidi_/bidirectional_rnn/bw/bw/transpose-bidi_/bidirectional_rnn/bw/bw/TensorArray_1:1*
T0*
_output_shapes
: *:
_class0
.,loc:@bidi_/bidirectional_rnn/bw/bw/transpose
i
'bidi_/bidirectional_rnn/bw/bw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 

%bidi_/bidirectional_rnn/bw/bw/MaximumMaximum'bidi_/bidirectional_rnn/bw/bw/Maximum/x!bidi_/bidirectional_rnn/bw/bw/Max*
T0*
_output_shapes
: 
§
%bidi_/bidirectional_rnn/bw/bw/MinimumMinimum-bidi_/bidirectional_rnn/bw/bw/strided_slice_1%bidi_/bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 
w
5bidi_/bidirectional_rnn/bw/bw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
û
)bidi_/bidirectional_rnn/bw/bw/while/EnterEnter5bidi_/bidirectional_rnn/bw/bw/while/iteration_counter*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes
: 
ê
+bidi_/bidirectional_rnn/bw/bw/while/Enter_1Enter"bidi_/bidirectional_rnn/bw/bw/time*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes
: 
ó
+bidi_/bidirectional_rnn/bw/bw/while/Enter_2Enter+bidi_/bidirectional_rnn/bw/bw/TensorArray:1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes
: 
§
+bidi_/bidirectional_rnn/bw/bw/while/Enter_3EnterMbidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
©
+bidi_/bidirectional_rnn/bw/bw/while/Enter_4EnterObidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¼
)bidi_/bidirectional_rnn/bw/bw/while/MergeMerge)bidi_/bidirectional_rnn/bw/bw/while/Enter1bidi_/bidirectional_rnn/bw/bw/while/NextIteration*
T0*
N*
_output_shapes
: : 
Â
+bidi_/bidirectional_rnn/bw/bw/while/Merge_1Merge+bidi_/bidirectional_rnn/bw/bw/while/Enter_13bidi_/bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
Â
+bidi_/bidirectional_rnn/bw/bw/while/Merge_2Merge+bidi_/bidirectional_rnn/bw/bw/while/Enter_23bidi_/bidirectional_rnn/bw/bw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
Ô
+bidi_/bidirectional_rnn/bw/bw/while/Merge_3Merge+bidi_/bidirectional_rnn/bw/bw/while/Enter_33bidi_/bidirectional_rnn/bw/bw/while/NextIteration_3*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
Ô
+bidi_/bidirectional_rnn/bw/bw/while/Merge_4Merge+bidi_/bidirectional_rnn/bw/bw/while/Enter_43bidi_/bidirectional_rnn/bw/bw/while/NextIteration_4*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
¬
(bidi_/bidirectional_rnn/bw/bw/while/LessLess)bidi_/bidirectional_rnn/bw/bw/while/Merge.bidi_/bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
ø
.bidi_/bidirectional_rnn/bw/bw/while/Less/EnterEnter-bidi_/bidirectional_rnn/bw/bw/strided_slice_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
: 
²
*bidi_/bidirectional_rnn/bw/bw/while/Less_1Less+bidi_/bidirectional_rnn/bw/bw/while/Merge_10bidi_/bidirectional_rnn/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
ò
0bidi_/bidirectional_rnn/bw/bw/while/Less_1/EnterEnter%bidi_/bidirectional_rnn/bw/bw/Minimum*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
: 
ª
.bidi_/bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd(bidi_/bidirectional_rnn/bw/bw/while/Less*bidi_/bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 

,bidi_/bidirectional_rnn/bw/bw/while/LoopCondLoopCond.bidi_/bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
î
*bidi_/bidirectional_rnn/bw/bw/while/SwitchSwitch)bidi_/bidirectional_rnn/bw/bw/while/Merge,bidi_/bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/while/Merge
ô
,bidi_/bidirectional_rnn/bw/bw/while/Switch_1Switch+bidi_/bidirectional_rnn/bw/bw/while/Merge_1,bidi_/bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/bw/while/Merge_1
ô
,bidi_/bidirectional_rnn/bw/bw/while/Switch_2Switch+bidi_/bidirectional_rnn/bw/bw/while/Merge_2,bidi_/bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/bw/while/Merge_2

,bidi_/bidirectional_rnn/bw/bw/while/Switch_3Switch+bidi_/bidirectional_rnn/bw/bw/while/Merge_3,bidi_/bidirectional_rnn/bw/bw/while/LoopCond*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/bw/while/Merge_3

,bidi_/bidirectional_rnn/bw/bw/while/Switch_4Switch+bidi_/bidirectional_rnn/bw/bw/while/Merge_4,bidi_/bidirectional_rnn/bw/bw/while/LoopCond*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/bw/while/Merge_4

,bidi_/bidirectional_rnn/bw/bw/while/IdentityIdentity,bidi_/bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 

.bidi_/bidirectional_rnn/bw/bw/while/Identity_1Identity.bidi_/bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 

.bidi_/bidirectional_rnn/bw/bw/while/Identity_2Identity.bidi_/bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 

.bidi_/bidirectional_rnn/bw/bw/while/Identity_3Identity.bidi_/bidirectional_rnn/bw/bw/while/Switch_3:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

.bidi_/bidirectional_rnn/bw/bw/while/Identity_4Identity.bidi_/bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

)bidi_/bidirectional_rnn/bw/bw/while/add/yConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
¨
'bidi_/bidirectional_rnn/bw/bw/while/addAdd,bidi_/bidirectional_rnn/bw/bw/while/Identity)bidi_/bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 
­
5bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV3;bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter.bidi_/bidirectional_rnn/bw/bw/while/Identity_1=bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

;bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter+bidi_/bidirectional_rnn/bw/bw/TensorArray_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
²
=bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1EnterXbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
: 
Ö
0bidi_/bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqual.bidi_/bidirectional_rnn/bw/bw/while/Identity_16bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

6bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter)bidi_/bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ý
Lbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"R     *
dtype0*
_output_shapes
:*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Ï
Jbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *Bµ½*
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Ï
Jbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *Bµ=*
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Ä
Tbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformLbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel*
T0* 
_output_shapes
:
Ò *
dtype0*

seed 
Ê
Jbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/subSubJbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxJbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Þ
Jbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulMulTbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformJbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Ð
Fbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniformAddJbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulJbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ã
+bidi_/bidirectional_rnn/bw/lstm_cell/kernel
VariableV2*
shared_name *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
Ò *
dtype0*
	container *
shape:
Ò 
Å
2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AssignAssign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelFbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel

0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/readIdentity+bidi_/bidirectional_rnn/bw/lstm_cell/kernel*
T0* 
_output_shapes
:
Ò 
È
;bidi_/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Õ
)bidi_/bidirectional_rnn/bw/lstm_cell/bias
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
: *
dtype0*
	container *
shape: 
¯
0bidi_/bidirectional_rnn/bw/lstm_cell/bias/AssignAssign)bidi_/bidirectional_rnn/bw/lstm_cell/bias;bidi_/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias

.bidi_/bidirectional_rnn/bw/lstm_cell/bias/readIdentity)bidi_/bidirectional_rnn/bw/lstm_cell/bias*
T0*
_output_shapes	
: 
ª
9bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axisConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ª
4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concatConcatV25bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3.bidi_/bidirectional_rnn/bw/bw/while/Identity_49bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ*

Tidx0

4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMulMatMul4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat:bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

:bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/EnterEnter0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/read*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(* 
_output_shapes
:
Ò 
ý
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAddBiasAdd4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul;bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 

;bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/EnterEnter.bidi_/bidirectional_rnn/bw/lstm_cell/bias/read*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes	
: 
¤
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/ConstConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
®
=bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dimConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
²
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/splitSplit=bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*
	num_split
§
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add/yConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
×
1bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/addAdd5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:23bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add/y*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¦
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/SigmoidSigmoid1bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ò
1bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mulMul5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid.bidi_/bidirectional_rnn/bw/bw/while/Identity_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ª
7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1Sigmoid3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
2bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/TanhTanh5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ú
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1Mul7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_12bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Õ
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1Add1bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¬
7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2Sigmoid5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1Tanh3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ü
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2Mul7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_24bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
1bidi_/bidirectional_rnn/bw/bw/while/dropout/ShapeShape3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
²
>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/minConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
²
>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/maxConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
å
Hbidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniformRandomUniform1bidi_/bidirectional_rnn/bw/bw/while/dropout/Shape*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
seed2 *

seed 
æ
>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/subSub>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/max>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
T0*
_output_shapes
: 

>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mulMulHbidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ô
:bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniformAdd>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul>bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ì
/bidi_/bidirectional_rnn/bw/bw/while/dropout/addAdd5bidi_/bidirectional_rnn/bw/bw/while/dropout/add/Enter:bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform*
T0*
_output_shapes
:
å
5bidi_/bidirectional_rnn/bw/bw/while/dropout/add/EnterEnterdropout_keep_prob*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

1bidi_/bidirectional_rnn/bw/bw/while/dropout/FloorFloor/bidi_/bidirectional_rnn/bw/bw/while/dropout/add*
T0*
_output_shapes
:
É
/bidi_/bidirectional_rnn/bw/bw/while/dropout/divRealDiv3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_25bidi_/bidirectional_rnn/bw/bw/while/dropout/add/Enter*
T0*
_output_shapes
:
Í
/bidi_/bidirectional_rnn/bw/bw/while/dropout/mulMul/bidi_/bidirectional_rnn/bw/bw/while/dropout/div1bidi_/bidirectional_rnn/bw/bw/while/dropout/Floor*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
À
*bidi_/bidirectional_rnn/bw/bw/while/SelectSelect0bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual0bidi_/bidirectional_rnn/bw/bw/while/Select/Enter/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/mul
Æ
0bidi_/bidirectional_rnn/bw/bw/while/Select/EnterEnter#bidi_/bidirectional_rnn/bw/bw/zeros*
is_constant(*B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/mul*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
È
,bidi_/bidirectional_rnn/bw/bw/while/Select_1Select0bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual.bidi_/bidirectional_rnn/bw/bw/while/Identity_33bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*F
_class<
:8loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1
È
,bidi_/bidirectional_rnn/bw/bw/while/Select_2Select0bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual.bidi_/bidirectional_rnn/bw/bw/while/Identity_43bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*F
_class<
:8loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2

Gbidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Mbidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter.bidi_/bidirectional_rnn/bw/bw/while/Identity_1*bidi_/bidirectional_rnn/bw/bw/while/Select.bidi_/bidirectional_rnn/bw/bw/while/Identity_2*
T0*
_output_shapes
: *B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/mul
Û
Mbidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter)bidi_/bidirectional_rnn/bw/bw/TensorArray*
is_constant(*B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/mul*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:

+bidi_/bidirectional_rnn/bw/bw/while/add_1/yConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
®
)bidi_/bidirectional_rnn/bw/bw/while/add_1Add.bidi_/bidirectional_rnn/bw/bw/while/Identity_1+bidi_/bidirectional_rnn/bw/bw/while/add_1/y*
T0*
_output_shapes
: 

1bidi_/bidirectional_rnn/bw/bw/while/NextIterationNextIteration'bidi_/bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 

3bidi_/bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration)bidi_/bidirectional_rnn/bw/bw/while/add_1*
T0*
_output_shapes
: 
®
3bidi_/bidirectional_rnn/bw/bw/while/NextIteration_2NextIterationGbidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
¥
3bidi_/bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration,bidi_/bidirectional_rnn/bw/bw/while/Select_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¥
3bidi_/bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration,bidi_/bidirectional_rnn/bw/bw/while/Select_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
}
(bidi_/bidirectional_rnn/bw/bw/while/ExitExit*bidi_/bidirectional_rnn/bw/bw/while/Switch*
T0*
_output_shapes
: 

*bidi_/bidirectional_rnn/bw/bw/while/Exit_1Exit,bidi_/bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 

*bidi_/bidirectional_rnn/bw/bw/while/Exit_2Exit,bidi_/bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 

*bidi_/bidirectional_rnn/bw/bw/while/Exit_3Exit,bidi_/bidirectional_rnn/bw/bw/while/Switch_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

*bidi_/bidirectional_rnn/bw/bw/while/Exit_4Exit,bidi_/bidirectional_rnn/bw/bw/while/Switch_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

@bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3)bidi_/bidirectional_rnn/bw/bw/TensorArray*bidi_/bidirectional_rnn/bw/bw/while/Exit_2*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray
º
:bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*
value	B : *
dtype0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray
º
:bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray
æ
4bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/rangeRange:bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/range/start@bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3:bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray

Bbidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3)bidi_/bidirectional_rnn/bw/bw/TensorArray4bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/range*bidi_/bidirectional_rnn/bw/bw/while/Exit_2*%
element_shape:ÿÿÿÿÿÿÿÿÿÈ*
dtype0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray
p
%bidi_/bidirectional_rnn/bw/bw/Const_4Const*
valueB:È*
dtype0*
_output_shapes
:
f
$bidi_/bidirectional_rnn/bw/bw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
m
+bidi_/bidirectional_rnn/bw/bw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
m
+bidi_/bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ö
%bidi_/bidirectional_rnn/bw/bw/range_1Range+bidi_/bidirectional_rnn/bw/bw/range_1/start$bidi_/bidirectional_rnn/bw/bw/Rank_1+bidi_/bidirectional_rnn/bw/bw/range_1/delta*
_output_shapes
:*

Tidx0

/bidi_/bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
m
+bidi_/bidirectional_rnn/bw/bw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
&bidi_/bidirectional_rnn/bw/bw/concat_2ConcatV2/bidi_/bidirectional_rnn/bw/bw/concat_2/values_0%bidi_/bidirectional_rnn/bw/bw/range_1+bidi_/bidirectional_rnn/bw/bw/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
ç
)bidi_/bidirectional_rnn/bw/bw/transpose_1	TransposeBbidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3&bidi_/bidirectional_rnn/bw/bw/concat_2*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
À
bidi_/ReverseSequenceReverseSequence)bidi_/bidirectional_rnn/bw/bw/transpose_1seqlens*
seq_dim*
T0*

Tlen0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	batch_dim 
O
concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
²
concat_1ConcatV2)bidi_/bidirectional_rnn/fw/fw/transpose_1bidi_/ReverseSequenceconcat_1/axis*
T0*
N*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0
Y
ExpandDims/dimConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
z

ExpandDims
ExpandDimsconcat_1ExpandDims/dim*
T0*

Tdim0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ñ
Cconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/shapeConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
¼
Bconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
¾
Dconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
²
Mconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/shape*
seed2 *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0*
T0*'
_output_shapes
:d*
dtype0*

seed 
Ä
Aconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/mulMulMconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/TruncatedNormalDconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/stddev*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
²
=conv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normalAddAconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/mulBconv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal/mean*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Û
 conv_maxpool_3_conv_1/W_filter_0
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¢
'conv_maxpool_3_conv_1/W_filter_0/AssignAssign conv_maxpool_3_conv_1/W_filter_0=conv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
º
%conv_maxpool_3_conv_1/W_filter_0/readIdentity conv_maxpool_3_conv_1/W_filter_0*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
´
2conv_maxpool_3_conv_1/B_filter_0/Initializer/ConstConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Á
 conv_maxpool_3_conv_1/B_filter_0
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0*
_output_shapes
:d*
dtype0*
	container *
shape:d

'conv_maxpool_3_conv_1/B_filter_0/AssignAssign conv_maxpool_3_conv_1/B_filter_02conv_maxpool_3_conv_1/B_filter_0/Initializer/Const*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
­
%conv_maxpool_3_conv_1/B_filter_0/readIdentity conv_maxpool_3_conv_1/B_filter_0*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0

"conv_maxpool_3_conv_1/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:

conv_maxpool_3_conv_1/PadPad
ExpandDims"conv_maxpool_3_conv_1/Pad/paddings*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	Tpaddings0

conv_maxpool_3_conv_1/conv_opConv2Dconv_maxpool_3_conv_1/Pad%conv_maxpool_3_conv_1/W_filter_0/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
use_cudnn_on_gpu(
À
conv_maxpool_3_conv_1/BiasAddBiasAddconv_maxpool_3_conv_1/conv_op%conv_maxpool_3_conv_1/B_filter_0/read*
T0*
data_formatNHWC*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

!conv_maxpool_3_conv_1/conv_nonlinReluconv_maxpool_3_conv_1/BiasAdd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
x
#conv_maxpool_3_conv_1/Reshape/shapeConst*!
valueB"ÿÿÿÿ   d   *
dtype0*
_output_shapes
:
µ
conv_maxpool_3_conv_1/ReshapeReshape!conv_maxpool_3_conv_1/conv_nonlin#conv_maxpool_3_conv_1/Reshape/shape*
T0*
Tshape0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ñ
Cconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/shapeConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
¼
Bconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
¾
Dconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
²
Mconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/shape*
seed2 *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1*
T0*'
_output_shapes
:d*
dtype0*

seed 
Ä
Aconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/mulMulMconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/TruncatedNormalDconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/stddev*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
²
=conv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normalAddAconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/mulBconv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal/mean*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Û
 conv_maxpool_5_conv_1/W_filter_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¢
'conv_maxpool_5_conv_1/W_filter_1/AssignAssign conv_maxpool_5_conv_1/W_filter_1=conv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
º
%conv_maxpool_5_conv_1/W_filter_1/readIdentity conv_maxpool_5_conv_1/W_filter_1*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
´
2conv_maxpool_5_conv_1/B_filter_1/Initializer/ConstConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Á
 conv_maxpool_5_conv_1/B_filter_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1*
_output_shapes
:d*
dtype0*
	container *
shape:d

'conv_maxpool_5_conv_1/B_filter_1/AssignAssign conv_maxpool_5_conv_1/B_filter_12conv_maxpool_5_conv_1/B_filter_1/Initializer/Const*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
­
%conv_maxpool_5_conv_1/B_filter_1/readIdentity conv_maxpool_5_conv_1/B_filter_1*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1

"conv_maxpool_5_conv_1/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:

conv_maxpool_5_conv_1/PadPad
ExpandDims"conv_maxpool_5_conv_1/Pad/paddings*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	Tpaddings0

conv_maxpool_5_conv_1/conv_opConv2Dconv_maxpool_5_conv_1/Pad%conv_maxpool_5_conv_1/W_filter_1/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
use_cudnn_on_gpu(
À
conv_maxpool_5_conv_1/BiasAddBiasAddconv_maxpool_5_conv_1/conv_op%conv_maxpool_5_conv_1/B_filter_1/read*
T0*
data_formatNHWC*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

!conv_maxpool_5_conv_1/conv_nonlinReluconv_maxpool_5_conv_1/BiasAdd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
x
#conv_maxpool_5_conv_1/Reshape/shapeConst*!
valueB"ÿÿÿÿ   d   *
dtype0*
_output_shapes
:
µ
conv_maxpool_5_conv_1/ReshapeReshape!conv_maxpool_5_conv_1/conv_nonlin#conv_maxpool_5_conv_1/Reshape/shape*
T0*
Tshape0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ñ
Cconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/shapeConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¼
Bconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¾
Dconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/stddevConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
²
Mconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/TruncatedNormalTruncatedNormalCconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/shape*
seed2 *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2*
T0*'
_output_shapes
:d*
dtype0*

seed 
Ä
Aconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/mulMulMconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/TruncatedNormalDconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/stddev*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
²
=conv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normalAddAconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/mulBconv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal/mean*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
Û
 conv_maxpool_7_conv_1/W_filter_2
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¢
'conv_maxpool_7_conv_1/W_filter_2/AssignAssign conv_maxpool_7_conv_1/W_filter_2=conv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
º
%conv_maxpool_7_conv_1/W_filter_2/readIdentity conv_maxpool_7_conv_1/W_filter_2*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
´
2conv_maxpool_7_conv_1/B_filter_2/Initializer/ConstConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Á
 conv_maxpool_7_conv_1/B_filter_2
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2*
_output_shapes
:d*
dtype0*
	container *
shape:d

'conv_maxpool_7_conv_1/B_filter_2/AssignAssign conv_maxpool_7_conv_1/B_filter_22conv_maxpool_7_conv_1/B_filter_2/Initializer/Const*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
­
%conv_maxpool_7_conv_1/B_filter_2/readIdentity conv_maxpool_7_conv_1/B_filter_2*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2

"conv_maxpool_7_conv_1/Pad/paddingsConst*9
value0B."                               *
dtype0*
_output_shapes

:

conv_maxpool_7_conv_1/PadPad
ExpandDims"conv_maxpool_7_conv_1/Pad/paddings*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	Tpaddings0

conv_maxpool_7_conv_1/conv_opConv2Dconv_maxpool_7_conv_1/Pad%conv_maxpool_7_conv_1/W_filter_2/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
use_cudnn_on_gpu(
À
conv_maxpool_7_conv_1/BiasAddBiasAddconv_maxpool_7_conv_1/conv_op%conv_maxpool_7_conv_1/B_filter_2/read*
T0*
data_formatNHWC*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

!conv_maxpool_7_conv_1/conv_nonlinReluconv_maxpool_7_conv_1/BiasAdd*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
x
#conv_maxpool_7_conv_1/Reshape/shapeConst*!
valueB"ÿÿÿÿ   d   *
dtype0*
_output_shapes
:
µ
conv_maxpool_7_conv_1/ReshapeReshape!conv_maxpool_7_conv_1/conv_nonlin#conv_maxpool_7_conv_1/Reshape/shape*
T0*
Tshape0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
O
concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Í
concat_2ConcatV2conv_maxpool_3_conv_1/Reshapeconv_maxpool_5_conv_1/Reshapeconv_maxpool_7_conv_1/Reshapeconcat_2/axis*
T0*
N*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*

Tidx0
O
concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 

concat_3ConcatV2concat_2concat_1concatconcat_3/axis*
T0*
N*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ*

Tidx0
^
Reshape/shapeConst*
valueB"ÿÿÿÿF  *
dtype0*
_output_shapes
:
l
ReshapeReshapeconcat_3Reshape/shape*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ

.out_weights/Initializer/truncated_normal/shapeConst*
valueB"F  R   *
dtype0*
_output_shapes
:*
_class
loc:@out_weights

-out_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@out_weights

/out_weights/Initializer/truncated_normal/stddevConst*
valueB
 *u~=*
dtype0*
_output_shapes
: *
_class
loc:@out_weights
ë
8out_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal.out_weights/Initializer/truncated_normal/shape*
seed2 *
_class
loc:@out_weights*
T0*
_output_shapes
:	ÆR*
dtype0*

seed 
è
,out_weights/Initializer/truncated_normal/mulMul8out_weights/Initializer/truncated_normal/TruncatedNormal/out_weights/Initializer/truncated_normal/stddev*
T0*
_output_shapes
:	ÆR*
_class
loc:@out_weights
Ö
(out_weights/Initializer/truncated_normalAdd,out_weights/Initializer/truncated_normal/mul-out_weights/Initializer/truncated_normal/mean*
T0*
_output_shapes
:	ÆR*
_class
loc:@out_weights
¡
out_weights
VariableV2*
shared_name *
_class
loc:@out_weights*
_output_shapes
:	ÆR*
dtype0*
	container *
shape:	ÆR
Æ
out_weights/AssignAssignout_weights(out_weights/Initializer/truncated_normal*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
s
out_weights/readIdentityout_weights*
T0*
_output_shapes
:	ÆR*
_class
loc:@out_weights

MatMulMatMulReshapeout_weights/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

out_bias/Initializer/ConstConst*
valueBR*    *
dtype0*
_output_shapes
:R*
_class
loc:@out_bias

out_bias
VariableV2*
shared_name *
_class
loc:@out_bias*
_output_shapes
:R*
dtype0*
	container *
shape:R
ª
out_bias/AssignAssignout_biasout_bias/Initializer/Const*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
e
out_bias/readIdentityout_bias*
T0*
_output_shapes
:R*
_class
loc:@out_bias
S
addAddMatMulout_bias/read*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
d
Reshape_1/shapeConst*!
valueB"ÿÿÿÿ   R   *
dtype0*
_output_shapes
:
o
	Reshape_1ReshapeaddReshape_1/shape*
T0*
Tshape0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
_
sparse_softmax_fun/ShapeShapetargets*
T0	*
out_type0*
_output_shapes
:
a
sparse_softmax_fun/Shape_1Shapetargets*
T0	*
out_type0*
_output_shapes
:
c
sparse_softmax_fun/Shape_2Shape	Reshape_1*
T0*
out_type0*
_output_shapes
:
p
&sparse_softmax_fun/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
{
(sparse_softmax_fun/strided_slice/stack_1Const*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
r
(sparse_softmax_fun/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Þ
 sparse_softmax_fun/strided_sliceStridedSlicesparse_softmax_fun/Shape_2&sparse_softmax_fun/strided_slice/stack(sparse_softmax_fun/strided_slice/stack_1(sparse_softmax_fun/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask *

begin_mask*
new_axis_mask *
_output_shapes
:*
ellipsis_mask 

%sparse_softmax_fun/assert_equal/EqualEqualsparse_softmax_fun/Shape_1 sparse_softmax_fun/strided_slice*
T0*
_output_shapes
:
o
%sparse_softmax_fun/assert_equal/ConstConst*
valueB: *
dtype0*
_output_shapes
:
­
#sparse_softmax_fun/assert_equal/AllAll%sparse_softmax_fun/assert_equal/Equal%sparse_softmax_fun/assert_equal/Const*
	keep_dims( *
_output_shapes
: *

Tidx0
m
,sparse_softmax_fun/assert_equal/Assert/ConstConst*
valueB B *
dtype0*
_output_shapes
: 

.sparse_softmax_fun/assert_equal/Assert/Const_1Const*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 

.sparse_softmax_fun/assert_equal/Assert/Const_2Const*4
value+B) B#x (sparse_softmax_fun/Shape_1:0) = *
dtype0*
_output_shapes
: 

.sparse_softmax_fun/assert_equal/Assert/Const_3Const*:
value1B/ B)y (sparse_softmax_fun/strided_slice:0) = *
dtype0*
_output_shapes
: 
u
4sparse_softmax_fun/assert_equal/Assert/Assert/data_0Const*
valueB B *
dtype0*
_output_shapes
: 
 
4sparse_softmax_fun/assert_equal/Assert/Assert/data_1Const*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 

4sparse_softmax_fun/assert_equal/Assert/Assert/data_2Const*4
value+B) B#x (sparse_softmax_fun/Shape_1:0) = *
dtype0*
_output_shapes
: 

4sparse_softmax_fun/assert_equal/Assert/Assert/data_4Const*:
value1B/ B)y (sparse_softmax_fun/strided_slice:0) = *
dtype0*
_output_shapes
: 

-sparse_softmax_fun/assert_equal/Assert/AssertAssert#sparse_softmax_fun/assert_equal/All4sparse_softmax_fun/assert_equal/Assert/Assert/data_04sparse_softmax_fun/assert_equal/Assert/Assert/data_14sparse_softmax_fun/assert_equal/Assert/Assert/data_2sparse_softmax_fun/Shape_14sparse_softmax_fun/assert_equal/Assert/Assert/data_4 sparse_softmax_fun/strided_slice*
T

2*
	summarize

sparse_softmax_fun/Shape_3Shape	Reshape_1.^sparse_softmax_fun/assert_equal/Assert/Assert*
T0*
out_type0*
_output_shapes
:

sparse_softmax_fun/RankConst.^sparse_softmax_fun/assert_equal/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 

sparse_softmax_fun/sub/yConst.^sparse_softmax_fun/assert_equal/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
q
sparse_softmax_fun/subSubsparse_softmax_fun/Ranksparse_softmax_fun/sub/y*
T0*
_output_shapes
: 

sparse_softmax_fun/add/yConst.^sparse_softmax_fun/assert_equal/Assert/Assert*
value	B :*
dtype0*
_output_shapes
: 
p
sparse_softmax_fun/addAddsparse_softmax_fun/subsparse_softmax_fun/add/y*
T0*
_output_shapes
: 

(sparse_softmax_fun/strided_slice_1/stackPacksparse_softmax_fun/sub*

axis *
T0*
N*
_output_shapes
:

*sparse_softmax_fun/strided_slice_1/stack_1Packsparse_softmax_fun/add*

axis *
T0*
N*
_output_shapes
:
¤
*sparse_softmax_fun/strided_slice_1/stack_2Const.^sparse_softmax_fun/assert_equal/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
â
"sparse_softmax_fun/strided_slice_1StridedSlicesparse_softmax_fun/Shape_3(sparse_softmax_fun/strided_slice_1/stack*sparse_softmax_fun/strided_slice_1/stack_1*sparse_softmax_fun/strided_slice_1/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 

"sparse_softmax_fun/Reshape/shape/0Const.^sparse_softmax_fun/assert_equal/Assert/Assert*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
ª
 sparse_softmax_fun/Reshape/shapePack"sparse_softmax_fun/Reshape/shape/0"sparse_softmax_fun/strided_slice_1*

axis *
T0*
N*
_output_shapes
:

sparse_softmax_fun/ReshapeReshape	Reshape_1 sparse_softmax_fun/Reshape/shape*
T0*
Tshape0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¥
"sparse_softmax_fun/Reshape_1/shapeConst.^sparse_softmax_fun/assert_equal/Assert/Assert*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:

sparse_softmax_fun/Reshape_1Reshapetargets"sparse_softmax_fun/Reshape_1/shape*
T0	*
Tshape0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ß
%sparse_softmax_fun/sparse_softmax_fun#SparseSoftmaxCrossEntropyWithLogitssparse_softmax_fun/Reshapesparse_softmax_fun/Reshape_1*
T0*?
_output_shapes-
+:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
Tlabels0	
©
sparse_softmax_fun/Reshape_2Reshape%sparse_softmax_fun/sparse_softmax_funsparse_softmax_fun/Shape*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
N
ShapeShape	Reshape_1*
T0*
out_type0*
_output_shapes
:
F
RankConst*
value	B :*
dtype0*
_output_shapes
: 
P
Shape_1Shape	Reshape_1*
T0*
out_type0*
_output_shapes
:
G
Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
8
SubSubRankSub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*

axis *
T0*
N*
_output_shapes
:
T

Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
Index0*
T0*
_output_shapes
:
d
concat_4/values_0Const*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
O
concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
w
concat_4ConcatV2concat_4/values_0Sliceconcat_4/axis*
T0*
N*
_output_shapes
:*

Tidx0
r
	Reshape_2Reshape	Reshape_1concat_4*
T0*
Tshape0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
X
SoftmaxSoftmax	Reshape_2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
i
	Reshape_3ReshapeSoftmaxShape*
T0*
Tshape0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
W
predictions/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

predictionsArgMax	Reshape_3predictions/dimension*
output_type0	*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0
W
EqualEqualpredictionstargets*
T0	*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
U
CastCastEqual*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

SrcT0

M
Cumsum/axisConst*
value	B :*
dtype0*
_output_shapes
: 

CumsumCumsumCastCumsum/axis*
T0*
	exclusive( *
reverse( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0
M
Shape_2Shapeinputs*
T0	*
out_type0*
_output_shapes
:
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
û
strided_sliceStridedSliceShape_2strided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
h
rangeRangerange/startstrided_slicerange/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0
G
sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
H
subSubseqlenssub/y*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
stackPackrangesub*

axis*
T0*
N*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
g
GatherNdGatherNdCumsumstack*
Tparams0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tindices0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
Y
SumSumGatherNdConst*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
\
Sum_1SumseqlensConst_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
E
Cast_1CastSum_1*

DstT0*
_output_shapes
: *

SrcT0
<
divRealDivSumCast_1*
T0*
_output_shapes
: 
V
accuracy/tagsConst*
valueB Baccuracy*
dtype0*
_output_shapes
: 
N
accuracyScalarSummaryaccuracy/tagsdiv*
T0*
_output_shapes
: 
X
Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
v
	mean_costMeansparse_softmax_fun/Reshape_2Const_2*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
N
	cost/tagsConst*
valueB
 Bcost*
dtype0*
_output_shapes
: 
L
costScalarSummary	cost/tags	mean_cost*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
Á
gradients/f_count_1Entergradients/f_count*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes
: 
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
|
gradients/SwitchSwitchgradients/Merge,bidi_/bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : 

gradients/Add/yConst-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
Ñ
gradients/NextIterationNextIterationgradients/AddO^gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2S^gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2S^gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2u^gradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2a^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2c^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPushV2a^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2c^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2Q^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2Y^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2e^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2g^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1c^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2W^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2e^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2g^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1S^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2U^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2e^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2g^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1S^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2U^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2c^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2e^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1S^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
N
gradients/f_count_2Exitgradients/Switch*
T0*
_output_shapes
: 
S
gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
Í
gradients/b_count_1Entergradients/f_count_2*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes
: 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ô
gradients/GreaterEqual/EnterEntergradients/b_count*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
: 
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0*
_output_shapes
: : 
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
Ì
gradients/NextIteration_1NextIterationgradients/Subp^gradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
U
gradients/f_count_3Const*
value	B : *
dtype0*
_output_shapes
: 
Ã
gradients/f_count_4Entergradients/f_count_3*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes
: 
v
gradients/Merge_2Mergegradients/f_count_4gradients/NextIteration_2*
T0*
N*
_output_shapes
: : 

gradients/Switch_2Switchgradients/Merge_2,bidi_/bidirectional_rnn/bw/bw/while/LoopCond*
T0*
_output_shapes
: : 

gradients/Add_1/yConst-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
gradients/Add_1Addgradients/Switch_2:1gradients/Add_1/y*
T0*
_output_shapes
: 
Õ
gradients/NextIteration_2NextIterationgradients/Add_1O^gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2S^gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2S^gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2u^gradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2a^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2c^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPushV2a^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2c^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1O^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2Q^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2Y^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2e^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2g^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1c^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2W^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2e^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2g^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1S^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2U^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2e^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2g^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1S^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2U^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2c^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2e^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1S^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
P
gradients/f_count_5Exitgradients/Switch_2*
T0*
_output_shapes
: 
U
gradients/b_count_4Const*
value	B :*
dtype0*
_output_shapes
: 
Í
gradients/b_count_5Entergradients/f_count_5*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes
: 
v
gradients/Merge_3Mergegradients/b_count_5gradients/NextIteration_3*
T0*
N*
_output_shapes
: : 
|
gradients/GreaterEqual_1GreaterEqualgradients/Merge_3gradients/GreaterEqual_1/Enter*
T0*
_output_shapes
: 
Ø
gradients/GreaterEqual_1/EnterEntergradients/b_count_4*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
: 
Q
gradients/b_count_6LoopCondgradients/GreaterEqual_1*
_output_shapes
: 
g
gradients/Switch_3Switchgradients/Merge_3gradients/b_count_6*
T0*
_output_shapes
: : 
m
gradients/Sub_1Subgradients/Switch_3:1gradients/GreaterEqual_1/Enter*
T0*
_output_shapes
: 
Î
gradients/NextIteration_3NextIterationgradients/Sub_1p^gradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_7Exitgradients/Switch_3*
T0*
_output_shapes
: 
w
&gradients/mean_cost_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

 gradients/mean_cost_grad/ReshapeReshapegradients/Fill&gradients/mean_cost_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
gradients/mean_cost_grad/ShapeShapesparse_softmax_fun/Reshape_2*
T0*
out_type0*
_output_shapes
:
¬
gradients/mean_cost_grad/TileTile gradients/mean_cost_grad/Reshapegradients/mean_cost_grad/Shape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tmultiples0
|
 gradients/mean_cost_grad/Shape_1Shapesparse_softmax_fun/Reshape_2*
T0*
out_type0*
_output_shapes
:
c
 gradients/mean_cost_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
h
gradients/mean_cost_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
¥
gradients/mean_cost_grad/ProdProd gradients/mean_cost_grad/Shape_1gradients/mean_cost_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
j
 gradients/mean_cost_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
©
gradients/mean_cost_grad/Prod_1Prod gradients/mean_cost_grad/Shape_2 gradients/mean_cost_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
d
"gradients/mean_cost_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

 gradients/mean_cost_grad/MaximumMaximumgradients/mean_cost_grad/Prod_1"gradients/mean_cost_grad/Maximum/y*
T0*
_output_shapes
: 

!gradients/mean_cost_grad/floordivFloorDivgradients/mean_cost_grad/Prod gradients/mean_cost_grad/Maximum*
T0*
_output_shapes
: 
x
gradients/mean_cost_grad/CastCast!gradients/mean_cost_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0

 gradients/mean_cost_grad/truedivRealDivgradients/mean_cost_grad/Tilegradients/mean_cost_grad/Cast*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

1gradients/sparse_softmax_fun/Reshape_2_grad/ShapeShape%sparse_softmax_fun/sparse_softmax_fun*
T0*
out_type0*
_output_shapes
:
Ï
3gradients/sparse_softmax_fun/Reshape_2_grad/ReshapeReshape gradients/mean_cost_grad/truediv1gradients/sparse_softmax_fun/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

gradients/zeros_like	ZerosLike'sparse_softmax_fun/sparse_softmax_fun:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ò
Dgradients/sparse_softmax_fun/sparse_softmax_fun_grad/PreventGradientPreventGradient'sparse_softmax_fun/sparse_softmax_fun:1*
T0*´
message¨¥Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

Cgradients/sparse_softmax_fun/sparse_softmax_fun_grad/ExpandDims/dimConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 

?gradients/sparse_softmax_fun/sparse_softmax_fun_grad/ExpandDims
ExpandDims3gradients/sparse_softmax_fun/Reshape_2_grad/ReshapeCgradients/sparse_softmax_fun/sparse_softmax_fun_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

8gradients/sparse_softmax_fun/sparse_softmax_fun_grad/mulMul?gradients/sparse_softmax_fun/sparse_softmax_fun_grad/ExpandDimsDgradients/sparse_softmax_fun/sparse_softmax_fun_grad/PreventGradient*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
x
/gradients/sparse_softmax_fun/Reshape_grad/ShapeShape	Reshape_1*
T0*
out_type0*
_output_shapes
:
ì
1gradients/sparse_softmax_fun/Reshape_grad/ReshapeReshape8gradients/sparse_softmax_fun/sparse_softmax_fun_grad/mul/gradients/sparse_softmax_fun/Reshape_grad/Shape*
T0*
Tshape0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
a
gradients/Reshape_1_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
¾
 gradients/Reshape_1_grad/ReshapeReshape1gradients/sparse_softmax_fun/Reshape_grad/Reshapegradients/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
^
gradients/add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:R*
dtype0*
_output_shapes
:
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
©
gradients/add_grad/SumSum gradients/Reshape_1_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
­
gradients/add_grad/Sum_1Sum gradients/Reshape_1_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:R
­
gradients/MatMul_grad/MatMulMatMulgradients/add_grad/Reshapeout_weights/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ

gradients/MatMul_grad/MatMul_1MatMulReshapegradients/add_grad/Reshape*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	ÆR
d
gradients/Reshape_grad/ShapeShapeconcat_3*
T0*
out_type0*
_output_shapes
:
«
gradients/Reshape_grad/ReshapeReshapegradients/MatMul_grad/MatMulgradients/Reshape_grad/Shape*
T0*
Tshape0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
^
gradients/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_3_grad/modFloorModconcat_3/axisgradients/concat_3_grad/Rank*
T0*
_output_shapes
: 
e
gradients/concat_3_grad/ShapeShapeconcat_2*
T0*
out_type0*
_output_shapes
:

gradients/concat_3_grad/ShapeNShapeNconcat_2concat_1concat*
T0*
out_type0*
N*&
_output_shapes
:::
æ
$gradients/concat_3_grad/ConcatOffsetConcatOffsetgradients/concat_3_grad/modgradients/concat_3_grad/ShapeN gradients/concat_3_grad/ShapeN:1 gradients/concat_3_grad/ShapeN:2*
N*&
_output_shapes
:::
Ñ
gradients/concat_3_grad/SliceSlicegradients/Reshape_grad/Reshape$gradients/concat_3_grad/ConcatOffsetgradients/concat_3_grad/ShapeN*
Index0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
×
gradients/concat_3_grad/Slice_1Slicegradients/Reshape_grad/Reshape&gradients/concat_3_grad/ConcatOffset:1 gradients/concat_3_grad/ShapeN:1*
Index0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
×
gradients/concat_3_grad/Slice_2Slicegradients/Reshape_grad/Reshape&gradients/concat_3_grad/ConcatOffset:2 gradients/concat_3_grad/ShapeN:2*
Index0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
^
gradients/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_2_grad/modFloorModconcat_2/axisgradients/concat_2_grad/Rank*
T0*
_output_shapes
: 
z
gradients/concat_2_grad/ShapeShapeconv_maxpool_3_conv_1/Reshape*
T0*
out_type0*
_output_shapes
:
Ï
gradients/concat_2_grad/ShapeNShapeNconv_maxpool_3_conv_1/Reshapeconv_maxpool_5_conv_1/Reshapeconv_maxpool_7_conv_1/Reshape*
T0*
out_type0*
N*&
_output_shapes
:::
æ
$gradients/concat_2_grad/ConcatOffsetConcatOffsetgradients/concat_2_grad/modgradients/concat_2_grad/ShapeN gradients/concat_2_grad/ShapeN:1 gradients/concat_2_grad/ShapeN:2*
N*&
_output_shapes
:::
Ï
gradients/concat_2_grad/SliceSlicegradients/concat_3_grad/Slice$gradients/concat_2_grad/ConcatOffsetgradients/concat_2_grad/ShapeN*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
gradients/concat_2_grad/Slice_1Slicegradients/concat_3_grad/Slice&gradients/concat_2_grad/ConcatOffset:1 gradients/concat_2_grad/ShapeN:1*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Õ
gradients/concat_2_grad/Slice_2Slicegradients/concat_3_grad/Slice&gradients/concat_2_grad/ConcatOffset:2 gradients/concat_2_grad/ShapeN:2*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

2gradients/conv_maxpool_3_conv_1/Reshape_grad/ShapeShape!conv_maxpool_3_conv_1/conv_nonlin*
T0*
out_type0*
_output_shapes
:
Û
4gradients/conv_maxpool_3_conv_1/Reshape_grad/ReshapeReshapegradients/concat_2_grad/Slice2gradients/conv_maxpool_3_conv_1/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

2gradients/conv_maxpool_5_conv_1/Reshape_grad/ShapeShape!conv_maxpool_5_conv_1/conv_nonlin*
T0*
out_type0*
_output_shapes
:
Ý
4gradients/conv_maxpool_5_conv_1/Reshape_grad/ReshapeReshapegradients/concat_2_grad/Slice_12gradients/conv_maxpool_5_conv_1/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

2gradients/conv_maxpool_7_conv_1/Reshape_grad/ShapeShape!conv_maxpool_7_conv_1/conv_nonlin*
T0*
out_type0*
_output_shapes
:
Ý
4gradients/conv_maxpool_7_conv_1/Reshape_grad/ReshapeReshapegradients/concat_2_grad/Slice_22gradients/conv_maxpool_7_conv_1/Reshape_grad/Shape*
T0*
Tshape0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ù
9gradients/conv_maxpool_3_conv_1/conv_nonlin_grad/ReluGradReluGrad4gradients/conv_maxpool_3_conv_1/Reshape_grad/Reshape!conv_maxpool_3_conv_1/conv_nonlin*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ù
9gradients/conv_maxpool_5_conv_1/conv_nonlin_grad/ReluGradReluGrad4gradients/conv_maxpool_5_conv_1/Reshape_grad/Reshape!conv_maxpool_5_conv_1/conv_nonlin*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Ù
9gradients/conv_maxpool_7_conv_1/conv_nonlin_grad/ReluGradReluGrad4gradients/conv_maxpool_7_conv_1/Reshape_grad/Reshape!conv_maxpool_7_conv_1/conv_nonlin*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
¾
8gradients/conv_maxpool_3_conv_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv_maxpool_3_conv_1/conv_nonlin_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
¾
8gradients/conv_maxpool_5_conv_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv_maxpool_5_conv_1/conv_nonlin_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
¾
8gradients/conv_maxpool_7_conv_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients/conv_maxpool_7_conv_1/conv_nonlin_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:d
Ã
3gradients/conv_maxpool_3_conv_1/conv_op_grad/ShapeNShapeNconv_maxpool_3_conv_1/Pad%conv_maxpool_3_conv_1/W_filter_0/read*
T0*
out_type0*
N* 
_output_shapes
::

2gradients/conv_maxpool_3_conv_1/conv_op_grad/ConstConst*%
valueB"        d   *
dtype0*
_output_shapes
:

@gradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/conv_maxpool_3_conv_1/conv_op_grad/ShapeN%conv_maxpool_3_conv_1/W_filter_0/read9gradients/conv_maxpool_3_conv_1/conv_nonlin_grad/ReluGrad*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
use_cudnn_on_gpu(

Agradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_maxpool_3_conv_1/Pad2gradients/conv_maxpool_3_conv_1/conv_op_grad/Const9gradients/conv_maxpool_3_conv_1/conv_nonlin_grad/ReluGrad*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*'
_output_shapes
:d*
use_cudnn_on_gpu(
Ã
3gradients/conv_maxpool_5_conv_1/conv_op_grad/ShapeNShapeNconv_maxpool_5_conv_1/Pad%conv_maxpool_5_conv_1/W_filter_1/read*
T0*
out_type0*
N* 
_output_shapes
::

2gradients/conv_maxpool_5_conv_1/conv_op_grad/ConstConst*%
valueB"        d   *
dtype0*
_output_shapes
:

@gradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/conv_maxpool_5_conv_1/conv_op_grad/ShapeN%conv_maxpool_5_conv_1/W_filter_1/read9gradients/conv_maxpool_5_conv_1/conv_nonlin_grad/ReluGrad*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
use_cudnn_on_gpu(

Agradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_maxpool_5_conv_1/Pad2gradients/conv_maxpool_5_conv_1/conv_op_grad/Const9gradients/conv_maxpool_5_conv_1/conv_nonlin_grad/ReluGrad*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*'
_output_shapes
:d*
use_cudnn_on_gpu(
Ã
3gradients/conv_maxpool_7_conv_1/conv_op_grad/ShapeNShapeNconv_maxpool_7_conv_1/Pad%conv_maxpool_7_conv_1/W_filter_2/read*
T0*
out_type0*
N* 
_output_shapes
::

2gradients/conv_maxpool_7_conv_1/conv_op_grad/ConstConst*%
valueB"        d   *
dtype0*
_output_shapes
:

@gradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/conv_maxpool_7_conv_1/conv_op_grad/ShapeN%conv_maxpool_7_conv_1/W_filter_2/read9gradients/conv_maxpool_7_conv_1/conv_nonlin_grad/ReluGrad*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
use_cudnn_on_gpu(

Agradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_maxpool_7_conv_1/Pad2gradients/conv_maxpool_7_conv_1/conv_op_grad/Const9gradients/conv_maxpool_7_conv_1/conv_nonlin_grad/ReluGrad*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*'
_output_shapes
:d*
use_cudnn_on_gpu(
o
-gradients/conv_maxpool_3_conv_1/Pad_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
r
0gradients/conv_maxpool_3_conv_1/Pad_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ñ
.gradients/conv_maxpool_3_conv_1/Pad_grad/stackPack-gradients/conv_maxpool_3_conv_1/Pad_grad/Rank0gradients/conv_maxpool_3_conv_1/Pad_grad/stack/1*

axis *
T0*
N*
_output_shapes
:

4gradients/conv_maxpool_3_conv_1/Pad_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
÷
.gradients/conv_maxpool_3_conv_1/Pad_grad/SliceSlice"conv_maxpool_3_conv_1/Pad/paddings4gradients/conv_maxpool_3_conv_1/Pad_grad/Slice/begin.gradients/conv_maxpool_3_conv_1/Pad_grad/stack*
Index0*
T0*
_output_shapes

:

6gradients/conv_maxpool_3_conv_1/Pad_grad/Reshape/shapeConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
Ö
0gradients/conv_maxpool_3_conv_1/Pad_grad/ReshapeReshape.gradients/conv_maxpool_3_conv_1/Pad_grad/Slice6gradients/conv_maxpool_3_conv_1/Pad_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
x
.gradients/conv_maxpool_3_conv_1/Pad_grad/ShapeShape
ExpandDims*
T0*
out_type0*
_output_shapes
:
¦
0gradients/conv_maxpool_3_conv_1/Pad_grad/Slice_1Slice@gradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropInput0gradients/conv_maxpool_3_conv_1/Pad_grad/Reshape.gradients/conv_maxpool_3_conv_1/Pad_grad/Shape*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
-gradients/conv_maxpool_5_conv_1/Pad_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
r
0gradients/conv_maxpool_5_conv_1/Pad_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ñ
.gradients/conv_maxpool_5_conv_1/Pad_grad/stackPack-gradients/conv_maxpool_5_conv_1/Pad_grad/Rank0gradients/conv_maxpool_5_conv_1/Pad_grad/stack/1*

axis *
T0*
N*
_output_shapes
:

4gradients/conv_maxpool_5_conv_1/Pad_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
÷
.gradients/conv_maxpool_5_conv_1/Pad_grad/SliceSlice"conv_maxpool_5_conv_1/Pad/paddings4gradients/conv_maxpool_5_conv_1/Pad_grad/Slice/begin.gradients/conv_maxpool_5_conv_1/Pad_grad/stack*
Index0*
T0*
_output_shapes

:

6gradients/conv_maxpool_5_conv_1/Pad_grad/Reshape/shapeConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
Ö
0gradients/conv_maxpool_5_conv_1/Pad_grad/ReshapeReshape.gradients/conv_maxpool_5_conv_1/Pad_grad/Slice6gradients/conv_maxpool_5_conv_1/Pad_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
x
.gradients/conv_maxpool_5_conv_1/Pad_grad/ShapeShape
ExpandDims*
T0*
out_type0*
_output_shapes
:
¦
0gradients/conv_maxpool_5_conv_1/Pad_grad/Slice_1Slice@gradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropInput0gradients/conv_maxpool_5_conv_1/Pad_grad/Reshape.gradients/conv_maxpool_5_conv_1/Pad_grad/Shape*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
o
-gradients/conv_maxpool_7_conv_1/Pad_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
r
0gradients/conv_maxpool_7_conv_1/Pad_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ñ
.gradients/conv_maxpool_7_conv_1/Pad_grad/stackPack-gradients/conv_maxpool_7_conv_1/Pad_grad/Rank0gradients/conv_maxpool_7_conv_1/Pad_grad/stack/1*

axis *
T0*
N*
_output_shapes
:

4gradients/conv_maxpool_7_conv_1/Pad_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
÷
.gradients/conv_maxpool_7_conv_1/Pad_grad/SliceSlice"conv_maxpool_7_conv_1/Pad/paddings4gradients/conv_maxpool_7_conv_1/Pad_grad/Slice/begin.gradients/conv_maxpool_7_conv_1/Pad_grad/stack*
Index0*
T0*
_output_shapes

:

6gradients/conv_maxpool_7_conv_1/Pad_grad/Reshape/shapeConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
Ö
0gradients/conv_maxpool_7_conv_1/Pad_grad/ReshapeReshape.gradients/conv_maxpool_7_conv_1/Pad_grad/Slice6gradients/conv_maxpool_7_conv_1/Pad_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
x
.gradients/conv_maxpool_7_conv_1/Pad_grad/ShapeShape
ExpandDims*
T0*
out_type0*
_output_shapes
:
¦
0gradients/conv_maxpool_7_conv_1/Pad_grad/Slice_1Slice@gradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropInput0gradients/conv_maxpool_7_conv_1/Pad_grad/Reshape.gradients/conv_maxpool_7_conv_1/Pad_grad/Shape*
Index0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¶
gradients/AddNAddN0gradients/conv_maxpool_3_conv_1/Pad_grad/Slice_10gradients/conv_maxpool_5_conv_1/Pad_grad/Slice_10gradients/conv_maxpool_7_conv_1/Pad_grad/Slice_1*
T0*
N*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*C
_class9
75loc:@gradients/conv_maxpool_3_conv_1/Pad_grad/Slice_1
g
gradients/ExpandDims_grad/ShapeShapeconcat_1*
T0*
out_type0*
_output_shapes
:
£
!gradients/ExpandDims_grad/ReshapeReshapegradients/AddNgradients/ExpandDims_grad/Shape*
T0*
Tshape0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ñ
gradients/AddN_1AddNgradients/concat_3_grad/Slice_1!gradients/ExpandDims_grad/Reshape*
T0*
N*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_class(
&$loc:@gradients/concat_3_grad/Slice_1
^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
u
gradients/concat_1_grad/modFloorModconcat_1/axisgradients/concat_1_grad/Rank*
T0*
_output_shapes
: 

gradients/concat_1_grad/ShapeShape)bidi_/bidirectional_rnn/fw/fw/transpose_1*
T0*
out_type0*
_output_shapes
:
®
gradients/concat_1_grad/ShapeNShapeN)bidi_/bidirectional_rnn/fw/fw/transpose_1bidi_/ReverseSequence*
T0*
out_type0*
N* 
_output_shapes
::
¾
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/modgradients/concat_1_grad/ShapeN gradients/concat_1_grad/ShapeN:1*
N* 
_output_shapes
::
Ã
gradients/concat_1_grad/SliceSlicegradients/AddN_1$gradients/concat_1_grad/ConcatOffsetgradients/concat_1_grad/ShapeN*
Index0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
É
gradients/concat_1_grad/Slice_1Slicegradients/AddN_1&gradients/concat_1_grad/ConcatOffset:1 gradients/concat_1_grad/ShapeN:1*
Index0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¬
Jgradients/bidi_/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutationInvertPermutation&bidi_/bidirectional_rnn/fw/fw/concat_2*
T0*
_output_shapes
:
ÿ
Bgradients/bidi_/bidirectional_rnn/fw/fw/transpose_1_grad/transpose	Transposegradients/concat_1_grad/SliceJgradients/bidi_/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutation*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Õ
4gradients/bidi_/ReverseSequence_grad/ReverseSequenceReverseSequencegradients/concat_1_grad/Slice_1seqlens*
seq_dim*
T0*

Tlen0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
	batch_dim 
Ò
sgradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3)bidi_/bidirectional_rnn/fw/fw/TensorArray*bidi_/bidirectional_rnn/fw/fw/while/Exit_2*
_output_shapes

:: *
source	gradients*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray
ü
ogradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity*bidi_/bidirectional_rnn/fw/fw/while/Exit_2t^gradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray

ygradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3sgradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV34bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/rangeBgradients/bidi_/bidirectional_rnn/fw/fw/transpose_1_grad/transposeogradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
¬
Jgradients/bidi_/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutationInvertPermutation&bidi_/bidirectional_rnn/bw/bw/concat_2*
T0*
_output_shapes
:

Bgradients/bidi_/bidirectional_rnn/bw/bw/transpose_1_grad/transpose	Transpose4gradients/bidi_/ReverseSequence_grad/ReverseSequenceJgradients/bidi_/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutation*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ò
sgradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3)bidi_/bidirectional_rnn/bw/bw/TensorArray*bidi_/bidirectional_rnn/bw/bw/while/Exit_2*
_output_shapes

:: *
source	gradients*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray
ü
ogradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity*bidi_/bidirectional_rnn/bw/bw/while/Exit_2t^gradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray

ygradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3sgradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV34bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/rangeBgradients/bidi_/bidirectional_rnn/bw/bw/transpose_1_grad/transposeogradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 

gradients/zeros_like_1	ZerosLike*bidi_/bidirectional_rnn/fw/fw/while/Exit_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

gradients/zeros_like_2	ZerosLike*bidi_/bidirectional_rnn/fw/fw/while/Exit_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
à
@gradients/bidi_/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEnterygradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes
: 

@gradients/bidi_/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

@gradients/bidi_/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEntergradients/zeros_like_2*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

gradients/zeros_like_3	ZerosLike*bidi_/bidirectional_rnn/bw/bw/while/Exit_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

gradients/zeros_like_4	ZerosLike*bidi_/bidirectional_rnn/bw/bw/while/Exit_4*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchMerge@gradients/bidi_/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitKgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 

Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchMerge@gradients/bidi_/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitKgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 

Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchMerge@gradients/bidi_/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitKgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
à
@gradients/bidi_/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEnterygradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes
: 

@gradients/bidi_/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEntergradients/zeros_like_3*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

@gradients/bidi_/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEntergradients/zeros_like_4*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¢
Agradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchSwitchDgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*
_output_shapes
: : *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch
Æ
Agradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchSwitchDgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch
Æ
Agradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchSwitchDgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchgradients/b_count_2*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch

Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchMerge@gradients/bidi_/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitKgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 

Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchMerge@gradients/bidi_/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitKgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 

Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchMerge@gradients/bidi_/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitKgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
«
?gradients/bidi_/bidirectional_rnn/fw/fw/while/Enter_2_grad/ExitExitAgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch*
T0*
_output_shapes
: 
½
?gradients/bidi_/bidirectional_rnn/fw/fw/while/Enter_3_grad/ExitExitAgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
½
?gradients/bidi_/bidirectional_rnn/fw/fw/while/Enter_4_grad/ExitExitAgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¢
Agradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchSwitchDgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchgradients/b_count_6*
T0*
_output_shapes
: : *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch
Æ
Agradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchSwitchDgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchgradients/b_count_6*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch
Æ
Agradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchSwitchDgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchgradients/b_count_6*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch
Ë
xgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3~gradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterCgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1*
_output_shapes

:: *
source	gradients*B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/mul

~gradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter)bidi_/bidirectional_rnn/fw/fw/TensorArray*
is_constant(*B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/mul*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
¥
tgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityCgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1y^gradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/mul

hgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3xgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3sgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2tgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ü
ngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *A
_class7
53loc:@bidi_/bidirectional_rnn/fw/fw/while/Identity_1
ë
ngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2ngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*
_output_shapes
:*

stack_name *A
_class7
53loc:@bidi_/bidirectional_rnn/fw/fw/while/Identity_1
ý
ngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ç
tgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2ngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter.bidi_/bidirectional_rnn/fw/fw/while/Identity_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
: 
µ
sgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2ygradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
: 

ygradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ß
ogradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerN^gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2R^gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2R^gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2t^gradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2b^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2`^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2P^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2X^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2d^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2f^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1b^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2V^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2d^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2f^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1R^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2T^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2d^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2f^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1R^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2T^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2b^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2d^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1R^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
³
bgradients/bidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ê
`gradients/bidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/SumSum?gradients/bidi_/bidirectional_rnn/fw/fw/while/Enter_3_grad/Exitbgradients/bidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Ù
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like	ZerosLikeQgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ú
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *A
_class7
53loc:@bidi_/bidirectional_rnn/fw/fw/while/Identity_3
§
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_accStackV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Const*
	elem_type0*
_output_shapes
:*

stack_name *A
_class7
53loc:@bidi_/bidirectional_rnn/fw/fw/while/Identity_3
¹
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
µ
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter.bidi_/bidirectional_rnn/fw/fw/while/Identity_3^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Û
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectSelectMgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Cgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch:1Fgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ø
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *C
_class9
75loc:@bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual
¡
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_accStackV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Const*
	elem_type0
*
_output_shapes
:*

stack_name *C
_class9
75loc:@bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual
±
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/EnterEnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ª
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2StackPushV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter0bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual^gradients/Add*
T0
*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ö
Mgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2
StackPopV2Sgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub*
	elem_type0
*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Æ
Sgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/EnterEnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Ý
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1SelectMgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Fgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_likeCgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
µ
dgradients/bidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Î
bgradients/bidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/SumSum?gradients/bidi_/bidirectional_rnn/fw/fw/while/Enter_4_grad/Exitdgradients/bidi_/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Ù
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like	ZerosLikeQgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ú
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *A
_class7
53loc:@bidi_/bidirectional_rnn/fw/fw/while/Identity_4
§
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_accStackV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Const*
	elem_type0*
_output_shapes
:*

stack_name *A
_class7
53loc:@bidi_/bidirectional_rnn/fw/fw/while/Identity_4
¹
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
µ
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter.bidi_/bidirectional_rnn/fw/fw/while/Identity_4^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Û
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectSelectMgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Cgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch:1Fgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ý
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1SelectMgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Fgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_likeCgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
«
?gradients/bidi_/bidirectional_rnn/bw/bw/while/Enter_2_grad/ExitExitAgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch*
T0*
_output_shapes
: 
½
?gradients/bidi_/bidirectional_rnn/bw/bw/while/Enter_3_grad/ExitExitAgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
½
?gradients/bidi_/bidirectional_rnn/bw/bw/while/Enter_4_grad/ExitExitAgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
à
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like	ZerosLikeJgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/Enter^gradients/Sub*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¦
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/EnterEnter#bidi_/bidirectional_rnn/fw/fw/zeros*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ü
@gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/SelectSelectMgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2hgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
þ
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/Select_1SelectMgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Dgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likehgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ë
xgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3~gradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterCgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1*
_output_shapes

:: *
source	gradients*B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/mul

~gradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter)bidi_/bidirectional_rnn/bw/bw/TensorArray*
is_constant(*B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/mul*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
¥
tgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityCgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1y^gradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/mul

hgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3xgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3sgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2tgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ü
ngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *A
_class7
53loc:@bidi_/bidirectional_rnn/bw/bw/while/Identity_1
ë
ngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2ngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*
_output_shapes
:*

stack_name *A
_class7
53loc:@bidi_/bidirectional_rnn/bw/bw/while/Identity_1
ý
ngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
é
tgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2ngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter.bidi_/bidirectional_rnn/bw/bw/while/Identity_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
: 
·
sgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2ygradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
: 

ygradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ß
ogradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerN^gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2R^gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2R^gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2t^gradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2`^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2b^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2`^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2b^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1N^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2P^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2X^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2d^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2f^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1b^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2V^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2d^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2f^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1R^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2T^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2d^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2f^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1R^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2T^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2b^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2d^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1R^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
³
bgradients/bidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ê
`gradients/bidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/SumSum?gradients/bidi_/bidirectional_rnn/bw/bw/while/Enter_3_grad/Exitbgradients/bidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Ù
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like	ZerosLikeQgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ú
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *A
_class7
53loc:@bidi_/bidirectional_rnn/bw/bw/while/Identity_3
§
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_accStackV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Const*
	elem_type0*
_output_shapes
:*

stack_name *A
_class7
53loc:@bidi_/bidirectional_rnn/bw/bw/while/Identity_3
¹
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
·
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter.bidi_/bidirectional_rnn/bw/bw/while/Identity_3^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
Û
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectSelectMgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Cgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch:1Fgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ø
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *C
_class9
75loc:@bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual
¡
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_accStackV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Const*
	elem_type0
*
_output_shapes
:*

stack_name *C
_class9
75loc:@bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual
±
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/EnterEnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¬
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2StackPushV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter0bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual^gradients/Add_1*
T0
*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ø
Mgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2
StackPopV2Sgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Æ
Sgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/EnterEnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
Ý
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1SelectMgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Fgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_likeCgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
µ
dgradients/bidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Î
bgradients/bidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/SumSum?gradients/bidi_/bidirectional_rnn/bw/bw/while/Enter_4_grad/Exitdgradients/bidi_/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Ù
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like	ZerosLikeQgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ú
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *A
_class7
53loc:@bidi_/bidirectional_rnn/bw/bw/while/Identity_4
§
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_accStackV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Const*
	elem_type0*
_output_shapes
:*

stack_name *A
_class7
53loc:@bidi_/bidirectional_rnn/bw/bw/while/Identity_4
¹
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
·
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter.bidi_/bidirectional_rnn/bw/bw/while/Identity_4^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
Û
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectSelectMgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Cgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch:1Fgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ý
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1SelectMgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Fgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_likeCgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¨
Egradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/ShapeShape#bidi_/bidirectional_rnn/fw/fw/zeros*
T0*
out_type0*
_output_shapes
:

Kgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¦
Egradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zerosFillEgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/ShapeKgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zeros/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0
Ã
Egradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_accEnterEgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zeros*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
Ggradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_1MergeEgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_accMgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/NextIteration*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
õ
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/SwitchSwitchGgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_1gradients/b_count_2*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ

Cgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/AddAddHgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/Switch:1@gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/Select*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ö
Mgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/NextIterationNextIterationCgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/Add*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ê
Ggradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_2ExitFgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¼
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ShapeShape/bidi_/bidirectional_rnn/fw/fw/while/dropout/div*
T0*
out_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1Shape1bidi_/bidirectional_rnn/fw/fw/while/dropout/Floor*
T0*
out_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
þ
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape
Ù
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape
Õ
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
â
`gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape^gradients/Add*
T0*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ê
egradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1
ß
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1
Ù
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
è
bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1^gradients/Add*
T0*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

agradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
ggradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ÿ
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulMulBgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/Select_1Mgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:
Ù
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *D
_class:
86loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/Floor
¢
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_accStackV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *D
_class:
86loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/Floor
±
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/EnterEnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
 
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter1bidi_/bidirectional_rnn/fw/fw/while/dropout/Floor^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
ë
Mgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Sgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
Æ
Sgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
£
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/SumSumBgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/MulTgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
§
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeReshapeBgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*
_output_shapes
:

Dgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1MulOgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2Bgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_grad/Select_1*
T0*
_output_shapes
:
Ù
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/div
¤
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_accStackV2Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *B
_class8
64loc:@bidi_/bidirectional_rnn/fw/fw/while/dropout/div
µ
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/EnterEnterJgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
¢
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter/bidi_/bidirectional_rnn/fw/fw/while/dropout/div^gradients/Add*
T0*
swap_memory( *
_output_shapes
:
ï
Ogradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Ugradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
Ê
Ugradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterJgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
©
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1SumDgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1Vgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
­
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Reshape_1ReshapeDgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Sum_1agradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*
_output_shapes
:
Â
Kgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIterationNextIterationCgradients/bidi_/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1*
T0*
_output_shapes
: 
â
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like	ZerosLikeJgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/Enter^gradients/Sub_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¦
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/EnterEnter#bidi_/bidirectional_rnn/bw/bw/zeros*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ü
@gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/SelectSelectMgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2hgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
þ
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/Select_1SelectMgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Dgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likehgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

8gradients/bidi_/bidirectional_rnn/fw/fw/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
þ
6gradients/bidi_/bidirectional_rnn/fw/fw/zeros_grad/SumSumGgradients/bidi_/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_28gradients/bidi_/bidirectional_rnn/fw/fw/zeros_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
·
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/ShapeShape3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
ó
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1Shape5bidi_/bidirectional_rnn/fw/fw/while/dropout/add/Enter-^bidi_/bidirectional_rnn/fw/fw/while/Identity*
T0*
out_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2agradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
þ
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape
Ù
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_accStackV2Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape
Õ
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Ù
`gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/EnterDgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

_gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ê
egradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1
ß
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1StackV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1
Ù
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1Enter\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
è
bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1^gradients/Add*
T0*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

agradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
ggradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDivRealDivFgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeLgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:

Lgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/EnterEnterdropout_keep_prob*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
§
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/SumSumFgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDivTgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
·
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/ReshapeReshapeBgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Sum_gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ë
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/NegNegMgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Û
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *F
_class<
:8loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2
¤
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_accStackV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Const*
	elem_type0*
_output_shapes
:*

stack_name *F
_class<
:8loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2
±
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/EnterEnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
²
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Enter3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
û
Mgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Æ
Sgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_1RealDivBgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/NegLgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:

Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_2RealDivHgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_1Lgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:
þ
Bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/mulMulFgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/ReshapeHgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Sum_1SumBgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/mulVgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
­
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Reshape_1ReshapeDgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Sum_1agradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*
_output_shapes
:
¨
Egradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/ShapeShape#bidi_/bidirectional_rnn/bw/bw/zeros*
T0*
out_type0*
_output_shapes
:

Kgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¦
Egradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zerosFillEgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/ShapeKgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zeros/Const*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*

index_type0
Ã
Egradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_accEnterEgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zeros*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¤
Ggradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_1MergeEgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_accMgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/NextIteration*
T0*
N**
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ: 
õ
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/SwitchSwitchGgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_1gradients/b_count_6*
T0*<
_output_shapes*
(:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ

Cgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/AddAddHgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/Switch:1@gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/Select*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ö
Mgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/NextIterationNextIterationCgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/Add*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ê
Ggradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_2ExitFgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¼
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ShapeShape/bidi_/bidirectional_rnn/bw/bw/while/dropout/div*
T0*
out_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1Shape1bidi_/bidirectional_rnn/bw/bw/while/dropout/Floor*
T0*
out_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2agradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
þ
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape
Ù
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_accStackV2Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape
Õ
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/EnterEnterZgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ä
`gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/EnterDgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape^gradients/Add_1*
T0*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ê
egradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1
ß
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1StackV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1
Ù
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Enter\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ê
bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
agradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
ggradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ÿ
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulMulBgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/Select_1Mgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:
Ù
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *D
_class:
86loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/Floor
¢
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_accStackV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *D
_class:
86loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/Floor
±
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/EnterEnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¢
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2StackPushV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter1bidi_/bidirectional_rnn/bw/bw/while/dropout/Floor^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
í
Mgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2
StackPopV2Sgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
Æ
Sgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPopV2/EnterEnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
£
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/SumSumBgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/MulTgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
§
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeReshapeBgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*
_output_shapes
:

Dgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1MulOgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2Bgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_grad/Select_1*
T0*
_output_shapes
:
Ù
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/div
¤
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_accStackV2Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *B
_class8
64loc:@bidi_/bidirectional_rnn/bw/bw/while/dropout/div
µ
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/EnterEnterJgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¤
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV2Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter/bidi_/bidirectional_rnn/bw/bw/while/dropout/div^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:
ñ
Ogradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2Ugradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
Ê
Ugradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnterJgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
©
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1SumDgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1Vgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
­
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Reshape_1ReshapeDgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Sum_1agradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*
_output_shapes
:
Â
Kgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIterationNextIterationCgradients/bidi_/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1*
T0*
_output_shapes
: 

8gradients/bidi_/bidirectional_rnn/bw/bw/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
þ
6gradients/bidi_/bidirectional_rnn/bw/bw/zeros_grad/SumSumGgradients/bidi_/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_28gradients/bidi_/bidirectional_rnn/bw/bw/zeros_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
·
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/ShapeShape3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
T0*
out_type0*
_output_shapes
:
ó
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1Shape5bidi_/bidirectional_rnn/bw/bw/while/dropout/add/Enter-^bidi_/bidirectional_rnn/bw/bw/while/Identity*
T0*
out_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs_gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2agradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
þ
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape
Ù
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_accStackV2Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape
Õ
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/EnterEnterZgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
Û
`gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2StackPushV2Zgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/EnterDgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

_gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2
StackPopV2egradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ê
egradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2/EnterEnterZgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1
ß
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1StackV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1
Ù
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1Enter\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ê
bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
agradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2ggradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
î
ggradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDivRealDivFgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeLgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:

Lgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/EnterEnterdropout_keep_prob*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
§
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/SumSumFgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDivTgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
·
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/ReshapeReshapeBgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Sum_gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ë
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/NegNegMgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Û
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *F
_class<
:8loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
¤
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_accStackV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Const*
	elem_type0*
_output_shapes
:*

stack_name *F
_class<
:8loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
±
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/EnterEnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
´
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPushV2StackPushV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Enter3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ý
Mgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2
StackPopV2Sgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Æ
Sgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPopV2/EnterEnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_1RealDivBgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/NegLgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:

Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_2RealDivHgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_1Lgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv/Enter*
T0*
_output_shapes
:
þ
Bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/mulMulFgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/ReshapeHgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
§
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Sum_1SumBgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/mulVgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
­
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Reshape_1ReshapeDgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Sum_1agradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*
_output_shapes
:
»
gradients/AddN_2AddNDgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Reshape*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1
¿
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ShapeShape7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
¾
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1Shape4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
ú
Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape
å
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape
Ý
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
å
dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1
ë
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1
á
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enter`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ë
fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ö
kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
å
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/MulMulgradients/AddN_2Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
à
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1
­
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1
¹
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
»
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
¯
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/SumSumFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/MulXgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeReshapeFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sumcgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
é
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1MulSgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
å
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2
´
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2
½
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterNgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Â
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Sgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ò
Ygradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
µ
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1SumHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1Zgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
É
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1ReshapeHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
µ
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradSgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¬
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
»
gradients/AddN_3AddNDgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Reshape*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1
¿
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ShapeShape7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
¾
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1Shape4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
ú
Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape
å
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape
Ý
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ç
dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1
ë
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1
á
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enter`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
í
fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ö
kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
å
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/MulMulgradients/AddN_3Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
à
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1
­
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1
¹
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
½
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¯
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/SumSumFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/MulXgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeReshapeFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sumcgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
é
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1MulSgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
å
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2
´
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2
½
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterNgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
Ä
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Sgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ò
Ygradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
µ
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1SumHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1Zgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
É
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1ReshapeHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Á
gradients/AddN_4AddNDgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGrad*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
¹
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ShapeShape1bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
½
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1Shape3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
ú
Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape
å
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape
Ý
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
å
dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1
ë
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1
á
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enter`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ë
fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ö
kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ù
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_4Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeReshapeFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sumcgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ý
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_4Zgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
É
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1ReshapeHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
µ
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradSgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¬
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
»
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ShapeShape5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
¶
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1Shape.bidi_/bidirectional_rnn/fw/fw/while/Identity_3*
T0*
out_type0*
_output_shapes
:
ô
Vgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2cgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape
ß
\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape
Ù
\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ß
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

agradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
î
ggradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1
å
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1
Ý
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
å
dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

Dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/MulMulJgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeQgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
©
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/SumSumDgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/MulVgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
½
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeReshapeDgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sumagradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1MulQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
á
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *H
_class>
<:loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid
®
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *H
_class>
<:loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid
¹
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
¼
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
¯
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1SumFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1ReshapeFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1cgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¿
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ShapeShape7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
¼
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1Shape2bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
ú
Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape
å
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape
Ý
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
å
dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1
ë
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1
á
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enter`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ë
fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
ö
kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
¡
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/MulMulLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Þ
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *E
_class;
97loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh
«
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *E
_class;
97loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh
¹
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
¹
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter2bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
¯
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/SumSumFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/MulXgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeReshapeFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sumcgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¥
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1MulSgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
å
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1
´
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1
½
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterNgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Â
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Sgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ò
Ygradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
µ
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1SumHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1Zgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
É
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1ReshapeHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1egradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Á
gradients/AddN_5AddNDgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGrad*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*W
_classM
KIloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
¹
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ShapeShape1bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
½
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1Shape3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
ú
Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape
å
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape
Ý
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ç
dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1
ë
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1
á
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enter`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
í
fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ö
kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ù
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_5Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeReshapeFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sumcgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ý
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_5Zgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
É
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1ReshapeHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¯
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
»
gradients/AddN_6AddNBgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectJgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*U
_classK
IGloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select
µ
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ª
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
»
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ShapeShape5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
¶
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1Shape.bidi_/bidirectional_rnn/bw/bw/while/Identity_3*
T0*
out_type0*
_output_shapes
:
ô
Vgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2cgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape
ß
\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape
Ù
\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
á
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

agradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
î
ggradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1
å
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1
Ý
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ç
dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

Dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/MulMulJgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeQgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
©
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/SumSumDgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/MulVgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
½
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeReshapeDgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sumagradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1MulQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
á
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *H
_class>
<:loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid
®
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *H
_class>
<:loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid
¹
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¾
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¯
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1SumFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1ReshapeFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1cgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¿
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ShapeShape7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
¼
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1Shape2bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
ú
Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgscgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape
å
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *[
_classQ
OMloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape
Ý
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
ç
dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

cgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ò
igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1
ë
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*
_output_shapes
:*

stack_name *]
_classS
QOloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1
á
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enter`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
í
fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
ö
kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¡
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/MulMulLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Þ
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *E
_class;
97loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh
«
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *E
_class;
97loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh
¹
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
»
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter2bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Î
Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
¯
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/SumSumFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/MulXgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ã
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeReshapeFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sumcgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¥
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1MulSgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
å
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1
´
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1
½
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterNgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
Ä
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

Sgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Ò
Ygradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
µ
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1SumHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1Zgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
É
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1ReshapeHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1egradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
»
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ShapeShape5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:

Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
Ù
Vgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape
ß
\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape
Ù
\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
ß
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape^gradients/Add*
T0*
swap_memory( *
_output_shapes
:

agradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:
î
ggradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
µ
Dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/SumSumPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradVgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
½
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeReshapeDgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sumagradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¹
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1SumPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1ReshapeFgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
¡
Kgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_6*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¯
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
»
gradients/AddN_7AddNBgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectJgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*U
_classK
IGloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select
µ
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ª
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¬
Igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatConcatV2Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGradHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/Const*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *

Tidx0
¡
Ogradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
»
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ShapeShape5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:

Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
Ù
Vgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsagradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ

\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape
ß
\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*
	elem_type0*
_output_shapes
:*

stack_name *Y
_classO
MKloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape
Ù
\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
á
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape^gradients/Add_1*
T0*
swap_memory( *
_output_shapes
:

agradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ggradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
î
ggradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter\gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
µ
Dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/SumSumPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradVgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
½
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeReshapeDgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sumagradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¹
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1SumPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1ReshapeFgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
¡
Kgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_7*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ç
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
: 
¬
Igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatConcatV2Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGradHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/Const*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *

Tidx0
£
Ogradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 
Ê
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulMatMulIgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
±
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/read*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(* 
_output_shapes
:
Ò 
Ë
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2Igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:
Ò 
æ
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat
¹
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat
Å
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Ç
Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ

Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
Ú
]gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:

Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB *    *
dtype0*
_output_shapes	
: 
Î
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes	
: 
º
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	: : 
ñ
Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
: : 
¢
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
: 
ß
Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
: 
Ó
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
: 
ç
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
: 

Igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Ggradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modFloorModIgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstHgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
¾
Igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeShape5bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
Â
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNShapeNUgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Qgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
å
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *H
_class>
<:loc:@bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3
¶
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Const*
	elem_type0*
_output_shapes
:*

stack_name *H
_class>
<:loc:@bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3
Á
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/EnterEnterPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
Ä
Vgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter5bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3^gradients/Add*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Ugradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2[gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
[gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant(*
_output_shapes
:
î
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modJgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNLgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
ü
Igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceSliceJgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetJgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1SliceJgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffset:1Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¨
Ogradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
Ò *    *
dtype0* 
_output_shapes
:
Ò 
Ñ
Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( * 
_output_shapes
:
Ò 
¼
Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
Ò : 
ù
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
Ò :
Ò 
¡
Mgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/AddAddRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch:1Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
Ò 
â
Wgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
Ò 
Ö
Qgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
Ò 
Ê
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulMatMulIgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
±
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/read*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(* 
_output_shapes
:
Ò 
Ë
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2Igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:
Ò 
æ
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat
¹
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat
Å
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
É
Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ

Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
Ú
]gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:

Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB *    *
dtype0*
_output_shapes	
: 
Î
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes	
: 
º
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	: : 
ñ
Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6*
T0*"
_output_shapes
: : 
¢
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
: 
ß
Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
: 
Ó
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
: 
î
fgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
_output_shapes

:: *
source	gradients*N
_classD
B@loc:@bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter

lgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter+bidi_/bidirectional_rnn/fw/fw/TensorArray_1*
is_constant(*N
_classD
B@loc:@bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
½
ngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterXbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*N
_classD
B@loc:@bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
¸
bgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityngradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1g^gradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *N
_classD
B@loc:@bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter
«
hgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3fgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3sgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Igradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slicebgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
¼
gradients/AddN_8AddNBgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectKgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*U
_classK
IGloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/Select

Igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 

Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/RankConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 

Ggradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modFloorModIgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstHgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
¾
Igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeShape5bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
Â
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNShapeNUgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Qgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
å
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *H
_class>
<:loc:@bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3
¶
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Const*
	elem_type0*
_output_shapes
:*

stack_name *H
_class>
<:loc:@bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3
Á
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/EnterEnterPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
parallel_iterations *A

frame_name31bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
Æ
Vgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter5bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3^gradients/Add_1*
T0*
swap_memory( *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Ugradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2[gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
[gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant(*
_output_shapes
:
î
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modJgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNLgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
ü
Igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceSliceJgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetJgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1SliceJgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffset:1Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
¨
Ogradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
Ò *    *
dtype0* 
_output_shapes
:
Ò 
Ñ
Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( * 
_output_shapes
:
Ò 
¼
Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
Ò : 
ù
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_6*
T0*,
_output_shapes
:
Ò :
Ò 
¡
Mgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/AddAddRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch:1Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
Ò 
â
Wgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
Ò 
Ö
Qgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
Ò 

Rgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Í
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterRgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/fw/fw/while/while_context*
is_constant( *
_output_shapes
: 
»
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeTgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Zgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
ë
Sgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchTgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0*
_output_shapes
: : 
¹
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/AddAddUgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch:1hgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Þ
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationPgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
Ò
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitSgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
¡
Kgradients/bidi_/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_8*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
ð
fgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enterngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_1*
_output_shapes

:: *
source	gradients*N
_classD
B@loc:@bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter

lgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter+bidi_/bidirectional_rnn/bw/bw/TensorArray_1*
is_constant(*N
_classD
B@loc:@bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
½
ngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterXbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*N
_classD
B@loc:@bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
¸
bgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityngradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1g^gradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *N
_classD
B@loc:@bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter
«
hgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3fgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3sgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Igradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slicebgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
¼
gradients/AddN_9AddNBgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectKgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1*
T0*
N*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*U
_classK
IGloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/Select

gradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3+bidi_/bidirectional_rnn/fw/fw/TensorArray_1Tgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *
source	gradients*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray_1
Ö
gradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityTgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^gradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/fw/TensorArray_1
±
{gradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3gradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV36bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangegradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Rgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Í
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterRgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
parallel_iterations *K

frame_name=;gradients/bidi_/bidirectional_rnn/bw/bw/while/while_context*
is_constant( *
_output_shapes
: 
»
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeTgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Zgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
ë
Sgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchTgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_6*
T0*
_output_shapes
: : 
¹
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/AddAddUgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch:1hgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Þ
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationPgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
Ò
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3ExitSgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
¡
Kgradients/bidi_/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_9*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ

gradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3+bidi_/bidirectional_rnn/bw/bw/TensorArray_1Tgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes

:: *
source	gradients*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray_1
Ö
gradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityTgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3^gradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/bw/TensorArray_1
±
{gradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3gradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV36bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangegradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
Hgradients/bidi_/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutationInvertPermutation$bidi_/bidirectional_rnn/fw/fw/concat*
T0*
_output_shapes
:
Ù
@gradients/bidi_/bidirectional_rnn/fw/fw/transpose_grad/transpose	Transpose{gradients/bidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3Hgradients/bidi_/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutation*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¨
Hgradients/bidi_/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutationInvertPermutation$bidi_/bidirectional_rnn/bw/bw/concat*
T0*
_output_shapes
:
Ù
@gradients/bidi_/bidirectional_rnn/bw/bw/transpose_grad/transpose	Transpose{gradients/bidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3Hgradients/bidi_/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutation*
T0*
Tperm0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Igradients/bidi_/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequenceReverseSequence@gradients/bidi_/bidirectional_rnn/bw/bw/transpose_grad/transposeseqlens*
seq_dim*
T0*

Tlen0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	batch_dim 
¼
gradients/AddN_10AddNgradients/concat_3_grad/Slice_2@gradients/bidi_/bidirectional_rnn/fw/fw/transpose_grad/transposeIgradients/bidi_/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequence*
T0*
N*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_class(
&$loc:@gradients/concat_3_grad/Slice_2
\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
o
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0*
_output_shapes
: 
j
gradients/concat_grad/ShapeShapeembedded_inputs*
T0*
out_type0*
_output_shapes
:

gradients/concat_grad/ShapeNShapeNembedded_inputs
split_cnts*
T0*
out_type0*
N* 
_output_shapes
::
¶
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1*
N* 
_output_shapes
::
¾
gradients/concat_grad/SliceSlicegradients/AddN_10"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
Index0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ã
gradients/concat_grad/Slice_1Slicegradients/AddN_10$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


$gradients/embedded_inputs_grad/ShapeConst*%
valueB	"7              *
dtype0	*
_output_shapes
:*
_class
loc:@embeddings
§
&gradients/embedded_inputs_grad/ToInt32Cast$gradients/embedded_inputs_grad/Shape*

DstT0*
_output_shapes
:*

SrcT0	*
_class
loc:@embeddings
d
#gradients/embedded_inputs_grad/SizeSizeinputs*
T0	*
out_type0*
_output_shapes
: 
o
-gradients/embedded_inputs_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
¼
)gradients/embedded_inputs_grad/ExpandDims
ExpandDims#gradients/embedded_inputs_grad/Size-gradients/embedded_inputs_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
|
2gradients/embedded_inputs_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
~
4gradients/embedded_inputs_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
~
4gradients/embedded_inputs_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

,gradients/embedded_inputs_grad/strided_sliceStridedSlice&gradients/embedded_inputs_grad/ToInt322gradients/embedded_inputs_grad/strided_slice/stack4gradients/embedded_inputs_grad/strided_slice/stack_14gradients/embedded_inputs_grad/strided_slice/stack_2*
Index0*
end_mask*
T0*
shrink_axis_mask *

begin_mask *
new_axis_mask *
_output_shapes
:*
ellipsis_mask 
l
*gradients/embedded_inputs_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ð
%gradients/embedded_inputs_grad/concatConcatV2)gradients/embedded_inputs_grad/ExpandDims,gradients/embedded_inputs_grad/strided_slice*gradients/embedded_inputs_grad/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
¶
&gradients/embedded_inputs_grad/ReshapeReshapegradients/concat_grad/Slice%gradients/embedded_inputs_grad/concat*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
(gradients/embedded_inputs_grad/Reshape_1Reshapeinputs)gradients/embedded_inputs_grad/ExpandDims*
T0	*
Tshape0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
global_norm/L2LossL2Loss&gradients/embedded_inputs_grad/Reshape*
T0*
_output_shapes
: *9
_class/
-+loc:@gradients/embedded_inputs_grad/Reshape
ø
global_norm/L2Loss_1L2LossQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*
_output_shapes
: *d
_classZ
XVloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3
ú
global_norm/L2Loss_2L2LossRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*
_output_shapes
: *e
_class[
YWloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3
ø
global_norm/L2Loss_3L2LossQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*
_output_shapes
: *d
_classZ
XVloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3
ú
global_norm/L2Loss_4L2LossRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*
_output_shapes
: *e
_class[
YWloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3
Ø
global_norm/L2Loss_5L2LossAgradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropFilter*
T0*
_output_shapes
: *T
_classJ
HFloc:@gradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropFilter
Æ
global_norm/L2Loss_6L2Loss8gradients/conv_maxpool_3_conv_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/conv_maxpool_3_conv_1/BiasAdd_grad/BiasAddGrad
Ø
global_norm/L2Loss_7L2LossAgradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropFilter*
T0*
_output_shapes
: *T
_classJ
HFloc:@gradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropFilter
Æ
global_norm/L2Loss_8L2Loss8gradients/conv_maxpool_5_conv_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/conv_maxpool_5_conv_1/BiasAdd_grad/BiasAddGrad
Ø
global_norm/L2Loss_9L2LossAgradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropFilter*
T0*
_output_shapes
: *T
_classJ
HFloc:@gradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropFilter
Ç
global_norm/L2Loss_10L2Loss8gradients/conv_maxpool_7_conv_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: *K
_classA
?=loc:@gradients/conv_maxpool_7_conv_1/BiasAdd_grad/BiasAddGrad

global_norm/L2Loss_11L2Lossgradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
: *1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1

global_norm/L2Loss_12L2Lossgradients/add_grad/Reshape_1*
T0*
_output_shapes
: */
_class%
#!loc:@gradients/add_grad/Reshape_1
ò
global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7global_norm/L2Loss_8global_norm/L2Loss_9global_norm/L2Loss_10global_norm/L2Loss_11global_norm/L2Loss_12*

axis *
T0*
N*
_output_shapes
:
[
global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
X
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
clip_by_global_norm/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
T0*
_output_shapes
: 
Ï
clip_by_global_norm/mul_1Mul&gradients/embedded_inputs_grad/Reshapeclip_by_global_norm/mul*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*9
_class/
-+loc:@gradients/embedded_inputs_grad/Reshape
¿
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*9
_class/
-+loc:@gradients/embedded_inputs_grad/Reshape

clip_by_global_norm/mul_2MulQgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0* 
_output_shapes
:
Ò *d
_classZ
XVloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3
â
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0* 
_output_shapes
:
Ò *d
_classZ
XVloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3

clip_by_global_norm/mul_3MulRgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*
_output_shapes	
: *e
_class[
YWloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3
Þ
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3*
T0*
_output_shapes	
: *e
_class[
YWloc:@gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3

clip_by_global_norm/mul_4MulQgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0* 
_output_shapes
:
Ò *d
_classZ
XVloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3
â
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0* 
_output_shapes
:
Ò *d
_classZ
XVloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3

clip_by_global_norm/mul_5MulRgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3clip_by_global_norm/mul*
T0*
_output_shapes	
: *e
_class[
YWloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3
Þ
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
T0*
_output_shapes	
: *e
_class[
YWloc:@gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3

clip_by_global_norm/mul_6MulAgradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropFilterclip_by_global_norm/mul*
T0*'
_output_shapes
:d*T
_classJ
HFloc:@gradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropFilter
Ù
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
T0*'
_output_shapes
:d*T
_classJ
HFloc:@gradients/conv_maxpool_3_conv_1/conv_op_grad/Conv2DBackpropFilter
å
clip_by_global_norm/mul_7Mul8gradients/conv_maxpool_3_conv_1/BiasAdd_grad/BiasAddGradclip_by_global_norm/mul*
T0*
_output_shapes
:d*K
_classA
?=loc:@gradients/conv_maxpool_3_conv_1/BiasAdd_grad/BiasAddGrad
Ã
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*
_output_shapes
:d*K
_classA
?=loc:@gradients/conv_maxpool_3_conv_1/BiasAdd_grad/BiasAddGrad

clip_by_global_norm/mul_8MulAgradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropFilterclip_by_global_norm/mul*
T0*'
_output_shapes
:d*T
_classJ
HFloc:@gradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropFilter
Ù
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
T0*'
_output_shapes
:d*T
_classJ
HFloc:@gradients/conv_maxpool_5_conv_1/conv_op_grad/Conv2DBackpropFilter
å
clip_by_global_norm/mul_9Mul8gradients/conv_maxpool_5_conv_1/BiasAdd_grad/BiasAddGradclip_by_global_norm/mul*
T0*
_output_shapes
:d*K
_classA
?=loc:@gradients/conv_maxpool_5_conv_1/BiasAdd_grad/BiasAddGrad
Ã
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9*
T0*
_output_shapes
:d*K
_classA
?=loc:@gradients/conv_maxpool_5_conv_1/BiasAdd_grad/BiasAddGrad

clip_by_global_norm/mul_10MulAgradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropFilterclip_by_global_norm/mul*
T0*'
_output_shapes
:d*T
_classJ
HFloc:@gradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropFilter
Ú
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10*
T0*'
_output_shapes
:d*T
_classJ
HFloc:@gradients/conv_maxpool_7_conv_1/conv_op_grad/Conv2DBackpropFilter
æ
clip_by_global_norm/mul_11Mul8gradients/conv_maxpool_7_conv_1/BiasAdd_grad/BiasAddGradclip_by_global_norm/mul*
T0*
_output_shapes
:d*K
_classA
?=loc:@gradients/conv_maxpool_7_conv_1/BiasAdd_grad/BiasAddGrad
Å
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11*
T0*
_output_shapes
:d*K
_classA
?=loc:@gradients/conv_maxpool_7_conv_1/BiasAdd_grad/BiasAddGrad
·
clip_by_global_norm/mul_12Mulgradients/MatMul_grad/MatMul_1clip_by_global_norm/mul*
T0*
_output_shapes
:	ÆR*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
°
+clip_by_global_norm/clip_by_global_norm/_11Identityclip_by_global_norm/mul_12*
T0*
_output_shapes
:	ÆR*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
®
clip_by_global_norm/mul_13Mulgradients/add_grad/Reshape_1clip_by_global_norm/mul*
T0*
_output_shapes
:R*/
_class%
#!loc:@gradients/add_grad/Reshape_1
©
+clip_by_global_norm/clip_by_global_norm/_12Identityclip_by_global_norm/mul_13*
T0*
_output_shapes
:R*/
_class%
#!loc:@gradients/add_grad/Reshape_1

beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
­
beta1_power
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes
: *
dtype0*
	container *
shape: 
Ì
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias

beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias

beta2_power/initial_valueConst*
valueB
 *w¾?*
dtype0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
­
beta2_power
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes
: *
dtype0*
	container *
shape: 
Ì
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias

beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
¡
1embeddings/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"7      *
dtype0*
_output_shapes
:*
_class
loc:@embeddings

'embeddings/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@embeddings
à
!embeddings/Adam/Initializer/zerosFill1embeddings/Adam/Initializer/zeros/shape_as_tensor'embeddings/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:	7*

index_type0*
_class
loc:@embeddings
¤
embeddings/Adam
VariableV2*
shared_name *
_class
loc:@embeddings*
_output_shapes
:	7*
dtype0*
	container *
shape:	7
Æ
embeddings/Adam/AssignAssignembeddings/Adam!embeddings/Adam/Initializer/zeros*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
z
embeddings/Adam/readIdentityembeddings/Adam*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
£
3embeddings/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"7      *
dtype0*
_output_shapes
:*
_class
loc:@embeddings

)embeddings/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@embeddings
æ
#embeddings/Adam_1/Initializer/zerosFill3embeddings/Adam_1/Initializer/zeros/shape_as_tensor)embeddings/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
:	7*

index_type0*
_class
loc:@embeddings
¦
embeddings/Adam_1
VariableV2*
shared_name *
_class
loc:@embeddings*
_output_shapes
:	7*
dtype0*
	container *
shape:	7
Ì
embeddings/Adam_1/AssignAssignembeddings/Adam_1#embeddings/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
~
embeddings/Adam_1/readIdentityembeddings/Adam_1*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
ã
Rbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"R     *
dtype0*
_output_shapes
:*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Í
Hbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
å
Bbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zerosFillRbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorHbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0* 
_output_shapes
:
Ò *

index_type0*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
è
0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam
VariableV2*
shared_name *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
Ò *
dtype0*
	container *
shape:
Ò 
Ë
7bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/AssignAssign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamBbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Þ
5bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/readIdentity0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
å
Tbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"R     *
dtype0*
_output_shapes
:*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ï
Jbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ë
Dbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zerosFillTbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorJbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_output_shapes
:
Ò *

index_type0*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ê
2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
Ò *
dtype0*
	container *
shape:
Ò 
Ñ
9bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/AssignAssign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1Dbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
â
7bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/readIdentity2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Í
@bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
Ú
.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
: *
dtype0*
	container *
shape: 
¾
5bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/AssignAssign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam@bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
Ó
3bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/readIdentity.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam*
T0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
Ï
Bbidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
Ü
0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
: *
dtype0*
	container *
shape: 
Ä
7bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/AssignAssign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1Bbidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
×
5bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/readIdentity0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1*
T0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ã
Rbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"R     *
dtype0*
_output_shapes
:*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Í
Hbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
Bbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zerosFillRbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorHbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0* 
_output_shapes
:
Ò *

index_type0*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
è
0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam
VariableV2*
shared_name *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
Ò *
dtype0*
	container *
shape:
Ò 
Ë
7bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/AssignAssign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamBbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Þ
5bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/readIdentity0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
Tbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"R     *
dtype0*
_output_shapes
:*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Ï
Jbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ë
Dbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zerosFillTbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorJbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_output_shapes
:
Ò *

index_type0*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ê
2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
Ò *
dtype0*
	container *
shape:
Ò 
Ñ
9bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/AssignAssign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1Dbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
â
7bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/readIdentity2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1*
T0* 
_output_shapes
:
Ò *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
Í
@bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ú
.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
: *
dtype0*
	container *
shape: 
¾
5bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/AssignAssign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam@bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ó
3bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/readIdentity.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam*
T0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ï
Bbidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ü
0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1
VariableV2*
shared_name *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
: *
dtype0*
	container *
shape: 
Ä
7bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/AssignAssign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1Bbidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
×
5bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/readIdentity0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1*
T0*
_output_shapes	
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Õ
Gconv_maxpool_3_conv_1/W_filter_0/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
·
=conv_maxpool_3_conv_1/W_filter_0/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
À
7conv_maxpool_3_conv_1/W_filter_0/Adam/Initializer/zerosFillGconv_maxpool_3_conv_1/W_filter_0/Adam/Initializer/zeros/shape_as_tensor=conv_maxpool_3_conv_1/W_filter_0/Adam/Initializer/zeros/Const*
T0*'
_output_shapes
:d*

index_type0*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
à
%conv_maxpool_3_conv_1/W_filter_0/Adam
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¦
,conv_maxpool_3_conv_1/W_filter_0/Adam/AssignAssign%conv_maxpool_3_conv_1/W_filter_0/Adam7conv_maxpool_3_conv_1/W_filter_0/Adam/Initializer/zeros*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ä
*conv_maxpool_3_conv_1/W_filter_0/Adam/readIdentity%conv_maxpool_3_conv_1/W_filter_0/Adam*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
×
Iconv_maxpool_3_conv_1/W_filter_0/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
¹
?conv_maxpool_3_conv_1/W_filter_0/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Æ
9conv_maxpool_3_conv_1/W_filter_0/Adam_1/Initializer/zerosFillIconv_maxpool_3_conv_1/W_filter_0/Adam_1/Initializer/zeros/shape_as_tensor?conv_maxpool_3_conv_1/W_filter_0/Adam_1/Initializer/zeros/Const*
T0*'
_output_shapes
:d*

index_type0*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
â
'conv_maxpool_3_conv_1/W_filter_0/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¬
.conv_maxpool_3_conv_1/W_filter_0/Adam_1/AssignAssign'conv_maxpool_3_conv_1/W_filter_0/Adam_19conv_maxpool_3_conv_1/W_filter_0/Adam_1/Initializer/zeros*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
È
,conv_maxpool_3_conv_1/W_filter_0/Adam_1/readIdentity'conv_maxpool_3_conv_1/W_filter_0/Adam_1*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
¹
7conv_maxpool_3_conv_1/B_filter_0/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Æ
%conv_maxpool_3_conv_1/B_filter_0/Adam
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0*
_output_shapes
:d*
dtype0*
	container *
shape:d

,conv_maxpool_3_conv_1/B_filter_0/Adam/AssignAssign%conv_maxpool_3_conv_1/B_filter_0/Adam7conv_maxpool_3_conv_1/B_filter_0/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
·
*conv_maxpool_3_conv_1/B_filter_0/Adam/readIdentity%conv_maxpool_3_conv_1/B_filter_0/Adam*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
»
9conv_maxpool_3_conv_1/B_filter_0/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
È
'conv_maxpool_3_conv_1/B_filter_0/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0*
_output_shapes
:d*
dtype0*
	container *
shape:d

.conv_maxpool_3_conv_1/B_filter_0/Adam_1/AssignAssign'conv_maxpool_3_conv_1/B_filter_0/Adam_19conv_maxpool_3_conv_1/B_filter_0/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
»
,conv_maxpool_3_conv_1/B_filter_0/Adam_1/readIdentity'conv_maxpool_3_conv_1/B_filter_0/Adam_1*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Õ
Gconv_maxpool_5_conv_1/W_filter_1/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
·
=conv_maxpool_5_conv_1/W_filter_1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
À
7conv_maxpool_5_conv_1/W_filter_1/Adam/Initializer/zerosFillGconv_maxpool_5_conv_1/W_filter_1/Adam/Initializer/zeros/shape_as_tensor=conv_maxpool_5_conv_1/W_filter_1/Adam/Initializer/zeros/Const*
T0*'
_output_shapes
:d*

index_type0*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
à
%conv_maxpool_5_conv_1/W_filter_1/Adam
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¦
,conv_maxpool_5_conv_1/W_filter_1/Adam/AssignAssign%conv_maxpool_5_conv_1/W_filter_1/Adam7conv_maxpool_5_conv_1/W_filter_1/Adam/Initializer/zeros*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ä
*conv_maxpool_5_conv_1/W_filter_1/Adam/readIdentity%conv_maxpool_5_conv_1/W_filter_1/Adam*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
×
Iconv_maxpool_5_conv_1/W_filter_1/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
¹
?conv_maxpool_5_conv_1/W_filter_1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Æ
9conv_maxpool_5_conv_1/W_filter_1/Adam_1/Initializer/zerosFillIconv_maxpool_5_conv_1/W_filter_1/Adam_1/Initializer/zeros/shape_as_tensor?conv_maxpool_5_conv_1/W_filter_1/Adam_1/Initializer/zeros/Const*
T0*'
_output_shapes
:d*

index_type0*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
â
'conv_maxpool_5_conv_1/W_filter_1/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¬
.conv_maxpool_5_conv_1/W_filter_1/Adam_1/AssignAssign'conv_maxpool_5_conv_1/W_filter_1/Adam_19conv_maxpool_5_conv_1/W_filter_1/Adam_1/Initializer/zeros*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
È
,conv_maxpool_5_conv_1/W_filter_1/Adam_1/readIdentity'conv_maxpool_5_conv_1/W_filter_1/Adam_1*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
¹
7conv_maxpool_5_conv_1/B_filter_1/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Æ
%conv_maxpool_5_conv_1/B_filter_1/Adam
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1*
_output_shapes
:d*
dtype0*
	container *
shape:d

,conv_maxpool_5_conv_1/B_filter_1/Adam/AssignAssign%conv_maxpool_5_conv_1/B_filter_1/Adam7conv_maxpool_5_conv_1/B_filter_1/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
·
*conv_maxpool_5_conv_1/B_filter_1/Adam/readIdentity%conv_maxpool_5_conv_1/B_filter_1/Adam*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
»
9conv_maxpool_5_conv_1/B_filter_1/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
È
'conv_maxpool_5_conv_1/B_filter_1/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1*
_output_shapes
:d*
dtype0*
	container *
shape:d

.conv_maxpool_5_conv_1/B_filter_1/Adam_1/AssignAssign'conv_maxpool_5_conv_1/B_filter_1/Adam_19conv_maxpool_5_conv_1/B_filter_1/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
»
,conv_maxpool_5_conv_1/B_filter_1/Adam_1/readIdentity'conv_maxpool_5_conv_1/B_filter_1/Adam_1*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Õ
Gconv_maxpool_7_conv_1/W_filter_2/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
·
=conv_maxpool_7_conv_1/W_filter_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
À
7conv_maxpool_7_conv_1/W_filter_2/Adam/Initializer/zerosFillGconv_maxpool_7_conv_1/W_filter_2/Adam/Initializer/zeros/shape_as_tensor=conv_maxpool_7_conv_1/W_filter_2/Adam/Initializer/zeros/Const*
T0*'
_output_shapes
:d*

index_type0*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
à
%conv_maxpool_7_conv_1/W_filter_2/Adam
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¦
,conv_maxpool_7_conv_1/W_filter_2/Adam/AssignAssign%conv_maxpool_7_conv_1/W_filter_2/Adam7conv_maxpool_7_conv_1/W_filter_2/Adam/Initializer/zeros*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
Ä
*conv_maxpool_7_conv_1/W_filter_2/Adam/readIdentity%conv_maxpool_7_conv_1/W_filter_2/Adam*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
×
Iconv_maxpool_7_conv_1/W_filter_2/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"        d   *
dtype0*
_output_shapes
:*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¹
?conv_maxpool_7_conv_1/W_filter_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
Æ
9conv_maxpool_7_conv_1/W_filter_2/Adam_1/Initializer/zerosFillIconv_maxpool_7_conv_1/W_filter_2/Adam_1/Initializer/zeros/shape_as_tensor?conv_maxpool_7_conv_1/W_filter_2/Adam_1/Initializer/zeros/Const*
T0*'
_output_shapes
:d*

index_type0*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
â
'conv_maxpool_7_conv_1/W_filter_2/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2*'
_output_shapes
:d*
dtype0*
	container *
shape:d
¬
.conv_maxpool_7_conv_1/W_filter_2/Adam_1/AssignAssign'conv_maxpool_7_conv_1/W_filter_2/Adam_19conv_maxpool_7_conv_1/W_filter_2/Adam_1/Initializer/zeros*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
È
,conv_maxpool_7_conv_1/W_filter_2/Adam_1/readIdentity'conv_maxpool_7_conv_1/W_filter_2/Adam_1*
T0*'
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¹
7conv_maxpool_7_conv_1/B_filter_2/Adam/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Æ
%conv_maxpool_7_conv_1/B_filter_2/Adam
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2*
_output_shapes
:d*
dtype0*
	container *
shape:d

,conv_maxpool_7_conv_1/B_filter_2/Adam/AssignAssign%conv_maxpool_7_conv_1/B_filter_2/Adam7conv_maxpool_7_conv_1/B_filter_2/Adam/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
·
*conv_maxpool_7_conv_1/B_filter_2/Adam/readIdentity%conv_maxpool_7_conv_1/B_filter_2/Adam*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
»
9conv_maxpool_7_conv_1/B_filter_2/Adam_1/Initializer/zerosConst*
valueBd*    *
dtype0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
È
'conv_maxpool_7_conv_1/B_filter_2/Adam_1
VariableV2*
shared_name *3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2*
_output_shapes
:d*
dtype0*
	container *
shape:d

.conv_maxpool_7_conv_1/B_filter_2/Adam_1/AssignAssign'conv_maxpool_7_conv_1/B_filter_2/Adam_19conv_maxpool_7_conv_1/B_filter_2/Adam_1/Initializer/zeros*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
»
,conv_maxpool_7_conv_1/B_filter_2/Adam_1/readIdentity'conv_maxpool_7_conv_1/B_filter_2/Adam_1*
T0*
_output_shapes
:d*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
£
2out_weights/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"F  R   *
dtype0*
_output_shapes
:*
_class
loc:@out_weights

(out_weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@out_weights
ä
"out_weights/Adam/Initializer/zerosFill2out_weights/Adam/Initializer/zeros/shape_as_tensor(out_weights/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:	ÆR*

index_type0*
_class
loc:@out_weights
¦
out_weights/Adam
VariableV2*
shared_name *
_class
loc:@out_weights*
_output_shapes
:	ÆR*
dtype0*
	container *
shape:	ÆR
Ê
out_weights/Adam/AssignAssignout_weights/Adam"out_weights/Adam/Initializer/zeros*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
}
out_weights/Adam/readIdentityout_weights/Adam*
T0*
_output_shapes
:	ÆR*
_class
loc:@out_weights
¥
4out_weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"F  R   *
dtype0*
_output_shapes
:*
_class
loc:@out_weights

*out_weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@out_weights
ê
$out_weights/Adam_1/Initializer/zerosFill4out_weights/Adam_1/Initializer/zeros/shape_as_tensor*out_weights/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
:	ÆR*

index_type0*
_class
loc:@out_weights
¨
out_weights/Adam_1
VariableV2*
shared_name *
_class
loc:@out_weights*
_output_shapes
:	ÆR*
dtype0*
	container *
shape:	ÆR
Ð
out_weights/Adam_1/AssignAssignout_weights/Adam_1$out_weights/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

out_weights/Adam_1/readIdentityout_weights/Adam_1*
T0*
_output_shapes
:	ÆR*
_class
loc:@out_weights

out_bias/Adam/Initializer/zerosConst*
valueBR*    *
dtype0*
_output_shapes
:R*
_class
loc:@out_bias

out_bias/Adam
VariableV2*
shared_name *
_class
loc:@out_bias*
_output_shapes
:R*
dtype0*
	container *
shape:R
¹
out_bias/Adam/AssignAssignout_bias/Adamout_bias/Adam/Initializer/zeros*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
o
out_bias/Adam/readIdentityout_bias/Adam*
T0*
_output_shapes
:R*
_class
loc:@out_bias

!out_bias/Adam_1/Initializer/zerosConst*
valueBR*    *
dtype0*
_output_shapes
:R*
_class
loc:@out_bias

out_bias/Adam_1
VariableV2*
shared_name *
_class
loc:@out_bias*
_output_shapes
:R*
dtype0*
	container *
shape:R
¿
out_bias/Adam_1/AssignAssignout_bias/Adam_1!out_bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
s
out_bias/Adam_1/readIdentityout_bias/Adam_1*
T0*
_output_shapes
:R*
_class
loc:@out_bias
`
optim_apply_gradients/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
`
optim_apply_gradients/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
b
optim_apply_gradients/epsilonConst*
valueB
 *wÌ+2*
dtype0*
_output_shapes
: 
Í
.optim_apply_gradients/update_embeddings/UniqueUnique(gradients/embedded_inputs_grad/Reshape_1*
T0	*
out_idx0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
_class
loc:@embeddings
º
-optim_apply_gradients/update_embeddings/ShapeShape.optim_apply_gradients/update_embeddings/Unique*
T0	*
out_type0*
_output_shapes
:*
_class
loc:@embeddings
¤
;optim_apply_gradients/update_embeddings/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:*
_class
loc:@embeddings
¦
=optim_apply_gradients/update_embeddings/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:*
_class
loc:@embeddings
¦
=optim_apply_gradients/update_embeddings/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:*
_class
loc:@embeddings
à
5optim_apply_gradients/update_embeddings/strided_sliceStridedSlice-optim_apply_gradients/update_embeddings/Shape;optim_apply_gradients/update_embeddings/strided_slice/stack=optim_apply_gradients/update_embeddings/strided_slice/stack_1=optim_apply_gradients/update_embeddings/strided_slice/stack_2*
Index0*
end_mask *
_class
loc:@embeddings*
T0*
shrink_axis_mask*

begin_mask *
new_axis_mask *
_output_shapes
: *
ellipsis_mask 
Û
:optim_apply_gradients/update_embeddings/UnsortedSegmentSumUnsortedSegmentSum*clip_by_global_norm/clip_by_global_norm/_00optim_apply_gradients/update_embeddings/Unique:15optim_apply_gradients/update_embeddings/strided_slice*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tindices0*
Tnumsegments0*
_class
loc:@embeddings

-optim_apply_gradients/update_embeddings/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: *
_class
loc:@embeddings
³
+optim_apply_gradients/update_embeddings/subSub-optim_apply_gradients/update_embeddings/sub/xbeta2_power/read*
T0*
_output_shapes
: *
_class
loc:@embeddings
¡
,optim_apply_gradients/update_embeddings/SqrtSqrt+optim_apply_gradients/update_embeddings/sub*
T0*
_output_shapes
: *
_class
loc:@embeddings
±
+optim_apply_gradients/update_embeddings/mulMullearning_rate,optim_apply_gradients/update_embeddings/Sqrt*
T0*
_output_shapes
:*
_class
loc:@embeddings

/optim_apply_gradients/update_embeddings/sub_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: *
_class
loc:@embeddings
·
-optim_apply_gradients/update_embeddings/sub_1Sub/optim_apply_gradients/update_embeddings/sub_1/xbeta1_power/read*
T0*
_output_shapes
: *
_class
loc:@embeddings
Ø
/optim_apply_gradients/update_embeddings/truedivRealDiv+optim_apply_gradients/update_embeddings/mul-optim_apply_gradients/update_embeddings/sub_1*
T0*
_output_shapes
:*
_class
loc:@embeddings

/optim_apply_gradients/update_embeddings/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: *
_class
loc:@embeddings
Â
-optim_apply_gradients/update_embeddings/sub_2Sub/optim_apply_gradients/update_embeddings/sub_2/xoptim_apply_gradients/beta1*
T0*
_output_shapes
: *
_class
loc:@embeddings
ñ
-optim_apply_gradients/update_embeddings/mul_1Mul:optim_apply_gradients/update_embeddings/UnsortedSegmentSum-optim_apply_gradients/update_embeddings/sub_2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
_class
loc:@embeddings
°
-optim_apply_gradients/update_embeddings/mul_2Mulembeddings/Adam/readoptim_apply_gradients/beta1*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
ê
.optim_apply_gradients/update_embeddings/AssignAssignembeddings/Adam-optim_apply_gradients/update_embeddings/mul_2*
T0*
_output_shapes
:	7*
use_locking( *
validate_shape(*
_class
loc:@embeddings
Í
2optim_apply_gradients/update_embeddings/ScatterAdd
ScatterAddembeddings/Adam.optim_apply_gradients/update_embeddings/Unique-optim_apply_gradients/update_embeddings/mul_1/^optim_apply_gradients/update_embeddings/Assign*
T0*
_output_shapes
:	7*
use_locking( *
Tindices0	*
_class
loc:@embeddings
þ
-optim_apply_gradients/update_embeddings/mul_3Mul:optim_apply_gradients/update_embeddings/UnsortedSegmentSum:optim_apply_gradients/update_embeddings/UnsortedSegmentSum*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
_class
loc:@embeddings

/optim_apply_gradients/update_embeddings/sub_3/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: *
_class
loc:@embeddings
Â
-optim_apply_gradients/update_embeddings/sub_3Sub/optim_apply_gradients/update_embeddings/sub_3/xoptim_apply_gradients/beta2*
T0*
_output_shapes
: *
_class
loc:@embeddings
ä
-optim_apply_gradients/update_embeddings/mul_4Mul-optim_apply_gradients/update_embeddings/mul_3-optim_apply_gradients/update_embeddings/sub_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
_class
loc:@embeddings
²
-optim_apply_gradients/update_embeddings/mul_5Mulembeddings/Adam_1/readoptim_apply_gradients/beta2*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
î
0optim_apply_gradients/update_embeddings/Assign_1Assignembeddings/Adam_1-optim_apply_gradients/update_embeddings/mul_5*
T0*
_output_shapes
:	7*
use_locking( *
validate_shape(*
_class
loc:@embeddings
Ó
4optim_apply_gradients/update_embeddings/ScatterAdd_1
ScatterAddembeddings/Adam_1.optim_apply_gradients/update_embeddings/Unique-optim_apply_gradients/update_embeddings/mul_41^optim_apply_gradients/update_embeddings/Assign_1*
T0*
_output_shapes
:	7*
use_locking( *
Tindices0	*
_class
loc:@embeddings
µ
.optim_apply_gradients/update_embeddings/Sqrt_1Sqrt4optim_apply_gradients/update_embeddings/ScatterAdd_1*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
Û
-optim_apply_gradients/update_embeddings/mul_6Mul/optim_apply_gradients/update_embeddings/truediv2optim_apply_gradients/update_embeddings/ScatterAdd*
T0*
_output_shapes
:*
_class
loc:@embeddings
Ê
+optim_apply_gradients/update_embeddings/addAdd.optim_apply_gradients/update_embeddings/Sqrt_1optim_apply_gradients/epsilon*
T0*
_output_shapes
:	7*
_class
loc:@embeddings
Ú
1optim_apply_gradients/update_embeddings/truediv_1RealDiv-optim_apply_gradients/update_embeddings/mul_6+optim_apply_gradients/update_embeddings/add*
T0*
_output_shapes
:*
_class
loc:@embeddings
Ù
1optim_apply_gradients/update_embeddings/AssignSub	AssignSub
embeddings1optim_apply_gradients/update_embeddings/truediv_1*
T0*
_output_shapes
:	7*
use_locking( *
_class
loc:@embeddings
ù
2optim_apply_gradients/update_embeddings/group_depsNoOp2^optim_apply_gradients/update_embeddings/AssignSub3^optim_apply_gradients/update_embeddings/ScatterAdd5^optim_apply_gradients/update_embeddings/ScatterAdd_1*
_class
loc:@embeddings
¼
Roptim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdam	ApplyAdam+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_1*
T0* 
_output_shapes
:
Ò *
use_locking( *
use_nesterov( *>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
­
Poptim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdam	ApplyAdam)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_2*
T0*
_output_shapes	
: *
use_locking( *
use_nesterov( *<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
¼
Roptim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdam	ApplyAdam+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
T0* 
_output_shapes
:
Ò *
use_locking( *
use_nesterov( *>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
­
Poptim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdam	ApplyAdam)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_4*
T0*
_output_shapes	
: *
use_locking( *
use_nesterov( *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias

Goptim_apply_gradients/update_conv_maxpool_3_conv_1/W_filter_0/ApplyAdam	ApplyAdam conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_5*
T0*'
_output_shapes
:d*
use_locking( *
use_nesterov( *3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ÿ
Goptim_apply_gradients/update_conv_maxpool_3_conv_1/B_filter_0/ApplyAdam	ApplyAdam conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_6*
T0*
_output_shapes
:d*
use_locking( *
use_nesterov( *3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0

Goptim_apply_gradients/update_conv_maxpool_5_conv_1/W_filter_1/ApplyAdam	ApplyAdam conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
T0*'
_output_shapes
:d*
use_locking( *
use_nesterov( *3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ÿ
Goptim_apply_gradients/update_conv_maxpool_5_conv_1/B_filter_1/ApplyAdam	ApplyAdam conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_8*
T0*
_output_shapes
:d*
use_locking( *
use_nesterov( *3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1

Goptim_apply_gradients/update_conv_maxpool_7_conv_1/W_filter_2/ApplyAdam	ApplyAdam conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon*clip_by_global_norm/clip_by_global_norm/_9*
T0*'
_output_shapes
:d*
use_locking( *
use_nesterov( *3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2

Goptim_apply_gradients/update_conv_maxpool_7_conv_1/B_filter_2/ApplyAdam	ApplyAdam conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon+clip_by_global_norm/clip_by_global_norm/_10*
T0*
_output_shapes
:d*
use_locking( *
use_nesterov( *3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2

2optim_apply_gradients/update_out_weights/ApplyAdam	ApplyAdamout_weightsout_weights/Adamout_weights/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon+clip_by_global_norm/clip_by_global_norm/_11*
T0*
_output_shapes
:	ÆR*
use_locking( *
use_nesterov( *
_class
loc:@out_weights

/optim_apply_gradients/update_out_bias/ApplyAdam	ApplyAdamout_biasout_bias/Adamout_bias/Adam_1beta1_power/readbeta2_power/readlearning_rateoptim_apply_gradients/beta1optim_apply_gradients/beta2optim_apply_gradients/epsilon+clip_by_global_norm/clip_by_global_norm/_12*
T0*
_output_shapes
:R*
use_locking( *
use_nesterov( *
_class
loc:@out_bias
Ö
optim_apply_gradients/mulMulbeta1_power/readoptim_apply_gradients/beta1Q^optim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamS^optim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamQ^optim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamS^optim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_3_conv_1/B_filter_0/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_3_conv_1/W_filter_0/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_5_conv_1/B_filter_1/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_5_conv_1/W_filter_1/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_7_conv_1/B_filter_2/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_7_conv_1/W_filter_2/ApplyAdam3^optim_apply_gradients/update_embeddings/group_deps0^optim_apply_gradients/update_out_bias/ApplyAdam3^optim_apply_gradients/update_out_weights/ApplyAdam*
T0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ö
optim_apply_gradients/AssignAssignbeta1_poweroptim_apply_gradients/mul*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ø
optim_apply_gradients/mul_1Mulbeta2_power/readoptim_apply_gradients/beta2Q^optim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamS^optim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamQ^optim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamS^optim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_3_conv_1/B_filter_0/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_3_conv_1/W_filter_0/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_5_conv_1/B_filter_1/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_5_conv_1/W_filter_1/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_7_conv_1/B_filter_2/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_7_conv_1/W_filter_2/ApplyAdam3^optim_apply_gradients/update_embeddings/group_deps0^optim_apply_gradients/update_out_bias/ApplyAdam3^optim_apply_gradients/update_out_weights/ApplyAdam*
T0*
_output_shapes
: *<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ú
optim_apply_gradients/Assign_1Assignbeta2_poweroptim_apply_gradients/mul_1*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Õ
optim_apply_gradients/NoOpNoOp^optim_apply_gradients/Assign^optim_apply_gradients/Assign_1Q^optim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdamS^optim_apply_gradients/update_bidi_/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamQ^optim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamS^optim_apply_gradients/update_bidi_/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_3_conv_1/B_filter_0/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_3_conv_1/W_filter_0/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_5_conv_1/B_filter_1/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_5_conv_1/W_filter_1/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_7_conv_1/B_filter_2/ApplyAdamH^optim_apply_gradients/update_conv_maxpool_7_conv_1/W_filter_2/ApplyAdam0^optim_apply_gradients/update_out_bias/ApplyAdam3^optim_apply_gradients/update_out_weights/ApplyAdam
Y
optim_apply_gradients/NoOp_1NoOp3^optim_apply_gradients/update_embeddings/group_deps
Y
optim_apply_gradientsNoOp^optim_apply_gradients/NoOp^optim_apply_gradients/NoOp_1
S
Merge/MergeSummaryMergeSummaryaccuracycost*
N*
_output_shapes
: 
¤
	init/NoOpNoOp^beta1_power/Assign^beta2_power/Assign6^bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign8^bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign1^bidi_/bidirectional_rnn/bw/lstm_cell/bias/Assign8^bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Assign:^bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Assign3^bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Assign6^bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign8^bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign1^bidi_/bidirectional_rnn/fw/lstm_cell/bias/Assign8^bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Assign:^bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Assign3^bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Assign-^conv_maxpool_3_conv_1/B_filter_0/Adam/Assign/^conv_maxpool_3_conv_1/B_filter_0/Adam_1/Assign(^conv_maxpool_3_conv_1/B_filter_0/Assign-^conv_maxpool_3_conv_1/W_filter_0/Adam/Assign/^conv_maxpool_3_conv_1/W_filter_0/Adam_1/Assign(^conv_maxpool_3_conv_1/W_filter_0/Assign-^conv_maxpool_5_conv_1/B_filter_1/Adam/Assign/^conv_maxpool_5_conv_1/B_filter_1/Adam_1/Assign(^conv_maxpool_5_conv_1/B_filter_1/Assign-^conv_maxpool_5_conv_1/W_filter_1/Adam/Assign/^conv_maxpool_5_conv_1/W_filter_1/Adam_1/Assign(^conv_maxpool_5_conv_1/W_filter_1/Assign-^conv_maxpool_7_conv_1/B_filter_2/Adam/Assign/^conv_maxpool_7_conv_1/B_filter_2/Adam_1/Assign(^conv_maxpool_7_conv_1/B_filter_2/Assign-^conv_maxpool_7_conv_1/W_filter_2/Adam/Assign/^conv_maxpool_7_conv_1/W_filter_2/Adam_1/Assign(^conv_maxpool_7_conv_1/W_filter_2/Assign^out_bias/Adam/Assign^out_bias/Adam_1/Assign^out_bias/Assign^out_weights/Adam/Assign^out_weights/Adam_1/Assign^out_weights/Assign
[
init/NoOp_1NoOp^embeddings/Adam/Assign^embeddings/Adam_1/Assign^embeddings/Assign
&
initNoOp
^init/NoOp^init/NoOp_1
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_cfdd9906aea84c7388cd67a8ba14258c/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Å
save/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
¯
save/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
ü
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
^
save/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards*
_output_shapes
: 

save/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
k
save/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
®
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2

save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1*
T0*
_output_shapes
: *)
_class
loc:@save/ShardedFilename_1
Ñ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1*
T0*
_output_shapes
: 
È
save/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ì
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
º
save/AssignAssignbeta1_powersave/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
¾
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
á
save/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
æ
save/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
è
save/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ï
save/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ñ
save/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
á
save/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
æ
save/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ñ
save/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ó
save/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ð
save/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Õ
save/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
×
save/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
â
save/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ä
save/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ð
save/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Õ
save/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
×
save/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
â
save/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ä
save/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ð
save/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Õ
save/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
×
save/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
â
save/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ä
save/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
 
save/Assign_32Assignout_biassave/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¥
save/Assign_33Assignout_bias/Adamsave/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
§
save/Assign_34Assignout_bias/Adam_1save/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save/Assign_35Assignout_weightssave/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
°
save/Assign_36Assignout_weights/Adamsave/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
²
save/Assign_37Assignout_weights/Adam_1save/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9

save/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
 
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¨
save/Assign_38Assign
embeddingssave/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
¯
save/Assign_39Assignembeddings/Adamsave/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
±
save/Assign_40Assignembeddings/Adam_1save/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
O
save/restore_shard_1NoOp^save/Assign_38^save/Assign_39^save/Assign_40
2
save/restore_all/NoOpNoOp^save/restore_shard
6
save/restore_all/NoOp_1NoOp^save/restore_shard_1
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_10b3b71fed8641489644c06db3d5e5f2/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
Ç
save_1/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_1/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
`
save_1/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_1/ShardedFilename_1ShardedFilenamesave_1/StringJoinsave_1/ShardedFilename_1/shardsave_1/num_shards*
_output_shapes
: 

save_1/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_1/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_1/SaveV2_1SaveV2save_1/ShardedFilename_1save_1/SaveV2_1/tensor_names save_1/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_1/control_dependency_1Identitysave_1/ShardedFilename_1^save_1/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_1/ShardedFilename_1
Û
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilenamesave_1/ShardedFilename_1^save_1/control_dependency^save_1/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
 
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency^save_1/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_1/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_1/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_1/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_1/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_1/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_1/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_1/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_1/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_1/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_1/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_1/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_1/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_1/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_1/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_1/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_1/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_1/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_1/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_1/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_1/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_1/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_1/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_1/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_1/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_1/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_1/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_1/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_1/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_1/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_1/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_1/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_1/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_1/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_1/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_1/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_1/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_1/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_1/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_1/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_1/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_1/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_1/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_1/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_1/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_1/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_1/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_1/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_1/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_1/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_1/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_1/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_1/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_1/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_1/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_1/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_1/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_1/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_1/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_1/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_1/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_1/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_1/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_1/Assign_32Assignout_biassave_1/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_1/Assign_33Assignout_bias/Adamsave_1/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_1/Assign_34Assignout_bias/Adam_1save_1/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_1/Assign_35Assignout_weightssave_1/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_1/Assign_36Assignout_weights/Adamsave_1/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_1/Assign_37Assignout_weights/Adam_1save_1/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9

save_1/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_1/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_1/Assign_38Assign
embeddingssave_1/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_1/Assign_39Assignembeddings/Adamsave_1/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_1/Assign_40Assignembeddings/Adam_1save_1/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_1/restore_shard_1NoOp^save_1/Assign_38^save_1/Assign_39^save_1/Assign_40
6
save_1/restore_all/NoOpNoOp^save_1/restore_shard
:
save_1/restore_all/NoOp_1NoOp^save_1/restore_shard_1
P
save_1/restore_allNoOp^save_1/restore_all/NoOp^save_1/restore_all/NoOp_1
R
save_2/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_9b099e5471b3431f911041c0481089d2/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
Ç
save_2/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_2/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_2/ShardedFilename
`
save_2/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_2/ShardedFilename_1ShardedFilenamesave_2/StringJoinsave_2/ShardedFilename_1/shardsave_2/num_shards*
_output_shapes
: 

save_2/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_2/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_2/SaveV2_1SaveV2save_2/ShardedFilename_1save_2/SaveV2_1/tensor_names save_2/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_2/control_dependency_1Identitysave_2/ShardedFilename_1^save_2/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_2/ShardedFilename_1
Û
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilenamesave_2/ShardedFilename_1^save_2/control_dependency^save_2/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
 
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency^save_2/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_2/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_2/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_2/Assign_1Assignbeta2_powersave_2/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_2/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_2/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_2/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_2/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_2/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_2/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_2/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_2/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_2/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_2/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_2/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_2/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_2/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_2/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_2/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_2/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_2/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_2/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_2/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_2/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_2/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_2/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_2/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_2/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_2/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_2/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_2/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_2/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_2/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_2/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_2/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_2/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_2/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_2/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_2/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_2/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_2/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_2/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_2/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_2/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_2/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_2/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_2/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_2/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_2/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_2/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_2/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_2/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_2/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_2/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_2/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_2/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_2/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_2/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_2/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_2/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_2/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_2/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_2/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_2/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_2/Assign_32Assignout_biassave_2/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_2/Assign_33Assignout_bias/Adamsave_2/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_2/Assign_34Assignout_bias/Adam_1save_2/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_2/Assign_35Assignout_weightssave_2/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_2/Assign_36Assignout_weights/Adamsave_2/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_2/Assign_37Assignout_weights/Adam_1save_2/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9

save_2/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_2/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_2/RestoreV2_1	RestoreV2save_2/Constsave_2/RestoreV2_1/tensor_names#save_2/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_2/Assign_38Assign
embeddingssave_2/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_2/Assign_39Assignembeddings/Adamsave_2/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_2/Assign_40Assignembeddings/Adam_1save_2/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_2/restore_shard_1NoOp^save_2/Assign_38^save_2/Assign_39^save_2/Assign_40
6
save_2/restore_all/NoOpNoOp^save_2/restore_shard
:
save_2/restore_all/NoOp_1NoOp^save_2/restore_shard_1
P
save_2/restore_allNoOp^save_2/restore_all/NoOp^save_2/restore_all/NoOp_1
R
save_3/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_3d915b6bdf6c4391ac052be4b4bc7770/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
Ç
save_3/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_3/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_3/ShardedFilename
`
save_3/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_3/ShardedFilename_1ShardedFilenamesave_3/StringJoinsave_3/ShardedFilename_1/shardsave_3/num_shards*
_output_shapes
: 

save_3/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_3/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_3/SaveV2_1SaveV2save_3/ShardedFilename_1save_3/SaveV2_1/tensor_names save_3/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_3/control_dependency_1Identitysave_3/ShardedFilename_1^save_3/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_3/ShardedFilename_1
Û
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilenamesave_3/ShardedFilename_1^save_3/control_dependency^save_3/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
 
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency^save_3/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_3/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_3/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_3/Assign_1Assignbeta2_powersave_3/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_3/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_3/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_3/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_3/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_3/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_3/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_3/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_3/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_3/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_3/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_3/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_3/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_3/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_3/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_3/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_3/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_3/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_3/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_3/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_3/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_3/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_3/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_3/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_3/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_3/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_3/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_3/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_3/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_3/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_3/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_3/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_3/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_3/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_3/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_3/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_3/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_3/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_3/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_3/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_3/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_3/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_3/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_3/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_3/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_3/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_3/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_3/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_3/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_3/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_3/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_3/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_3/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_3/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_3/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_3/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_3/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_3/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_3/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_3/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_3/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_3/Assign_32Assignout_biassave_3/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_3/Assign_33Assignout_bias/Adamsave_3/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_3/Assign_34Assignout_bias/Adam_1save_3/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_3/Assign_35Assignout_weightssave_3/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_3/Assign_36Assignout_weights/Adamsave_3/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_3/Assign_37Assignout_weights/Adam_1save_3/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9

save_3/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_3/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_3/RestoreV2_1	RestoreV2save_3/Constsave_3/RestoreV2_1/tensor_names#save_3/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_3/Assign_38Assign
embeddingssave_3/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_3/Assign_39Assignembeddings/Adamsave_3/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_3/Assign_40Assignembeddings/Adam_1save_3/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_3/restore_shard_1NoOp^save_3/Assign_38^save_3/Assign_39^save_3/Assign_40
6
save_3/restore_all/NoOpNoOp^save_3/restore_shard
:
save_3/restore_all/NoOp_1NoOp^save_3/restore_shard_1
P
save_3/restore_allNoOp^save_3/restore_all/NoOp^save_3/restore_all/NoOp_1
R
save_4/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_8a49bdd494fb4acaad65e32c7be65acc/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
Ç
save_4/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_4/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename
`
save_4/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_4/ShardedFilename_1ShardedFilenamesave_4/StringJoinsave_4/ShardedFilename_1/shardsave_4/num_shards*
_output_shapes
: 

save_4/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_4/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_4/SaveV2_1SaveV2save_4/ShardedFilename_1save_4/SaveV2_1/tensor_names save_4/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_4/control_dependency_1Identitysave_4/ShardedFilename_1^save_4/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_4/ShardedFilename_1
Û
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilenamesave_4/ShardedFilename_1^save_4/control_dependency^save_4/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
 
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency^save_4/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_4/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_4/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_4/Assign_1Assignbeta2_powersave_4/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_4/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_4/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_4/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_4/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_4/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_4/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_4/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_4/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_4/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_4/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_4/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_4/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_4/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_4/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_4/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_4/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_4/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_4/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_4/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_4/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_4/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_4/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_4/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_4/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_4/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_4/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_4/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_4/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_4/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_4/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_4/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_4/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_4/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_4/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_4/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_4/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_4/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_4/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_4/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_4/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_4/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_4/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_4/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_4/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_4/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_4/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_4/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_4/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_4/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_4/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_4/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_4/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_4/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_4/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_4/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_4/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_4/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_4/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_4/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_4/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_4/Assign_32Assignout_biassave_4/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_4/Assign_33Assignout_bias/Adamsave_4/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_4/Assign_34Assignout_bias/Adam_1save_4/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_4/Assign_35Assignout_weightssave_4/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_4/Assign_36Assignout_weights/Adamsave_4/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_4/Assign_37Assignout_weights/Adam_1save_4/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9

save_4/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_4/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_4/RestoreV2_1	RestoreV2save_4/Constsave_4/RestoreV2_1/tensor_names#save_4/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_4/Assign_38Assign
embeddingssave_4/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_4/Assign_39Assignembeddings/Adamsave_4/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_4/Assign_40Assignembeddings/Adam_1save_4/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_4/restore_shard_1NoOp^save_4/Assign_38^save_4/Assign_39^save_4/Assign_40
6
save_4/restore_all/NoOpNoOp^save_4/restore_shard
:
save_4/restore_all/NoOp_1NoOp^save_4/restore_shard_1
P
save_4/restore_allNoOp^save_4/restore_all/NoOp^save_4/restore_all/NoOp_1
R
save_5/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_8b16896d6ee544bfaa385538b9eaab30/part*
dtype0*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
Ç
save_5/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_5/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_5/ShardedFilename
`
save_5/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_5/ShardedFilename_1ShardedFilenamesave_5/StringJoinsave_5/ShardedFilename_1/shardsave_5/num_shards*
_output_shapes
: 

save_5/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_5/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_5/SaveV2_1SaveV2save_5/ShardedFilename_1save_5/SaveV2_1/tensor_names save_5/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_5/control_dependency_1Identitysave_5/ShardedFilename_1^save_5/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_5/ShardedFilename_1
Û
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilenamesave_5/ShardedFilename_1^save_5/control_dependency^save_5/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
 
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency^save_5/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_5/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_5/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_5/Assign_1Assignbeta2_powersave_5/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_5/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_5/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_5/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_5/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_5/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_5/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_5/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_5/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_5/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_5/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_5/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_5/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_5/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_5/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_5/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_5/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_5/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_5/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_5/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_5/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_5/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_5/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_5/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_5/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_5/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_5/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_5/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_5/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_5/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_5/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_5/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_5/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_5/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_5/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_5/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_5/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_5/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_5/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_5/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_5/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_5/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_5/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_5/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_5/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_5/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_5/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_5/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_5/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_5/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_5/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_5/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_5/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_5/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_5/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_5/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_5/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_5/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_5/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_5/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_5/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_5/Assign_32Assignout_biassave_5/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_5/Assign_33Assignout_bias/Adamsave_5/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_5/Assign_34Assignout_bias/Adam_1save_5/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_5/Assign_35Assignout_weightssave_5/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_5/Assign_36Assignout_weights/Adamsave_5/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_5/Assign_37Assignout_weights/Adam_1save_5/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9

save_5/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_5/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_5/RestoreV2_1	RestoreV2save_5/Constsave_5/RestoreV2_1/tensor_names#save_5/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_5/Assign_38Assign
embeddingssave_5/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_5/Assign_39Assignembeddings/Adamsave_5/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_5/Assign_40Assignembeddings/Adam_1save_5/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_5/restore_shard_1NoOp^save_5/Assign_38^save_5/Assign_39^save_5/Assign_40
6
save_5/restore_all/NoOpNoOp^save_5/restore_shard
:
save_5/restore_all/NoOp_1NoOp^save_5/restore_shard_1
P
save_5/restore_allNoOp^save_5/restore_all/NoOp^save_5/restore_all/NoOp_1
R
save_6/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_3be514f2b23249f0a48a6a8c411a7336/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
Ç
save_6/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_6/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_6/ShardedFilename
`
save_6/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_6/ShardedFilename_1ShardedFilenamesave_6/StringJoinsave_6/ShardedFilename_1/shardsave_6/num_shards*
_output_shapes
: 

save_6/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_6/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_6/SaveV2_1SaveV2save_6/ShardedFilename_1save_6/SaveV2_1/tensor_names save_6/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_6/control_dependency_1Identitysave_6/ShardedFilename_1^save_6/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_6/ShardedFilename_1
Û
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilenamesave_6/ShardedFilename_1^save_6/control_dependency^save_6/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
 
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency^save_6/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_6/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_6/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_6/Assign_1Assignbeta2_powersave_6/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_6/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_6/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_6/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_6/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_6/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_6/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_6/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_6/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_6/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_6/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_6/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_6/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_6/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_6/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_6/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_6/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_6/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_6/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_6/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_6/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_6/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_6/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_6/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_6/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_6/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_6/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_6/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_6/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_6/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_6/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_6/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_6/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_6/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_6/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_6/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_6/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_6/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_6/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_6/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_6/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_6/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_6/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_6/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_6/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_6/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_6/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_6/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_6/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_6/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_6/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_6/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_6/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_6/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_6/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_6/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_6/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_6/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_6/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_6/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_6/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_6/Assign_32Assignout_biassave_6/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_6/Assign_33Assignout_bias/Adamsave_6/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_6/Assign_34Assignout_bias/Adam_1save_6/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_6/Assign_35Assignout_weightssave_6/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_6/Assign_36Assignout_weights/Adamsave_6/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_6/Assign_37Assignout_weights/Adam_1save_6/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9

save_6/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_6/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_6/RestoreV2_1	RestoreV2save_6/Constsave_6/RestoreV2_1/tensor_names#save_6/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_6/Assign_38Assign
embeddingssave_6/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_6/Assign_39Assignembeddings/Adamsave_6/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_6/Assign_40Assignembeddings/Adam_1save_6/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_6/restore_shard_1NoOp^save_6/Assign_38^save_6/Assign_39^save_6/Assign_40
6
save_6/restore_all/NoOpNoOp^save_6/restore_shard
:
save_6/restore_all/NoOp_1NoOp^save_6/restore_shard_1
P
save_6/restore_allNoOp^save_6/restore_all/NoOp^save_6/restore_all/NoOp_1
R
save_7/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_8f3350e8479744f6b7c987cc677f53fb/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
Ç
save_7/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_7/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_7/ShardedFilename
`
save_7/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_7/ShardedFilename_1ShardedFilenamesave_7/StringJoinsave_7/ShardedFilename_1/shardsave_7/num_shards*
_output_shapes
: 

save_7/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_7/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_7/SaveV2_1SaveV2save_7/ShardedFilename_1save_7/SaveV2_1/tensor_names save_7/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_7/control_dependency_1Identitysave_7/ShardedFilename_1^save_7/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_7/ShardedFilename_1
Û
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilenamesave_7/ShardedFilename_1^save_7/control_dependency^save_7/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
 
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency^save_7/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_7/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_7/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_7/Assign_1Assignbeta2_powersave_7/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_7/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_7/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_7/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_7/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_7/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_7/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_7/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_7/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_7/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_7/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_7/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_7/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_7/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_7/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_7/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_7/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_7/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_7/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_7/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_7/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_7/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_7/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_7/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_7/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_7/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_7/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_7/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_7/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_7/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_7/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_7/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_7/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_7/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_7/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_7/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_7/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_7/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_7/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_7/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_7/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_7/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_7/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_7/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_7/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_7/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_7/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_7/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_7/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_7/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_7/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_7/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_7/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_7/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_7/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_7/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_7/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_7/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_7/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_7/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_7/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_7/Assign_32Assignout_biassave_7/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_7/Assign_33Assignout_bias/Adamsave_7/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_7/Assign_34Assignout_bias/Adam_1save_7/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_7/Assign_35Assignout_weightssave_7/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_7/Assign_36Assignout_weights/Adamsave_7/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_7/Assign_37Assignout_weights/Adam_1save_7/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_4^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9

save_7/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_7/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_7/RestoreV2_1	RestoreV2save_7/Constsave_7/RestoreV2_1/tensor_names#save_7/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_7/Assign_38Assign
embeddingssave_7/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_7/Assign_39Assignembeddings/Adamsave_7/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_7/Assign_40Assignembeddings/Adam_1save_7/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_7/restore_shard_1NoOp^save_7/Assign_38^save_7/Assign_39^save_7/Assign_40
6
save_7/restore_all/NoOpNoOp^save_7/restore_shard
:
save_7/restore_all/NoOp_1NoOp^save_7/restore_shard_1
P
save_7/restore_allNoOp^save_7/restore_all/NoOp^save_7/restore_all/NoOp_1
R
save_8/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_a8588df7f92b4cab81c566b57e1bb48d/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_8/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
Ç
save_8/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_8/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_8/ShardedFilename
`
save_8/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_8/ShardedFilename_1ShardedFilenamesave_8/StringJoinsave_8/ShardedFilename_1/shardsave_8/num_shards*
_output_shapes
: 

save_8/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_8/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_8/SaveV2_1SaveV2save_8/ShardedFilename_1save_8/SaveV2_1/tensor_names save_8/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_8/control_dependency_1Identitysave_8/ShardedFilename_1^save_8/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_8/ShardedFilename_1
Û
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilenamesave_8/ShardedFilename_1^save_8/control_dependency^save_8/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
 
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency^save_8/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_8/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_8/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_8/Assign_1Assignbeta2_powersave_8/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_8/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_8/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_8/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_8/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_8/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_8/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_8/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_8/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_8/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_8/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_8/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_8/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_8/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_8/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_8/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_8/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_8/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_8/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_8/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_8/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_8/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_8/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_8/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_8/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_8/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_8/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_8/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_8/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_8/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_8/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_8/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_8/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_8/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_8/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_8/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_8/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_8/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_8/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_8/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_8/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_8/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_8/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_8/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_8/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_8/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_8/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_8/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_8/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_8/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_8/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_8/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_8/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_8/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_8/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_8/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_8/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_8/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_8/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_8/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_8/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_8/Assign_32Assignout_biassave_8/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_8/Assign_33Assignout_bias/Adamsave_8/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_8/Assign_34Assignout_bias/Adam_1save_8/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_8/Assign_35Assignout_weightssave_8/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_8/Assign_36Assignout_weights/Adamsave_8/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_8/Assign_37Assignout_weights/Adam_1save_8/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_4^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9

save_8/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_8/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_8/RestoreV2_1	RestoreV2save_8/Constsave_8/RestoreV2_1/tensor_names#save_8/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_8/Assign_38Assign
embeddingssave_8/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_8/Assign_39Assignembeddings/Adamsave_8/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_8/Assign_40Assignembeddings/Adam_1save_8/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_8/restore_shard_1NoOp^save_8/Assign_38^save_8/Assign_39^save_8/Assign_40
6
save_8/restore_all/NoOpNoOp^save_8/restore_shard
:
save_8/restore_all/NoOp_1NoOp^save_8/restore_shard_1
P
save_8/restore_allNoOp^save_8/restore_all/NoOp^save_8/restore_all/NoOp_1
R
save_9/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_9/StringJoin/inputs_1Const*<
value3B1 B+_temp_c9f4cc02ae7f4b9ab90db3df12468e18/part*
dtype0*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
Ç
save_9/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
±
save_9/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_9/ShardedFilename
`
save_9/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_9/ShardedFilename_1ShardedFilenamesave_9/StringJoinsave_9/ShardedFilename_1/shardsave_9/num_shards*
_output_shapes
: 

save_9/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
m
 save_9/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¶
save_9/SaveV2_1SaveV2save_9/ShardedFilename_1save_9/SaveV2_1/tensor_names save_9/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¡
save_9/control_dependency_1Identitysave_9/ShardedFilename_1^save_9/SaveV2_1*
T0*
_output_shapes
: *+
_class!
loc:@save_9/ShardedFilename_1
Û
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilenamesave_9/ShardedFilename_1^save_9/control_dependency^save_9/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
 
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency^save_9/control_dependency_1*
T0*
_output_shapes
: 
Ê
save_9/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
´
!save_9/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ô
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
¾
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Â
save_9/Assign_1Assignbeta2_powersave_9/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
å
save_9/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_9/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ê
save_9/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_9/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_9/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_9/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_9/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_9/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ó
save_9/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_9/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_9/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_9/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
å
save_9/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_9/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ê
save_9/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_9/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
î
save_9/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_9/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_9/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_9/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
õ
save_9/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_9/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_9/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_9/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ô
save_9/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_9/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ù
save_9/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_9/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_9/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_9/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
á
save_9/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_9/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
æ
save_9/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_9/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_9/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_9/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ô
save_9/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_9/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ù
save_9/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_9/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_9/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_9/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
á
save_9/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_9/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
æ
save_9/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_9/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_9/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_9/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ô
save_9/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_9/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ù
save_9/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_9/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_9/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_9/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
á
save_9/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_9/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
æ
save_9/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_9/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_9/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_9/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¤
save_9/Assign_32Assignout_biassave_9/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
©
save_9/Assign_33Assignout_bias/Adamsave_9/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_9/Assign_34Assignout_bias/Adam_1save_9/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
¯
save_9/Assign_35Assignout_weightssave_9/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
´
save_9/Assign_36Assignout_weights/Adamsave_9/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_9/Assign_37Assignout_weights/Adam_1save_9/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
â
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_4^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9

save_9/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
p
#save_9/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¨
save_9/RestoreV2_1	RestoreV2save_9/Constsave_9/RestoreV2_1/tensor_names#save_9/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
¬
save_9/Assign_38Assign
embeddingssave_9/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
³
save_9/Assign_39Assignembeddings/Adamsave_9/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_9/Assign_40Assignembeddings/Adam_1save_9/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
W
save_9/restore_shard_1NoOp^save_9/Assign_38^save_9/Assign_39^save_9/Assign_40
6
save_9/restore_all/NoOpNoOp^save_9/restore_shard
:
save_9/restore_all/NoOp_1NoOp^save_9/restore_shard_1
P
save_9/restore_allNoOp^save_9/restore_all/NoOp^save_9/restore_all/NoOp_1
S
save_10/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_10/StringJoin/inputs_1Const*<
value3B1 B+_temp_fa0f2ae2614c4839bccd3e9a3cbcce8a/part*
dtype0*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
È
save_10/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_10/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_10/ShardedFilename
a
save_10/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_10/ShardedFilename_1ShardedFilenamesave_10/StringJoinsave_10/ShardedFilename_1/shardsave_10/num_shards*
_output_shapes
: 

save_10/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_10/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_10/SaveV2_1SaveV2save_10/ShardedFilename_1save_10/SaveV2_1/tensor_names!save_10/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_10/control_dependency_1Identitysave_10/ShardedFilename_1^save_10/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_10/ShardedFilename_1
à
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilenamesave_10/ShardedFilename_1^save_10/control_dependency^save_10/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
¥
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency^save_10/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_10/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_10/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_10/Assign_1Assignbeta2_powersave_10/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_10/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_10/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_10/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_10/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_10/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_10/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_10/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_10/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_10/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_10/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_10/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_10/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_10/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_10/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_10/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_10/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_10/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_10/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_10/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_10/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_10/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_10/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_10/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_10/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_10/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_10/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_10/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_10/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_10/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_10/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_10/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_10/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_10/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_10/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_10/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_10/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_10/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_10/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_10/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_10/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_10/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_10/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_10/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_10/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_10/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_10/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_10/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_10/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_10/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_10/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_10/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_10/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_10/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_10/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_10/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_10/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_10/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_10/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_10/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_10/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_10/Assign_32Assignout_biassave_10/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_10/Assign_33Assignout_bias/Adamsave_10/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_10/Assign_34Assignout_bias/Adam_1save_10/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_10/Assign_35Assignout_weightssave_10/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_10/Assign_36Assignout_weights/Adamsave_10/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_10/Assign_37Assignout_weights/Adam_1save_10/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_4^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9

 save_10/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_10/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_10/RestoreV2_1	RestoreV2save_10/Const save_10/RestoreV2_1/tensor_names$save_10/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_10/Assign_38Assign
embeddingssave_10/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_10/Assign_39Assignembeddings/Adamsave_10/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_10/Assign_40Assignembeddings/Adam_1save_10/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_10/restore_shard_1NoOp^save_10/Assign_38^save_10/Assign_39^save_10/Assign_40
8
save_10/restore_all/NoOpNoOp^save_10/restore_shard
<
save_10/restore_all/NoOp_1NoOp^save_10/restore_shard_1
S
save_10/restore_allNoOp^save_10/restore_all/NoOp^save_10/restore_all/NoOp_1
S
save_11/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_11/StringJoin/inputs_1Const*<
value3B1 B+_temp_a210ceab8c234006a14d062db44d22ce/part*
dtype0*
_output_shapes
: 
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_11/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_11/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
È
save_11/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_11/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_11/ShardedFilename
a
save_11/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_11/ShardedFilename_1ShardedFilenamesave_11/StringJoinsave_11/ShardedFilename_1/shardsave_11/num_shards*
_output_shapes
: 

save_11/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_11/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_11/SaveV2_1SaveV2save_11/ShardedFilename_1save_11/SaveV2_1/tensor_names!save_11/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_11/control_dependency_1Identitysave_11/ShardedFilename_1^save_11/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_11/ShardedFilename_1
à
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilenamesave_11/ShardedFilename_1^save_11/control_dependency^save_11/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
¥
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency^save_11/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_11/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_11/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_11/AssignAssignbeta1_powersave_11/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_11/Assign_1Assignbeta2_powersave_11/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_11/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_11/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_11/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_11/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_11/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_11/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_11/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_11/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_11/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_11/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_11/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_11/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_11/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_11/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_11/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_11/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_11/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_11/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_11/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_11/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_11/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_11/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_11/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_11/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_11/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_11/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_11/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_11/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_11/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_11/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_11/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_11/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_11/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_11/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_11/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_11/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_11/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_11/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_11/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_11/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_11/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_11/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_11/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_11/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_11/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_11/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_11/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_11/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_11/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_11/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_11/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_11/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_11/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_11/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_11/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_11/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_11/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_11/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_11/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_11/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_11/Assign_32Assignout_biassave_11/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_11/Assign_33Assignout_bias/Adamsave_11/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_11/Assign_34Assignout_bias/Adam_1save_11/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_11/Assign_35Assignout_weightssave_11/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_11/Assign_36Assignout_weights/Adamsave_11/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_11/Assign_37Assignout_weights/Adam_1save_11/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_4^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9

 save_11/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_11/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_11/RestoreV2_1	RestoreV2save_11/Const save_11/RestoreV2_1/tensor_names$save_11/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_11/Assign_38Assign
embeddingssave_11/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_11/Assign_39Assignembeddings/Adamsave_11/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_11/Assign_40Assignembeddings/Adam_1save_11/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_11/restore_shard_1NoOp^save_11/Assign_38^save_11/Assign_39^save_11/Assign_40
8
save_11/restore_all/NoOpNoOp^save_11/restore_shard
<
save_11/restore_all/NoOp_1NoOp^save_11/restore_shard_1
S
save_11/restore_allNoOp^save_11/restore_all/NoOp^save_11/restore_all/NoOp_1
S
save_12/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_12/StringJoin/inputs_1Const*<
value3B1 B+_temp_a8a03bb3f5a3498b8ee415f2ae253406/part*
dtype0*
_output_shapes
: 
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_12/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
È
save_12/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_12/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_12/ShardedFilename
a
save_12/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_12/ShardedFilename_1ShardedFilenamesave_12/StringJoinsave_12/ShardedFilename_1/shardsave_12/num_shards*
_output_shapes
: 

save_12/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_12/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_12/SaveV2_1SaveV2save_12/ShardedFilename_1save_12/SaveV2_1/tensor_names!save_12/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_12/control_dependency_1Identitysave_12/ShardedFilename_1^save_12/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_12/ShardedFilename_1
à
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilenamesave_12/ShardedFilename_1^save_12/control_dependency^save_12/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
¥
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency^save_12/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_12/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_12/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_12/Assign_1Assignbeta2_powersave_12/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_12/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_12/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_12/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_12/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_12/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_12/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_12/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_12/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_12/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_12/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_12/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_12/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_12/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_12/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_12/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_12/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_12/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_12/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_12/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_12/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_12/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_12/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_12/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_12/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_12/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_12/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_12/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_12/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_12/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_12/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_12/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_12/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_12/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_12/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_12/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_12/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_12/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_12/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_12/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_12/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_12/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_12/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_12/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_12/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_12/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_12/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_12/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_12/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_12/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_12/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_12/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_12/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_12/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_12/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_12/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_12/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_12/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_12/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_12/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_12/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_12/Assign_32Assignout_biassave_12/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_12/Assign_33Assignout_bias/Adamsave_12/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_12/Assign_34Assignout_bias/Adam_1save_12/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_12/Assign_35Assignout_weightssave_12/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_12/Assign_36Assignout_weights/Adamsave_12/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_12/Assign_37Assignout_weights/Adam_1save_12/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_4^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9

 save_12/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_12/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_12/RestoreV2_1	RestoreV2save_12/Const save_12/RestoreV2_1/tensor_names$save_12/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_12/Assign_38Assign
embeddingssave_12/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_12/Assign_39Assignembeddings/Adamsave_12/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_12/Assign_40Assignembeddings/Adam_1save_12/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_12/restore_shard_1NoOp^save_12/Assign_38^save_12/Assign_39^save_12/Assign_40
8
save_12/restore_all/NoOpNoOp^save_12/restore_shard
<
save_12/restore_all/NoOp_1NoOp^save_12/restore_shard_1
S
save_12/restore_allNoOp^save_12/restore_all/NoOp^save_12/restore_all/NoOp_1
S
save_13/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_13/StringJoin/inputs_1Const*<
value3B1 B+_temp_d113e682e9ae496cbf0094b995ca2678/part*
dtype0*
_output_shapes
: 
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_13/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
È
save_13/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_13/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_13/ShardedFilename
a
save_13/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_13/ShardedFilename_1ShardedFilenamesave_13/StringJoinsave_13/ShardedFilename_1/shardsave_13/num_shards*
_output_shapes
: 

save_13/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_13/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_13/SaveV2_1SaveV2save_13/ShardedFilename_1save_13/SaveV2_1/tensor_names!save_13/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_13/control_dependency_1Identitysave_13/ShardedFilename_1^save_13/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_13/ShardedFilename_1
à
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilenamesave_13/ShardedFilename_1^save_13/control_dependency^save_13/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
¥
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency^save_13/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_13/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_13/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_13/Assign_1Assignbeta2_powersave_13/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_13/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_13/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_13/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_13/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_13/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_13/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_13/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_13/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_13/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_13/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_13/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_13/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_13/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_13/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_13/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_13/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_13/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_13/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_13/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_13/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_13/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_13/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_13/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_13/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_13/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_13/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_13/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_13/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_13/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_13/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_13/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_13/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_13/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_13/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_13/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_13/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_13/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_13/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_13/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_13/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_13/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_13/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_13/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_13/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_13/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_13/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_13/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_13/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_13/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_13/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_13/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_13/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_13/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_13/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_13/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_13/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_13/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_13/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_13/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_13/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_13/Assign_32Assignout_biassave_13/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_13/Assign_33Assignout_bias/Adamsave_13/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_13/Assign_34Assignout_bias/Adam_1save_13/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_13/Assign_35Assignout_weightssave_13/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_13/Assign_36Assignout_weights/Adamsave_13/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_13/Assign_37Assignout_weights/Adam_1save_13/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_4^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9

 save_13/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_13/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_13/RestoreV2_1	RestoreV2save_13/Const save_13/RestoreV2_1/tensor_names$save_13/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_13/Assign_38Assign
embeddingssave_13/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_13/Assign_39Assignembeddings/Adamsave_13/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_13/Assign_40Assignembeddings/Adam_1save_13/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_13/restore_shard_1NoOp^save_13/Assign_38^save_13/Assign_39^save_13/Assign_40
8
save_13/restore_all/NoOpNoOp^save_13/restore_shard
<
save_13/restore_all/NoOp_1NoOp^save_13/restore_shard_1
S
save_13/restore_allNoOp^save_13/restore_all/NoOp^save_13/restore_all/NoOp_1
S
save_14/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_14/StringJoin/inputs_1Const*<
value3B1 B+_temp_51e8bae34d344eef94b4b81c37388ef3/part*
dtype0*
_output_shapes
: 
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_14/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_14/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
È
save_14/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_14/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_14/ShardedFilename
a
save_14/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_14/ShardedFilename_1ShardedFilenamesave_14/StringJoinsave_14/ShardedFilename_1/shardsave_14/num_shards*
_output_shapes
: 

save_14/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_14/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_14/SaveV2_1SaveV2save_14/ShardedFilename_1save_14/SaveV2_1/tensor_names!save_14/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_14/control_dependency_1Identitysave_14/ShardedFilename_1^save_14/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_14/ShardedFilename_1
à
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilenamesave_14/ShardedFilename_1^save_14/control_dependency^save_14/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
¥
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency^save_14/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_14/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_14/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_14/Assign_1Assignbeta2_powersave_14/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_14/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_14/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_14/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_14/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_14/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_14/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_14/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_14/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_14/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_14/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_14/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_14/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_14/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_14/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_14/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_14/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_14/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_14/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_14/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_14/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_14/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_14/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_14/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_14/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_14/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_14/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_14/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_14/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_14/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_14/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_14/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_14/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_14/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_14/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_14/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_14/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_14/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_14/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_14/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_14/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_14/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_14/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_14/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_14/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_14/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_14/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_14/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_14/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_14/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_14/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_14/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_14/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_14/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_14/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_14/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_14/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_14/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_14/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_14/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_14/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_14/Assign_32Assignout_biassave_14/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_14/Assign_33Assignout_bias/Adamsave_14/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_14/Assign_34Assignout_bias/Adam_1save_14/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_14/Assign_35Assignout_weightssave_14/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_14/Assign_36Assignout_weights/Adamsave_14/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_14/Assign_37Assignout_weights/Adam_1save_14/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_4^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9

 save_14/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_14/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_14/RestoreV2_1	RestoreV2save_14/Const save_14/RestoreV2_1/tensor_names$save_14/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_14/Assign_38Assign
embeddingssave_14/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_14/Assign_39Assignembeddings/Adamsave_14/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_14/Assign_40Assignembeddings/Adam_1save_14/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_14/restore_shard_1NoOp^save_14/Assign_38^save_14/Assign_39^save_14/Assign_40
8
save_14/restore_all/NoOpNoOp^save_14/restore_shard
<
save_14/restore_all/NoOp_1NoOp^save_14/restore_shard_1
S
save_14/restore_allNoOp^save_14/restore_all/NoOp^save_14/restore_all/NoOp_1
S
save_15/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_15/StringJoin/inputs_1Const*<
value3B1 B+_temp_3492040a8a1342649a7d37d43f19fa4c/part*
dtype0*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_15/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
È
save_15/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_15/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_15/ShardedFilename
a
save_15/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_15/ShardedFilename_1ShardedFilenamesave_15/StringJoinsave_15/ShardedFilename_1/shardsave_15/num_shards*
_output_shapes
: 

save_15/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_15/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_15/SaveV2_1SaveV2save_15/ShardedFilename_1save_15/SaveV2_1/tensor_names!save_15/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_15/control_dependency_1Identitysave_15/ShardedFilename_1^save_15/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_15/ShardedFilename_1
à
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilenamesave_15/ShardedFilename_1^save_15/control_dependency^save_15/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
¥
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency^save_15/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_15/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_15/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_15/Assign_1Assignbeta2_powersave_15/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_15/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_15/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_15/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_15/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_15/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_15/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_15/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_15/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_15/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_15/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_15/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_15/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_15/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_15/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_15/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_15/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_15/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_15/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_15/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_15/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_15/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_15/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_15/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_15/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_15/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_15/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_15/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_15/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_15/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_15/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_15/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_15/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_15/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_15/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_15/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_15/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_15/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_15/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_15/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_15/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_15/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_15/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_15/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_15/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_15/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_15/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_15/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_15/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_15/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_15/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_15/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_15/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_15/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_15/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_15/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_15/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_15/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_15/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_15/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_15/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_15/Assign_32Assignout_biassave_15/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_15/Assign_33Assignout_bias/Adamsave_15/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_15/Assign_34Assignout_bias/Adam_1save_15/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_15/Assign_35Assignout_weightssave_15/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_15/Assign_36Assignout_weights/Adamsave_15/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_15/Assign_37Assignout_weights/Adam_1save_15/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_4^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9

 save_15/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_15/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_15/RestoreV2_1	RestoreV2save_15/Const save_15/RestoreV2_1/tensor_names$save_15/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_15/Assign_38Assign
embeddingssave_15/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_15/Assign_39Assignembeddings/Adamsave_15/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_15/Assign_40Assignembeddings/Adam_1save_15/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_15/restore_shard_1NoOp^save_15/Assign_38^save_15/Assign_39^save_15/Assign_40
8
save_15/restore_all/NoOpNoOp^save_15/restore_shard
<
save_15/restore_all/NoOp_1NoOp^save_15/restore_shard_1
S
save_15/restore_allNoOp^save_15/restore_all/NoOp^save_15/restore_all/NoOp_1
S
save_16/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_16/StringJoin/inputs_1Const*<
value3B1 B+_temp_1b90289c29f2419a83e595540a5764f4/part*
dtype0*
_output_shapes
: 
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_16/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_16/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
È
save_16/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_16/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_16/ShardedFilename
a
save_16/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_16/ShardedFilename_1ShardedFilenamesave_16/StringJoinsave_16/ShardedFilename_1/shardsave_16/num_shards*
_output_shapes
: 

save_16/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_16/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_16/SaveV2_1SaveV2save_16/ShardedFilename_1save_16/SaveV2_1/tensor_names!save_16/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_16/control_dependency_1Identitysave_16/ShardedFilename_1^save_16/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_16/ShardedFilename_1
à
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilenamesave_16/ShardedFilename_1^save_16/control_dependency^save_16/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(
¥
save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency^save_16/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_16/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_16/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_16/AssignAssignbeta1_powersave_16/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_16/Assign_1Assignbeta2_powersave_16/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_16/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_16/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_16/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_16/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_16/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_16/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_16/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_16/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_16/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_16/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_16/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_16/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_16/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_16/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_16/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_16/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_16/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_16/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_16/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_16/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_16/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_16/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_16/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_16/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_16/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_16/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_16/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_16/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_16/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_16/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_16/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_16/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_16/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_16/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_16/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_16/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_16/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_16/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_16/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_16/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_16/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_16/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_16/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_16/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_16/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_16/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_16/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_16/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_16/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_16/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_16/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_16/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_16/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_16/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_16/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_16/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_16/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_16/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_16/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_16/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_16/Assign_32Assignout_biassave_16/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_16/Assign_33Assignout_bias/Adamsave_16/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_16/Assign_34Assignout_bias/Adam_1save_16/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_16/Assign_35Assignout_weightssave_16/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_16/Assign_36Assignout_weights/Adamsave_16/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_16/Assign_37Assignout_weights/Adam_1save_16/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_20^save_16/Assign_21^save_16/Assign_22^save_16/Assign_23^save_16/Assign_24^save_16/Assign_25^save_16/Assign_26^save_16/Assign_27^save_16/Assign_28^save_16/Assign_29^save_16/Assign_3^save_16/Assign_30^save_16/Assign_31^save_16/Assign_32^save_16/Assign_33^save_16/Assign_34^save_16/Assign_35^save_16/Assign_36^save_16/Assign_37^save_16/Assign_4^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9

 save_16/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_16/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_16/RestoreV2_1	RestoreV2save_16/Const save_16/RestoreV2_1/tensor_names$save_16/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_16/Assign_38Assign
embeddingssave_16/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_16/Assign_39Assignembeddings/Adamsave_16/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_16/Assign_40Assignembeddings/Adam_1save_16/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_16/restore_shard_1NoOp^save_16/Assign_38^save_16/Assign_39^save_16/Assign_40
8
save_16/restore_all/NoOpNoOp^save_16/restore_shard
<
save_16/restore_all/NoOp_1NoOp^save_16/restore_shard_1
S
save_16/restore_allNoOp^save_16/restore_all/NoOp^save_16/restore_all/NoOp_1
S
save_17/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_17/StringJoin/inputs_1Const*<
value3B1 B+_temp_e347c0fb3c0e4c6bae4dd6eff78560e0/part*
dtype0*
_output_shapes
: 
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_17/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
È
save_17/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_17/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_17/ShardedFilename
a
save_17/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_17/ShardedFilename_1ShardedFilenamesave_17/StringJoinsave_17/ShardedFilename_1/shardsave_17/num_shards*
_output_shapes
: 

save_17/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_17/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_17/SaveV2_1SaveV2save_17/ShardedFilename_1save_17/SaveV2_1/tensor_names!save_17/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_17/control_dependency_1Identitysave_17/ShardedFilename_1^save_17/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_17/ShardedFilename_1
à
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilenamesave_17/ShardedFilename_1^save_17/control_dependency^save_17/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(
¥
save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency^save_17/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_17/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_17/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_17/AssignAssignbeta1_powersave_17/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_17/Assign_1Assignbeta2_powersave_17/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_17/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_17/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_17/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_17/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_17/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_17/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_17/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_17/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_17/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_17/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_17/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_17/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_17/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_17/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_17/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_17/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_17/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_17/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_17/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_17/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_17/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_17/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_17/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_17/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_17/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_17/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_17/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_17/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_17/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_17/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_17/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_17/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_17/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_17/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_17/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_17/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_17/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_17/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_17/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_17/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_17/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_17/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_17/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_17/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_17/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_17/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_17/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_17/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_17/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_17/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_17/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_17/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_17/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_17/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_17/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_17/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_17/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_17/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_17/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_17/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_17/Assign_32Assignout_biassave_17/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_17/Assign_33Assignout_bias/Adamsave_17/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_17/Assign_34Assignout_bias/Adam_1save_17/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_17/Assign_35Assignout_weightssave_17/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_17/Assign_36Assignout_weights/Adamsave_17/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_17/Assign_37Assignout_weights/Adam_1save_17/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_20^save_17/Assign_21^save_17/Assign_22^save_17/Assign_23^save_17/Assign_24^save_17/Assign_25^save_17/Assign_26^save_17/Assign_27^save_17/Assign_28^save_17/Assign_29^save_17/Assign_3^save_17/Assign_30^save_17/Assign_31^save_17/Assign_32^save_17/Assign_33^save_17/Assign_34^save_17/Assign_35^save_17/Assign_36^save_17/Assign_37^save_17/Assign_4^save_17/Assign_5^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9

 save_17/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_17/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_17/RestoreV2_1	RestoreV2save_17/Const save_17/RestoreV2_1/tensor_names$save_17/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_17/Assign_38Assign
embeddingssave_17/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_17/Assign_39Assignembeddings/Adamsave_17/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_17/Assign_40Assignembeddings/Adam_1save_17/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_17/restore_shard_1NoOp^save_17/Assign_38^save_17/Assign_39^save_17/Assign_40
8
save_17/restore_all/NoOpNoOp^save_17/restore_shard
<
save_17/restore_all/NoOp_1NoOp^save_17/restore_shard_1
S
save_17/restore_allNoOp^save_17/restore_all/NoOp^save_17/restore_all/NoOp_1
S
save_18/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_18/StringJoin/inputs_1Const*<
value3B1 B+_temp_de1a111749984724b03e6618ee92552e/part*
dtype0*
_output_shapes
: 
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_18/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_18/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
È
save_18/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_18/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_18/ShardedFilename
a
save_18/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_18/ShardedFilename_1ShardedFilenamesave_18/StringJoinsave_18/ShardedFilename_1/shardsave_18/num_shards*
_output_shapes
: 

save_18/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_18/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_18/SaveV2_1SaveV2save_18/ShardedFilename_1save_18/SaveV2_1/tensor_names!save_18/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_18/control_dependency_1Identitysave_18/ShardedFilename_1^save_18/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_18/ShardedFilename_1
à
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilenamesave_18/ShardedFilename_1^save_18/control_dependency^save_18/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(
¥
save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency^save_18/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_18/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_18/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_18/AssignAssignbeta1_powersave_18/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_18/Assign_1Assignbeta2_powersave_18/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_18/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_18/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_18/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_18/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_18/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_18/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_18/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_18/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_18/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_18/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_18/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_18/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_18/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_18/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_18/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_18/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_18/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_18/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_18/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_18/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_18/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_18/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_18/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_18/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_18/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_18/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_18/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_18/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_18/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_18/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_18/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_18/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_18/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_18/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_18/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_18/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_18/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_18/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_18/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_18/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_18/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_18/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_18/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_18/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_18/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_18/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_18/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_18/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_18/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_18/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_18/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_18/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_18/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_18/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_18/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_18/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_18/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_18/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_18/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_18/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_18/Assign_32Assignout_biassave_18/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_18/Assign_33Assignout_bias/Adamsave_18/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_18/Assign_34Assignout_bias/Adam_1save_18/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_18/Assign_35Assignout_weightssave_18/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_18/Assign_36Assignout_weights/Adamsave_18/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_18/Assign_37Assignout_weights/Adam_1save_18/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_20^save_18/Assign_21^save_18/Assign_22^save_18/Assign_23^save_18/Assign_24^save_18/Assign_25^save_18/Assign_26^save_18/Assign_27^save_18/Assign_28^save_18/Assign_29^save_18/Assign_3^save_18/Assign_30^save_18/Assign_31^save_18/Assign_32^save_18/Assign_33^save_18/Assign_34^save_18/Assign_35^save_18/Assign_36^save_18/Assign_37^save_18/Assign_4^save_18/Assign_5^save_18/Assign_6^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9

 save_18/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_18/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_18/RestoreV2_1	RestoreV2save_18/Const save_18/RestoreV2_1/tensor_names$save_18/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_18/Assign_38Assign
embeddingssave_18/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_18/Assign_39Assignembeddings/Adamsave_18/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_18/Assign_40Assignembeddings/Adam_1save_18/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_18/restore_shard_1NoOp^save_18/Assign_38^save_18/Assign_39^save_18/Assign_40
8
save_18/restore_all/NoOpNoOp^save_18/restore_shard
<
save_18/restore_all/NoOp_1NoOp^save_18/restore_shard_1
S
save_18/restore_allNoOp^save_18/restore_all/NoOp^save_18/restore_all/NoOp_1
S
save_19/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_19/StringJoin/inputs_1Const*<
value3B1 B+_temp_8bfae581e3384d2391cc7475ade05522/part*
dtype0*
_output_shapes
: 
~
save_19/StringJoin
StringJoinsave_19/Constsave_19/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_19/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_19/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_19/ShardedFilenameShardedFilenamesave_19/StringJoinsave_19/ShardedFilename/shardsave_19/num_shards*
_output_shapes
: 
È
save_19/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_19/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_19/SaveV2SaveV2save_19/ShardedFilenamesave_19/SaveV2/tensor_namessave_19/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_19/control_dependencyIdentitysave_19/ShardedFilename^save_19/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_19/ShardedFilename
a
save_19/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_19/ShardedFilename_1ShardedFilenamesave_19/StringJoinsave_19/ShardedFilename_1/shardsave_19/num_shards*
_output_shapes
: 

save_19/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_19/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_19/SaveV2_1SaveV2save_19/ShardedFilename_1save_19/SaveV2_1/tensor_names!save_19/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_19/control_dependency_1Identitysave_19/ShardedFilename_1^save_19/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_19/ShardedFilename_1
à
.save_19/MergeV2Checkpoints/checkpoint_prefixesPacksave_19/ShardedFilenamesave_19/ShardedFilename_1^save_19/control_dependency^save_19/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_19/MergeV2CheckpointsMergeV2Checkpoints.save_19/MergeV2Checkpoints/checkpoint_prefixessave_19/Const*
delete_old_dirs(
¥
save_19/IdentityIdentitysave_19/Const^save_19/MergeV2Checkpoints^save_19/control_dependency^save_19/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_19/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_19/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_19/RestoreV2	RestoreV2save_19/Constsave_19/RestoreV2/tensor_names"save_19/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_19/AssignAssignbeta1_powersave_19/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_19/Assign_1Assignbeta2_powersave_19/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_19/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_19/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_19/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_19/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_19/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_19/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_19/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_19/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_19/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_19/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_19/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_19/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_19/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_19/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_19/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_19/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_19/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_19/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_19/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_19/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_19/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_19/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_19/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_19/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_19/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_19/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_19/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_19/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_19/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_19/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_19/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_19/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_19/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_19/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_19/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_19/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_19/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_19/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_19/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_19/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_19/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_19/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_19/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_19/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_19/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_19/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_19/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_19/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_19/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_19/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_19/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_19/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_19/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_19/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_19/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_19/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_19/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_19/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_19/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_19/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_19/Assign_32Assignout_biassave_19/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_19/Assign_33Assignout_bias/Adamsave_19/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_19/Assign_34Assignout_bias/Adam_1save_19/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_19/Assign_35Assignout_weightssave_19/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_19/Assign_36Assignout_weights/Adamsave_19/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_19/Assign_37Assignout_weights/Adam_1save_19/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_19/restore_shardNoOp^save_19/Assign^save_19/Assign_1^save_19/Assign_10^save_19/Assign_11^save_19/Assign_12^save_19/Assign_13^save_19/Assign_14^save_19/Assign_15^save_19/Assign_16^save_19/Assign_17^save_19/Assign_18^save_19/Assign_19^save_19/Assign_2^save_19/Assign_20^save_19/Assign_21^save_19/Assign_22^save_19/Assign_23^save_19/Assign_24^save_19/Assign_25^save_19/Assign_26^save_19/Assign_27^save_19/Assign_28^save_19/Assign_29^save_19/Assign_3^save_19/Assign_30^save_19/Assign_31^save_19/Assign_32^save_19/Assign_33^save_19/Assign_34^save_19/Assign_35^save_19/Assign_36^save_19/Assign_37^save_19/Assign_4^save_19/Assign_5^save_19/Assign_6^save_19/Assign_7^save_19/Assign_8^save_19/Assign_9

 save_19/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_19/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_19/RestoreV2_1	RestoreV2save_19/Const save_19/RestoreV2_1/tensor_names$save_19/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_19/Assign_38Assign
embeddingssave_19/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_19/Assign_39Assignembeddings/Adamsave_19/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_19/Assign_40Assignembeddings/Adam_1save_19/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_19/restore_shard_1NoOp^save_19/Assign_38^save_19/Assign_39^save_19/Assign_40
8
save_19/restore_all/NoOpNoOp^save_19/restore_shard
<
save_19/restore_all/NoOp_1NoOp^save_19/restore_shard_1
S
save_19/restore_allNoOp^save_19/restore_all/NoOp^save_19/restore_all/NoOp_1
S
save_20/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_20/StringJoin/inputs_1Const*<
value3B1 B+_temp_e2de57b9341243c58c52d9da0b827041/part*
dtype0*
_output_shapes
: 
~
save_20/StringJoin
StringJoinsave_20/Constsave_20/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_20/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_20/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_20/ShardedFilenameShardedFilenamesave_20/StringJoinsave_20/ShardedFilename/shardsave_20/num_shards*
_output_shapes
: 
È
save_20/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_20/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_20/SaveV2SaveV2save_20/ShardedFilenamesave_20/SaveV2/tensor_namessave_20/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_20/control_dependencyIdentitysave_20/ShardedFilename^save_20/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_20/ShardedFilename
a
save_20/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_20/ShardedFilename_1ShardedFilenamesave_20/StringJoinsave_20/ShardedFilename_1/shardsave_20/num_shards*
_output_shapes
: 

save_20/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_20/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_20/SaveV2_1SaveV2save_20/ShardedFilename_1save_20/SaveV2_1/tensor_names!save_20/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_20/control_dependency_1Identitysave_20/ShardedFilename_1^save_20/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_20/ShardedFilename_1
à
.save_20/MergeV2Checkpoints/checkpoint_prefixesPacksave_20/ShardedFilenamesave_20/ShardedFilename_1^save_20/control_dependency^save_20/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_20/MergeV2CheckpointsMergeV2Checkpoints.save_20/MergeV2Checkpoints/checkpoint_prefixessave_20/Const*
delete_old_dirs(
¥
save_20/IdentityIdentitysave_20/Const^save_20/MergeV2Checkpoints^save_20/control_dependency^save_20/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_20/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_20/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_20/RestoreV2	RestoreV2save_20/Constsave_20/RestoreV2/tensor_names"save_20/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_20/AssignAssignbeta1_powersave_20/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_20/Assign_1Assignbeta2_powersave_20/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_20/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_20/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_20/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_20/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_20/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_20/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_20/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_20/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_20/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_20/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_20/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_20/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_20/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_20/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_20/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_20/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_20/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_20/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_20/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_20/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_20/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_20/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_20/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_20/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_20/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_20/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_20/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_20/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_20/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_20/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_20/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_20/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_20/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_20/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_20/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_20/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_20/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_20/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_20/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_20/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_20/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_20/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_20/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_20/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_20/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_20/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_20/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_20/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_20/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_20/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_20/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_20/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_20/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_20/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_20/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_20/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_20/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_20/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_20/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_20/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_20/Assign_32Assignout_biassave_20/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_20/Assign_33Assignout_bias/Adamsave_20/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_20/Assign_34Assignout_bias/Adam_1save_20/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_20/Assign_35Assignout_weightssave_20/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_20/Assign_36Assignout_weights/Adamsave_20/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_20/Assign_37Assignout_weights/Adam_1save_20/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_20/restore_shardNoOp^save_20/Assign^save_20/Assign_1^save_20/Assign_10^save_20/Assign_11^save_20/Assign_12^save_20/Assign_13^save_20/Assign_14^save_20/Assign_15^save_20/Assign_16^save_20/Assign_17^save_20/Assign_18^save_20/Assign_19^save_20/Assign_2^save_20/Assign_20^save_20/Assign_21^save_20/Assign_22^save_20/Assign_23^save_20/Assign_24^save_20/Assign_25^save_20/Assign_26^save_20/Assign_27^save_20/Assign_28^save_20/Assign_29^save_20/Assign_3^save_20/Assign_30^save_20/Assign_31^save_20/Assign_32^save_20/Assign_33^save_20/Assign_34^save_20/Assign_35^save_20/Assign_36^save_20/Assign_37^save_20/Assign_4^save_20/Assign_5^save_20/Assign_6^save_20/Assign_7^save_20/Assign_8^save_20/Assign_9

 save_20/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_20/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_20/RestoreV2_1	RestoreV2save_20/Const save_20/RestoreV2_1/tensor_names$save_20/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_20/Assign_38Assign
embeddingssave_20/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_20/Assign_39Assignembeddings/Adamsave_20/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_20/Assign_40Assignembeddings/Adam_1save_20/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_20/restore_shard_1NoOp^save_20/Assign_38^save_20/Assign_39^save_20/Assign_40
8
save_20/restore_all/NoOpNoOp^save_20/restore_shard
<
save_20/restore_all/NoOp_1NoOp^save_20/restore_shard_1
S
save_20/restore_allNoOp^save_20/restore_all/NoOp^save_20/restore_all/NoOp_1
S
save_21/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_21/StringJoin/inputs_1Const*<
value3B1 B+_temp_00bfd4dd03494e08a2c2379053386cb0/part*
dtype0*
_output_shapes
: 
~
save_21/StringJoin
StringJoinsave_21/Constsave_21/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_21/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_21/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_21/ShardedFilenameShardedFilenamesave_21/StringJoinsave_21/ShardedFilename/shardsave_21/num_shards*
_output_shapes
: 
È
save_21/SaveV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
²
save_21/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&

save_21/SaveV2SaveV2save_21/ShardedFilenamesave_21/SaveV2/tensor_namessave_21/SaveV2/shape_and_slicesbeta1_powerbeta2_power)bidi_/bidirectional_rnn/bw/lstm_cell/bias.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/bw/lstm_cell/kernel0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1)bidi_/bidirectional_rnn/fw/lstm_cell/bias.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1+bidi_/bidirectional_rnn/fw/lstm_cell/kernel0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 conv_maxpool_3_conv_1/B_filter_0%conv_maxpool_3_conv_1/B_filter_0/Adam'conv_maxpool_3_conv_1/B_filter_0/Adam_1 conv_maxpool_3_conv_1/W_filter_0%conv_maxpool_3_conv_1/W_filter_0/Adam'conv_maxpool_3_conv_1/W_filter_0/Adam_1 conv_maxpool_5_conv_1/B_filter_1%conv_maxpool_5_conv_1/B_filter_1/Adam'conv_maxpool_5_conv_1/B_filter_1/Adam_1 conv_maxpool_5_conv_1/W_filter_1%conv_maxpool_5_conv_1/W_filter_1/Adam'conv_maxpool_5_conv_1/W_filter_1/Adam_1 conv_maxpool_7_conv_1/B_filter_2%conv_maxpool_7_conv_1/B_filter_2/Adam'conv_maxpool_7_conv_1/B_filter_2/Adam_1 conv_maxpool_7_conv_1/W_filter_2%conv_maxpool_7_conv_1/W_filter_2/Adam'conv_maxpool_7_conv_1/W_filter_2/Adam_1out_biasout_bias/Adamout_bias/Adam_1out_weightsout_weights/Adamout_weights/Adam_1*4
dtypes*
(2&

save_21/control_dependencyIdentitysave_21/ShardedFilename^save_21/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_21/ShardedFilename
a
save_21/ShardedFilename_1/shardConst*
value	B :*
dtype0*
_output_shapes
: 

save_21/ShardedFilename_1ShardedFilenamesave_21/StringJoinsave_21/ShardedFilename_1/shardsave_21/num_shards*
_output_shapes
: 

save_21/SaveV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
n
!save_21/SaveV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
º
save_21/SaveV2_1SaveV2save_21/ShardedFilename_1save_21/SaveV2_1/tensor_names!save_21/SaveV2_1/shape_and_slices
embeddingsembeddings/Adamembeddings/Adam_1*
dtypes
2
¥
save_21/control_dependency_1Identitysave_21/ShardedFilename_1^save_21/SaveV2_1*
T0*
_output_shapes
: *,
_class"
 loc:@save_21/ShardedFilename_1
à
.save_21/MergeV2Checkpoints/checkpoint_prefixesPacksave_21/ShardedFilenamesave_21/ShardedFilename_1^save_21/control_dependency^save_21/control_dependency_1*

axis *
T0*
N*
_output_shapes
:

save_21/MergeV2CheckpointsMergeV2Checkpoints.save_21/MergeV2Checkpoints/checkpoint_prefixessave_21/Const*
delete_old_dirs(
¥
save_21/IdentityIdentitysave_21/Const^save_21/MergeV2Checkpoints^save_21/control_dependency^save_21/control_dependency_1*
T0*
_output_shapes
: 
Ë
save_21/RestoreV2/tensor_namesConst*ø

valueî
Bë
&Bbeta1_powerBbeta2_powerB)bidi_/bidirectional_rnn/bw/lstm_cell/biasB.bidi_/bidirectional_rnn/bw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/bw/lstm_cell/kernelB0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B)bidi_/bidirectional_rnn/fw/lstm_cell/biasB.bidi_/bidirectional_rnn/fw/lstm_cell/bias/AdamB0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B+bidi_/bidirectional_rnn/fw/lstm_cell/kernelB0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/AdamB2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B conv_maxpool_3_conv_1/B_filter_0B%conv_maxpool_3_conv_1/B_filter_0/AdamB'conv_maxpool_3_conv_1/B_filter_0/Adam_1B conv_maxpool_3_conv_1/W_filter_0B%conv_maxpool_3_conv_1/W_filter_0/AdamB'conv_maxpool_3_conv_1/W_filter_0/Adam_1B conv_maxpool_5_conv_1/B_filter_1B%conv_maxpool_5_conv_1/B_filter_1/AdamB'conv_maxpool_5_conv_1/B_filter_1/Adam_1B conv_maxpool_5_conv_1/W_filter_1B%conv_maxpool_5_conv_1/W_filter_1/AdamB'conv_maxpool_5_conv_1/W_filter_1/Adam_1B conv_maxpool_7_conv_1/B_filter_2B%conv_maxpool_7_conv_1/B_filter_2/AdamB'conv_maxpool_7_conv_1/B_filter_2/Adam_1B conv_maxpool_7_conv_1/W_filter_2B%conv_maxpool_7_conv_1/W_filter_2/AdamB'conv_maxpool_7_conv_1/W_filter_2/Adam_1Bout_biasBout_bias/AdamBout_bias/Adam_1Bout_weightsBout_weights/AdamBout_weights/Adam_1*
dtype0*
_output_shapes
:&
µ
"save_21/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
Ø
save_21/RestoreV2	RestoreV2save_21/Constsave_21/RestoreV2/tensor_names"save_21/RestoreV2/shape_and_slices*4
dtypes*
(2&*®
_output_shapes
::::::::::::::::::::::::::::::::::::::
À
save_21/AssignAssignbeta1_powersave_21/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
Ä
save_21/Assign_1Assignbeta2_powersave_21/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ç
save_21/Assign_2Assign)bidi_/bidirectional_rnn/bw/lstm_cell/biassave_21/RestoreV2:2*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ì
save_21/Assign_3Assign.bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adamsave_21/RestoreV2:3*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
î
save_21/Assign_4Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save_21/RestoreV2:4*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/bw/lstm_cell/bias
ð
save_21/Assign_5Assign+bidi_/bidirectional_rnn/bw/lstm_cell/kernelsave_21/RestoreV2:5*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
õ
save_21/Assign_6Assign0bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave_21/RestoreV2:6*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
÷
save_21/Assign_7Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save_21/RestoreV2:7*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/bw/lstm_cell/kernel
ç
save_21/Assign_8Assign)bidi_/bidirectional_rnn/fw/lstm_cell/biassave_21/RestoreV2:8*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ì
save_21/Assign_9Assign.bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adamsave_21/RestoreV2:9*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ð
save_21/Assign_10Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save_21/RestoreV2:10*
T0*
_output_shapes	
: *
use_locking(*
validate_shape(*<
_class2
0.loc:@bidi_/bidirectional_rnn/fw/lstm_cell/bias
ò
save_21/Assign_11Assign+bidi_/bidirectional_rnn/fw/lstm_cell/kernelsave_21/RestoreV2:11*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
÷
save_21/Assign_12Assign0bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave_21/RestoreV2:12*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
ù
save_21/Assign_13Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save_21/RestoreV2:13*
T0* 
_output_shapes
:
Ò *
use_locking(*
validate_shape(*>
_class4
20loc:@bidi_/bidirectional_rnn/fw/lstm_cell/kernel
Ö
save_21/Assign_14Assign conv_maxpool_3_conv_1/B_filter_0save_21/RestoreV2:14*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Û
save_21/Assign_15Assign%conv_maxpool_3_conv_1/B_filter_0/Adamsave_21/RestoreV2:15*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
Ý
save_21/Assign_16Assign'conv_maxpool_3_conv_1/B_filter_0/Adam_1save_21/RestoreV2:16*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/B_filter_0
ã
save_21/Assign_17Assign conv_maxpool_3_conv_1/W_filter_0save_21/RestoreV2:17*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
è
save_21/Assign_18Assign%conv_maxpool_3_conv_1/W_filter_0/Adamsave_21/RestoreV2:18*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
ê
save_21/Assign_19Assign'conv_maxpool_3_conv_1/W_filter_0/Adam_1save_21/RestoreV2:19*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_3_conv_1/W_filter_0
Ö
save_21/Assign_20Assign conv_maxpool_5_conv_1/B_filter_1save_21/RestoreV2:20*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Û
save_21/Assign_21Assign%conv_maxpool_5_conv_1/B_filter_1/Adamsave_21/RestoreV2:21*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
Ý
save_21/Assign_22Assign'conv_maxpool_5_conv_1/B_filter_1/Adam_1save_21/RestoreV2:22*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/B_filter_1
ã
save_21/Assign_23Assign conv_maxpool_5_conv_1/W_filter_1save_21/RestoreV2:23*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
è
save_21/Assign_24Assign%conv_maxpool_5_conv_1/W_filter_1/Adamsave_21/RestoreV2:24*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
ê
save_21/Assign_25Assign'conv_maxpool_5_conv_1/W_filter_1/Adam_1save_21/RestoreV2:25*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_5_conv_1/W_filter_1
Ö
save_21/Assign_26Assign conv_maxpool_7_conv_1/B_filter_2save_21/RestoreV2:26*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Û
save_21/Assign_27Assign%conv_maxpool_7_conv_1/B_filter_2/Adamsave_21/RestoreV2:27*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
Ý
save_21/Assign_28Assign'conv_maxpool_7_conv_1/B_filter_2/Adam_1save_21/RestoreV2:28*
T0*
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/B_filter_2
ã
save_21/Assign_29Assign conv_maxpool_7_conv_1/W_filter_2save_21/RestoreV2:29*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
è
save_21/Assign_30Assign%conv_maxpool_7_conv_1/W_filter_2/Adamsave_21/RestoreV2:30*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
ê
save_21/Assign_31Assign'conv_maxpool_7_conv_1/W_filter_2/Adam_1save_21/RestoreV2:31*
T0*'
_output_shapes
:d*
use_locking(*
validate_shape(*3
_class)
'%loc:@conv_maxpool_7_conv_1/W_filter_2
¦
save_21/Assign_32Assignout_biassave_21/RestoreV2:32*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
«
save_21/Assign_33Assignout_bias/Adamsave_21/RestoreV2:33*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
­
save_21/Assign_34Assignout_bias/Adam_1save_21/RestoreV2:34*
T0*
_output_shapes
:R*
use_locking(*
validate_shape(*
_class
loc:@out_bias
±
save_21/Assign_35Assignout_weightssave_21/RestoreV2:35*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¶
save_21/Assign_36Assignout_weights/Adamsave_21/RestoreV2:36*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights
¸
save_21/Assign_37Assignout_weights/Adam_1save_21/RestoreV2:37*
T0*
_output_shapes
:	ÆR*
use_locking(*
validate_shape(*
_class
loc:@out_weights

save_21/restore_shardNoOp^save_21/Assign^save_21/Assign_1^save_21/Assign_10^save_21/Assign_11^save_21/Assign_12^save_21/Assign_13^save_21/Assign_14^save_21/Assign_15^save_21/Assign_16^save_21/Assign_17^save_21/Assign_18^save_21/Assign_19^save_21/Assign_2^save_21/Assign_20^save_21/Assign_21^save_21/Assign_22^save_21/Assign_23^save_21/Assign_24^save_21/Assign_25^save_21/Assign_26^save_21/Assign_27^save_21/Assign_28^save_21/Assign_29^save_21/Assign_3^save_21/Assign_30^save_21/Assign_31^save_21/Assign_32^save_21/Assign_33^save_21/Assign_34^save_21/Assign_35^save_21/Assign_36^save_21/Assign_37^save_21/Assign_4^save_21/Assign_5^save_21/Assign_6^save_21/Assign_7^save_21/Assign_8^save_21/Assign_9

 save_21/RestoreV2_1/tensor_namesConst*C
value:B8B
embeddingsBembeddings/AdamBembeddings/Adam_1*
dtype0*
_output_shapes
:
q
$save_21/RestoreV2_1/shape_and_slicesConst*
valueBB B B *
dtype0*
_output_shapes
:
¬
save_21/RestoreV2_1	RestoreV2save_21/Const save_21/RestoreV2_1/tensor_names$save_21/RestoreV2_1/shape_and_slices*
dtypes
2* 
_output_shapes
:::
®
save_21/Assign_38Assign
embeddingssave_21/RestoreV2_1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
µ
save_21/Assign_39Assignembeddings/Adamsave_21/RestoreV2_1:1*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
·
save_21/Assign_40Assignembeddings/Adam_1save_21/RestoreV2_1:2*
T0*
_output_shapes
:	7*
use_locking(*
validate_shape(*
_class
loc:@embeddings
[
save_21/restore_shard_1NoOp^save_21/Assign_38^save_21/Assign_39^save_21/Assign_40
8
save_21/restore_all/NoOpNoOp^save_21/restore_shard
<
save_21/restore_all/NoOp_1NoOp^save_21/restore_shard_1
S
save_21/restore_allNoOp^save_21/restore_all/NoOp^save_21/restore_all/NoOp_1 "E
save_21/Const:0save_21/Identity:0save_21/restore_all (5 @F8"ÁÈ
while_context®ÈªÈ
¤
1bidi_/bidirectional_rnn/fw/fw/while/while_context *.bidi_/bidirectional_rnn/fw/fw/while/LoopCond:02+bidi_/bidirectional_rnn/fw/fw/while/Merge:0:.bidi_/bidirectional_rnn/fw/fw/while/Identity:0B*bidi_/bidirectional_rnn/fw/fw/while/Exit:0B,bidi_/bidirectional_rnn/fw/fw/while/Exit_1:0B,bidi_/bidirectional_rnn/fw/fw/while/Exit_2:0B,bidi_/bidirectional_rnn/fw/fw/while/Exit_3:0B,bidi_/bidirectional_rnn/fw/fw/while/Exit_4:0Bgradients/f_count_2:0J
+bidi_/bidirectional_rnn/fw/fw/CheckSeqLen:0
'bidi_/bidirectional_rnn/fw/fw/Minimum:0
+bidi_/bidirectional_rnn/fw/fw/TensorArray:0
Zbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
-bidi_/bidirectional_rnn/fw/fw/TensorArray_1:0
/bidi_/bidirectional_rnn/fw/fw/strided_slice_1:0
+bidi_/bidirectional_rnn/fw/fw/while/Enter:0
-bidi_/bidirectional_rnn/fw/fw/while/Enter_1:0
-bidi_/bidirectional_rnn/fw/fw/while/Enter_2:0
-bidi_/bidirectional_rnn/fw/fw/while/Enter_3:0
-bidi_/bidirectional_rnn/fw/fw/while/Enter_4:0
*bidi_/bidirectional_rnn/fw/fw/while/Exit:0
,bidi_/bidirectional_rnn/fw/fw/while/Exit_1:0
,bidi_/bidirectional_rnn/fw/fw/while/Exit_2:0
,bidi_/bidirectional_rnn/fw/fw/while/Exit_3:0
,bidi_/bidirectional_rnn/fw/fw/while/Exit_4:0
8bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
2bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual:0
.bidi_/bidirectional_rnn/fw/fw/while/Identity:0
0bidi_/bidirectional_rnn/fw/fw/while/Identity_1:0
0bidi_/bidirectional_rnn/fw/fw/while/Identity_2:0
0bidi_/bidirectional_rnn/fw/fw/while/Identity_3:0
0bidi_/bidirectional_rnn/fw/fw/while/Identity_4:0
0bidi_/bidirectional_rnn/fw/fw/while/Less/Enter:0
*bidi_/bidirectional_rnn/fw/fw/while/Less:0
2bidi_/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
,bidi_/bidirectional_rnn/fw/fw/while/Less_1:0
0bidi_/bidirectional_rnn/fw/fw/while/LogicalAnd:0
.bidi_/bidirectional_rnn/fw/fw/while/LoopCond:0
+bidi_/bidirectional_rnn/fw/fw/while/Merge:0
+bidi_/bidirectional_rnn/fw/fw/while/Merge:1
-bidi_/bidirectional_rnn/fw/fw/while/Merge_1:0
-bidi_/bidirectional_rnn/fw/fw/while/Merge_1:1
-bidi_/bidirectional_rnn/fw/fw/while/Merge_2:0
-bidi_/bidirectional_rnn/fw/fw/while/Merge_2:1
-bidi_/bidirectional_rnn/fw/fw/while/Merge_3:0
-bidi_/bidirectional_rnn/fw/fw/while/Merge_3:1
-bidi_/bidirectional_rnn/fw/fw/while/Merge_4:0
-bidi_/bidirectional_rnn/fw/fw/while/Merge_4:1
3bidi_/bidirectional_rnn/fw/fw/while/NextIteration:0
5bidi_/bidirectional_rnn/fw/fw/while/NextIteration_1:0
5bidi_/bidirectional_rnn/fw/fw/while/NextIteration_2:0
5bidi_/bidirectional_rnn/fw/fw/while/NextIteration_3:0
5bidi_/bidirectional_rnn/fw/fw/while/NextIteration_4:0
2bidi_/bidirectional_rnn/fw/fw/while/Select/Enter:0
,bidi_/bidirectional_rnn/fw/fw/while/Select:0
.bidi_/bidirectional_rnn/fw/fw/while/Select_1:0
.bidi_/bidirectional_rnn/fw/fw/while/Select_2:0
,bidi_/bidirectional_rnn/fw/fw/while/Switch:0
,bidi_/bidirectional_rnn/fw/fw/while/Switch:1
.bidi_/bidirectional_rnn/fw/fw/while/Switch_1:0
.bidi_/bidirectional_rnn/fw/fw/while/Switch_1:1
.bidi_/bidirectional_rnn/fw/fw/while/Switch_2:0
.bidi_/bidirectional_rnn/fw/fw/while/Switch_2:1
.bidi_/bidirectional_rnn/fw/fw/while/Switch_3:0
.bidi_/bidirectional_rnn/fw/fw/while/Switch_3:1
.bidi_/bidirectional_rnn/fw/fw/while/Switch_4:0
.bidi_/bidirectional_rnn/fw/fw/while/Switch_4:1
=bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
?bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
7bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
Obidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Ibidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
+bidi_/bidirectional_rnn/fw/fw/while/add/y:0
)bidi_/bidirectional_rnn/fw/fw/while/add:0
-bidi_/bidirectional_rnn/fw/fw/while/add_1/y:0
+bidi_/bidirectional_rnn/fw/fw/while/add_1:0
3bidi_/bidirectional_rnn/fw/fw/while/dropout/Floor:0
3bidi_/bidirectional_rnn/fw/fw/while/dropout/Shape:0
7bidi_/bidirectional_rnn/fw/fw/while/dropout/add/Enter:0
1bidi_/bidirectional_rnn/fw/fw/while/dropout/add:0
1bidi_/bidirectional_rnn/fw/fw/while/dropout/div:0
1bidi_/bidirectional_rnn/fw/fw/while/dropout/mul:0
Jbidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/RandomUniform:0
@bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/max:0
@bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/min:0
@bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/mul:0
@bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform/sub:0
<bidi_/bidirectional_rnn/fw/fw/while/dropout/random_uniform:0
=bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0
7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd:0
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Const:0
<bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0
6bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul:0
7bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid:0
9bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1:0
9bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2:0
4bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh:0
6bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1:0
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add/y:0
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add:0
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1:0
;bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis:0
6bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat:0
3bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul:0
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1:0
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2:0
?bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim:0
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:0
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:1
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:2
5bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/split:3
%bidi_/bidirectional_rnn/fw/fw/zeros:0
0bidi_/bidirectional_rnn/fw/lstm_cell/bias/read:0
2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/read:0
dropout_keep_prob:0
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0
pgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
vgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
pgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2:0
dgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2_1:0
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Enter:0
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/StackPushV2:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc:0
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape:0
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Shape_1:0
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2:0
dgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter:0
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/StackPushV2:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc:0
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter:0
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/StackPushV2:0
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc:0
Fgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape:0
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Shape_1:0
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
Zgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape:0
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape:0
Kgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Shape:0
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0
Xgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Vgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape:0
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Vgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape:0
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
dgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Hgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0 
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0~
+bidi_/bidirectional_rnn/fw/fw/TensorArray:0Obidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0À
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0 
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0¤
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/f_acc:0Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/Neg/Enter:0À
^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0n
-bidi_/bidirectional_rnn/fw/fw/TensorArray_1:0=bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0Ä
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
Zbidi_/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0?bidi_/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0g
+bidi_/bidirectional_rnn/fw/fw/CheckSeqLen:08bidi_/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/f_acc:0Jgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul/Enter:0r
2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/read:0<bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0[
%bidi_/bidirectional_rnn/fw/fw/zeros:02bidi_/bidirectional_rnn/fw/fw/while/Select/Enter:0À
^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0^gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0¤
Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Pgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0¨
Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Rgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0 
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0Ngradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0ä
pgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0pgradients/bidi_/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0¬
Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Tgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0Ä
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0Ä
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0È
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
Jgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0Jgradients/bidi_/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0È
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
Lgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/f_acc:0Lgradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/Mul_1/Enter:0À
^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1:0^gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1:0Ä
`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0`gradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0q
0bidi_/bidirectional_rnn/fw/lstm_cell/bias/read:0=bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0¼
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0]
'bidi_/bidirectional_rnn/fw/fw/Minimum:02bidi_/bidirectional_rnn/fw/fw/while/Less_1/Enter:0 
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0N
dropout_keep_prob:07bidi_/bidirectional_rnn/fw/fw/while/dropout/add/Enter:0c
/bidi_/bidirectional_rnn/fw/fw/strided_slice_1:00bidi_/bidirectional_rnn/fw/fw/while/Less/Enter:0¼
\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/bidi_/bidirectional_rnn/fw/fw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0È
bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0bgradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0 
Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Ngradients/bidi_/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0R+bidi_/bidirectional_rnn/fw/fw/while/Enter:0R-bidi_/bidirectional_rnn/fw/fw/while/Enter_1:0R-bidi_/bidirectional_rnn/fw/fw/while/Enter_2:0R-bidi_/bidirectional_rnn/fw/fw/while/Enter_3:0R-bidi_/bidirectional_rnn/fw/fw/while/Enter_4:0Rgradients/f_count_1:0Z/bidi_/bidirectional_rnn/fw/fw/strided_slice_1:0
¤
1bidi_/bidirectional_rnn/bw/bw/while/while_context *.bidi_/bidirectional_rnn/bw/bw/while/LoopCond:02+bidi_/bidirectional_rnn/bw/bw/while/Merge:0:.bidi_/bidirectional_rnn/bw/bw/while/Identity:0B*bidi_/bidirectional_rnn/bw/bw/while/Exit:0B,bidi_/bidirectional_rnn/bw/bw/while/Exit_1:0B,bidi_/bidirectional_rnn/bw/bw/while/Exit_2:0B,bidi_/bidirectional_rnn/bw/bw/while/Exit_3:0B,bidi_/bidirectional_rnn/bw/bw/while/Exit_4:0Bgradients/f_count_5:0J¥
+bidi_/bidirectional_rnn/bw/bw/CheckSeqLen:0
'bidi_/bidirectional_rnn/bw/bw/Minimum:0
+bidi_/bidirectional_rnn/bw/bw/TensorArray:0
Zbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
-bidi_/bidirectional_rnn/bw/bw/TensorArray_1:0
/bidi_/bidirectional_rnn/bw/bw/strided_slice_1:0
+bidi_/bidirectional_rnn/bw/bw/while/Enter:0
-bidi_/bidirectional_rnn/bw/bw/while/Enter_1:0
-bidi_/bidirectional_rnn/bw/bw/while/Enter_2:0
-bidi_/bidirectional_rnn/bw/bw/while/Enter_3:0
-bidi_/bidirectional_rnn/bw/bw/while/Enter_4:0
*bidi_/bidirectional_rnn/bw/bw/while/Exit:0
,bidi_/bidirectional_rnn/bw/bw/while/Exit_1:0
,bidi_/bidirectional_rnn/bw/bw/while/Exit_2:0
,bidi_/bidirectional_rnn/bw/bw/while/Exit_3:0
,bidi_/bidirectional_rnn/bw/bw/while/Exit_4:0
8bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
2bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual:0
.bidi_/bidirectional_rnn/bw/bw/while/Identity:0
0bidi_/bidirectional_rnn/bw/bw/while/Identity_1:0
0bidi_/bidirectional_rnn/bw/bw/while/Identity_2:0
0bidi_/bidirectional_rnn/bw/bw/while/Identity_3:0
0bidi_/bidirectional_rnn/bw/bw/while/Identity_4:0
0bidi_/bidirectional_rnn/bw/bw/while/Less/Enter:0
*bidi_/bidirectional_rnn/bw/bw/while/Less:0
2bidi_/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
,bidi_/bidirectional_rnn/bw/bw/while/Less_1:0
0bidi_/bidirectional_rnn/bw/bw/while/LogicalAnd:0
.bidi_/bidirectional_rnn/bw/bw/while/LoopCond:0
+bidi_/bidirectional_rnn/bw/bw/while/Merge:0
+bidi_/bidirectional_rnn/bw/bw/while/Merge:1
-bidi_/bidirectional_rnn/bw/bw/while/Merge_1:0
-bidi_/bidirectional_rnn/bw/bw/while/Merge_1:1
-bidi_/bidirectional_rnn/bw/bw/while/Merge_2:0
-bidi_/bidirectional_rnn/bw/bw/while/Merge_2:1
-bidi_/bidirectional_rnn/bw/bw/while/Merge_3:0
-bidi_/bidirectional_rnn/bw/bw/while/Merge_3:1
-bidi_/bidirectional_rnn/bw/bw/while/Merge_4:0
-bidi_/bidirectional_rnn/bw/bw/while/Merge_4:1
3bidi_/bidirectional_rnn/bw/bw/while/NextIteration:0
5bidi_/bidirectional_rnn/bw/bw/while/NextIteration_1:0
5bidi_/bidirectional_rnn/bw/bw/while/NextIteration_2:0
5bidi_/bidirectional_rnn/bw/bw/while/NextIteration_3:0
5bidi_/bidirectional_rnn/bw/bw/while/NextIteration_4:0
2bidi_/bidirectional_rnn/bw/bw/while/Select/Enter:0
,bidi_/bidirectional_rnn/bw/bw/while/Select:0
.bidi_/bidirectional_rnn/bw/bw/while/Select_1:0
.bidi_/bidirectional_rnn/bw/bw/while/Select_2:0
,bidi_/bidirectional_rnn/bw/bw/while/Switch:0
,bidi_/bidirectional_rnn/bw/bw/while/Switch:1
.bidi_/bidirectional_rnn/bw/bw/while/Switch_1:0
.bidi_/bidirectional_rnn/bw/bw/while/Switch_1:1
.bidi_/bidirectional_rnn/bw/bw/while/Switch_2:0
.bidi_/bidirectional_rnn/bw/bw/while/Switch_2:1
.bidi_/bidirectional_rnn/bw/bw/while/Switch_3:0
.bidi_/bidirectional_rnn/bw/bw/while/Switch_3:1
.bidi_/bidirectional_rnn/bw/bw/while/Switch_4:0
.bidi_/bidirectional_rnn/bw/bw/while/Switch_4:1
=bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
?bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
7bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
Obidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Ibidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
+bidi_/bidirectional_rnn/bw/bw/while/add/y:0
)bidi_/bidirectional_rnn/bw/bw/while/add:0
-bidi_/bidirectional_rnn/bw/bw/while/add_1/y:0
+bidi_/bidirectional_rnn/bw/bw/while/add_1:0
3bidi_/bidirectional_rnn/bw/bw/while/dropout/Floor:0
3bidi_/bidirectional_rnn/bw/bw/while/dropout/Shape:0
7bidi_/bidirectional_rnn/bw/bw/while/dropout/add/Enter:0
1bidi_/bidirectional_rnn/bw/bw/while/dropout/add:0
1bidi_/bidirectional_rnn/bw/bw/while/dropout/div:0
1bidi_/bidirectional_rnn/bw/bw/while/dropout/mul:0
Jbidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/RandomUniform:0
@bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/max:0
@bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/min:0
@bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/mul:0
@bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform/sub:0
<bidi_/bidirectional_rnn/bw/bw/while/dropout/random_uniform:0
=bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0
7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd:0
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Const:0
<bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0
6bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul:0
7bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid:0
9bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1:0
9bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2:0
4bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh:0
6bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1:0
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add/y:0
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add:0
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1:0
;bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis:0
6bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat:0
3bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul:0
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1:0
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2:0
?bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim:0
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:0
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:1
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:2
5bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/split:3
%bidi_/bidirectional_rnn/bw/bw/zeros:0
0bidi_/bidirectional_rnn/bw/lstm_cell/bias/read:0
2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/read:0
dropout_keep_prob:0
gradients/Add_1/y:0
gradients/Add_1:0
gradients/Merge_2:0
gradients/Merge_2:1
gradients/NextIteration_2:0
gradients/Switch_2:0
gradients/Switch_2:1
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0
pgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
vgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
pgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2:0
dgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/StackPushV2_1:0
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Enter:0
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/StackPushV2:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc:0
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape:0
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Shape_1:0
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2:0
dgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter:0
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/StackPushV2:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc:0
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter:0
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/StackPushV2:0
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc:0
Fgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape:0
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Shape_1:0
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
Zgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape:0
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape:0
Kgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Shape:0
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0
Xgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Vgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape:0
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
Vgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape:0
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
dgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Hgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count_3:0
gradients/f_count_4:0
gradients/f_count_5:0¬
Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Tgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0Ä
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0g
+bidi_/bidirectional_rnn/bw/bw/CheckSeqLen:08bidi_/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0À
^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc_1:0^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter_1:0¤
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0ä
pgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0pgradients/bidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0n
-bidi_/bidirectional_rnn/bw/bw/TensorArray_1:0=bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0¤
Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Pgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/f_acc:0Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/Neg/Enter:0c
/bidi_/bidirectional_rnn/bw/bw/strided_slice_1:00bidi_/bidirectional_rnn/bw/bw/while/Less/Enter:0¼
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc:0\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter:0 
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0
Lgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/f_acc:0Lgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul_1/Enter:0Ä
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0[
%bidi_/bidirectional_rnn/bw/bw/zeros:02bidi_/bidirectional_rnn/bw/bw/while/Select/Enter:0¨
Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Rgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0 
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0¼
\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/f_acc:0\gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/BroadcastGradientArgs/Enter:0È
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0 
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0Ä
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0Jgradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0 
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0Ngradients/bidi_/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0 
Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Ngradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0À
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0À
^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0^gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0~
+bidi_/bidirectional_rnn/bw/bw/TensorArray:0Obidi_/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0N
dropout_keep_prob:07bidi_/bidirectional_rnn/bw/bw/while/dropout/add/Enter:0È
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
Zbidi_/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0?bidi_/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0À
^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/f_acc_1:0^gradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/div_grad/BroadcastGradientArgs/Enter_1:0
Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/f_acc:0Jgradients/bidi_/bidirectional_rnn/bw/bw/while/dropout/mul_grad/Mul/Enter:0]
'bidi_/bidirectional_rnn/bw/bw/Minimum:02bidi_/bidirectional_rnn/bw/bw/while/Less_1/Enter:0q
0bidi_/bidirectional_rnn/bw/lstm_cell/bias/read:0=bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0Ä
`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0`gradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0r
2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/read:0<bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0È
bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0bgradients/bidi_/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0R+bidi_/bidirectional_rnn/bw/bw/while/Enter:0R-bidi_/bidirectional_rnn/bw/bw/while/Enter_1:0R-bidi_/bidirectional_rnn/bw/bw/while/Enter_2:0R-bidi_/bidirectional_rnn/bw/bw/while/Enter_3:0R-bidi_/bidirectional_rnn/bw/bw/while/Enter_4:0Rgradients/f_count_4:0Z/bidi_/bidirectional_rnn/bw/bw/strided_slice_1:0"Ö
trainable_variables¾»
]
embeddings:0embeddings/Assignembeddings/read:02'embeddings/Initializer/random_uniform:0
á
-bidi_/bidirectional_rnn/fw/lstm_cell/kernel:02bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/read:02Hbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:0
Ð
+bidi_/bidirectional_rnn/fw/lstm_cell/bias:00bidi_/bidirectional_rnn/fw/lstm_cell/bias/Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/read:02=bidi_/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:0
á
-bidi_/bidirectional_rnn/bw/lstm_cell/kernel:02bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/read:02Hbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:0
Ð
+bidi_/bidirectional_rnn/bw/lstm_cell/bias:00bidi_/bidirectional_rnn/bw/lstm_cell/bias/Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/read:02=bidi_/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:0
·
"conv_maxpool_3_conv_1/W_filter_0:0'conv_maxpool_3_conv_1/W_filter_0/Assign'conv_maxpool_3_conv_1/W_filter_0/read:02?conv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal:0
¬
"conv_maxpool_3_conv_1/B_filter_0:0'conv_maxpool_3_conv_1/B_filter_0/Assign'conv_maxpool_3_conv_1/B_filter_0/read:024conv_maxpool_3_conv_1/B_filter_0/Initializer/Const:0
·
"conv_maxpool_5_conv_1/W_filter_1:0'conv_maxpool_5_conv_1/W_filter_1/Assign'conv_maxpool_5_conv_1/W_filter_1/read:02?conv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal:0
¬
"conv_maxpool_5_conv_1/B_filter_1:0'conv_maxpool_5_conv_1/B_filter_1/Assign'conv_maxpool_5_conv_1/B_filter_1/read:024conv_maxpool_5_conv_1/B_filter_1/Initializer/Const:0
·
"conv_maxpool_7_conv_1/W_filter_2:0'conv_maxpool_7_conv_1/W_filter_2/Assign'conv_maxpool_7_conv_1/W_filter_2/read:02?conv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal:0
¬
"conv_maxpool_7_conv_1/B_filter_2:0'conv_maxpool_7_conv_1/B_filter_2/Assign'conv_maxpool_7_conv_1/B_filter_2/read:024conv_maxpool_7_conv_1/B_filter_2/Initializer/Const:0
c
out_weights:0out_weights/Assignout_weights/read:02*out_weights/Initializer/truncated_normal:0
L

out_bias:0out_bias/Assignout_bias/read:02out_bias/Initializer/Const:0"%
train_op

optim_apply_gradients"Ð9
	variablesÂ9¿9
]
embeddings:0embeddings/Assignembeddings/read:02'embeddings/Initializer/random_uniform:0
á
-bidi_/bidirectional_rnn/fw/lstm_cell/kernel:02bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Assign2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/read:02Hbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:0
Ð
+bidi_/bidirectional_rnn/fw/lstm_cell/bias:00bidi_/bidirectional_rnn/fw/lstm_cell/bias/Assign0bidi_/bidirectional_rnn/fw/lstm_cell/bias/read:02=bidi_/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:0
á
-bidi_/bidirectional_rnn/bw/lstm_cell/kernel:02bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Assign2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/read:02Hbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:0
Ð
+bidi_/bidirectional_rnn/bw/lstm_cell/bias:00bidi_/bidirectional_rnn/bw/lstm_cell/bias/Assign0bidi_/bidirectional_rnn/bw/lstm_cell/bias/read:02=bidi_/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:0
·
"conv_maxpool_3_conv_1/W_filter_0:0'conv_maxpool_3_conv_1/W_filter_0/Assign'conv_maxpool_3_conv_1/W_filter_0/read:02?conv_maxpool_3_conv_1/W_filter_0/Initializer/truncated_normal:0
¬
"conv_maxpool_3_conv_1/B_filter_0:0'conv_maxpool_3_conv_1/B_filter_0/Assign'conv_maxpool_3_conv_1/B_filter_0/read:024conv_maxpool_3_conv_1/B_filter_0/Initializer/Const:0
·
"conv_maxpool_5_conv_1/W_filter_1:0'conv_maxpool_5_conv_1/W_filter_1/Assign'conv_maxpool_5_conv_1/W_filter_1/read:02?conv_maxpool_5_conv_1/W_filter_1/Initializer/truncated_normal:0
¬
"conv_maxpool_5_conv_1/B_filter_1:0'conv_maxpool_5_conv_1/B_filter_1/Assign'conv_maxpool_5_conv_1/B_filter_1/read:024conv_maxpool_5_conv_1/B_filter_1/Initializer/Const:0
·
"conv_maxpool_7_conv_1/W_filter_2:0'conv_maxpool_7_conv_1/W_filter_2/Assign'conv_maxpool_7_conv_1/W_filter_2/read:02?conv_maxpool_7_conv_1/W_filter_2/Initializer/truncated_normal:0
¬
"conv_maxpool_7_conv_1/B_filter_2:0'conv_maxpool_7_conv_1/B_filter_2/Assign'conv_maxpool_7_conv_1/B_filter_2/read:024conv_maxpool_7_conv_1/B_filter_2/Initializer/Const:0
c
out_weights:0out_weights/Assignout_weights/read:02*out_weights/Initializer/truncated_normal:0
L

out_bias:0out_bias/Assignout_bias/read:02out_bias/Initializer/Const:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
h
embeddings/Adam:0embeddings/Adam/Assignembeddings/Adam/read:02#embeddings/Adam/Initializer/zeros:0
p
embeddings/Adam_1:0embeddings/Adam_1/Assignembeddings/Adam_1/read:02%embeddings/Adam_1/Initializer/zeros:0
ì
2bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam:07bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Assign7bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/read:02Dbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros:0
ô
4bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1:09bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Assign9bidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/read:02Fbidi_/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ä
0bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam:05bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign5bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/read:02Bbidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros:0
ì
2bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1:07bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign7bidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/read:02Dbidi_/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros:0
ì
2bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam:07bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Assign7bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/read:02Dbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros:0
ô
4bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1:09bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Assign9bidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/read:02Fbidi_/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
ä
0bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam:05bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign5bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/read:02Bbidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros:0
ì
2bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1:07bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign7bidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/read:02Dbidi_/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros:0
À
'conv_maxpool_3_conv_1/W_filter_0/Adam:0,conv_maxpool_3_conv_1/W_filter_0/Adam/Assign,conv_maxpool_3_conv_1/W_filter_0/Adam/read:029conv_maxpool_3_conv_1/W_filter_0/Adam/Initializer/zeros:0
È
)conv_maxpool_3_conv_1/W_filter_0/Adam_1:0.conv_maxpool_3_conv_1/W_filter_0/Adam_1/Assign.conv_maxpool_3_conv_1/W_filter_0/Adam_1/read:02;conv_maxpool_3_conv_1/W_filter_0/Adam_1/Initializer/zeros:0
À
'conv_maxpool_3_conv_1/B_filter_0/Adam:0,conv_maxpool_3_conv_1/B_filter_0/Adam/Assign,conv_maxpool_3_conv_1/B_filter_0/Adam/read:029conv_maxpool_3_conv_1/B_filter_0/Adam/Initializer/zeros:0
È
)conv_maxpool_3_conv_1/B_filter_0/Adam_1:0.conv_maxpool_3_conv_1/B_filter_0/Adam_1/Assign.conv_maxpool_3_conv_1/B_filter_0/Adam_1/read:02;conv_maxpool_3_conv_1/B_filter_0/Adam_1/Initializer/zeros:0
À
'conv_maxpool_5_conv_1/W_filter_1/Adam:0,conv_maxpool_5_conv_1/W_filter_1/Adam/Assign,conv_maxpool_5_conv_1/W_filter_1/Adam/read:029conv_maxpool_5_conv_1/W_filter_1/Adam/Initializer/zeros:0
È
)conv_maxpool_5_conv_1/W_filter_1/Adam_1:0.conv_maxpool_5_conv_1/W_filter_1/Adam_1/Assign.conv_maxpool_5_conv_1/W_filter_1/Adam_1/read:02;conv_maxpool_5_conv_1/W_filter_1/Adam_1/Initializer/zeros:0
À
'conv_maxpool_5_conv_1/B_filter_1/Adam:0,conv_maxpool_5_conv_1/B_filter_1/Adam/Assign,conv_maxpool_5_conv_1/B_filter_1/Adam/read:029conv_maxpool_5_conv_1/B_filter_1/Adam/Initializer/zeros:0
È
)conv_maxpool_5_conv_1/B_filter_1/Adam_1:0.conv_maxpool_5_conv_1/B_filter_1/Adam_1/Assign.conv_maxpool_5_conv_1/B_filter_1/Adam_1/read:02;conv_maxpool_5_conv_1/B_filter_1/Adam_1/Initializer/zeros:0
À
'conv_maxpool_7_conv_1/W_filter_2/Adam:0,conv_maxpool_7_conv_1/W_filter_2/Adam/Assign,conv_maxpool_7_conv_1/W_filter_2/Adam/read:029conv_maxpool_7_conv_1/W_filter_2/Adam/Initializer/zeros:0
È
)conv_maxpool_7_conv_1/W_filter_2/Adam_1:0.conv_maxpool_7_conv_1/W_filter_2/Adam_1/Assign.conv_maxpool_7_conv_1/W_filter_2/Adam_1/read:02;conv_maxpool_7_conv_1/W_filter_2/Adam_1/Initializer/zeros:0
À
'conv_maxpool_7_conv_1/B_filter_2/Adam:0,conv_maxpool_7_conv_1/B_filter_2/Adam/Assign,conv_maxpool_7_conv_1/B_filter_2/Adam/read:029conv_maxpool_7_conv_1/B_filter_2/Adam/Initializer/zeros:0
È
)conv_maxpool_7_conv_1/B_filter_2/Adam_1:0.conv_maxpool_7_conv_1/B_filter_2/Adam_1/Assign.conv_maxpool_7_conv_1/B_filter_2/Adam_1/read:02;conv_maxpool_7_conv_1/B_filter_2/Adam_1/Initializer/zeros:0
l
out_weights/Adam:0out_weights/Adam/Assignout_weights/Adam/read:02$out_weights/Adam/Initializer/zeros:0
t
out_weights/Adam_1:0out_weights/Adam_1/Assignout_weights/Adam_1/read:02&out_weights/Adam_1/Initializer/zeros:0
`
out_bias/Adam:0out_bias/Adam/Assignout_bias/Adam/read:02!out_bias/Adam/Initializer/zeros:0
h
out_bias/Adam_1:0out_bias/Adam_1/Assignout_bias/Adam_1/read:02#out_bias/Adam_1/Initializer/zeros:0"#
	summaries


accuracy:0
cost:0*
serving_default
&
dropout
dropout_keep_prob:0
'
seqlens
	seqlens:0ÿÿÿÿÿÿÿÿÿ
5
	splitcnts(
split_cnts:0ÿÿÿÿÿÿÿÿÿ

%
x 
inputs:0	ÿÿÿÿÿÿÿÿÿ4
predictions%
predictions:0	ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict