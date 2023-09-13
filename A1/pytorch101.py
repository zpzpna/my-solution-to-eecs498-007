import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch101.py!')


def create_sample_tensor() -> Tensor:
    """
    Return a torch Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.

    Returns:
        Tensor of shape (3, 2) as described above.
    """
    x = None
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    x = torch.tensor([[0,10],[100,0],[0,0]])
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    """
    Mutate the tensor x according to indices and values. Specifically, indices
    is a list [(i0, j0), (i1, j1), ... ] of integer indices, and values is a
    list [v0, v1, ...] of values. This function should mutate x by setting:

    x[i0, j0] = v0
    x[i1, j1] = v1

    and so on.

    If the same index pair appears multiple times in indices, you should set x
    to the last one.

    Args:
        x: A Tensor of shape (H, W)
        indices: A list of N tuples [(i0, j0), (i1, j1), ..., ]
        values: A list of N values [v0, v1, ...]

    Returns:
        The input tensor x
    """
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    
    #经验:unpack多个列表的方法，先把列表zip组合为一个列表，再一组组提取
    for (i,j),v in zip(indices,values):
      x[(i,j)] = v
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    Count the number of scalar elements in a tensor x.

    For example, a tensor of shape (10,) has 10 elements; a tensor of shape
    (3, 4) has 12 elements; a tensor of shape (2, 3, 4) has 24 elements, etc.

    You may not use the functions torch.numel or x.numel. The input tensor
    should not be modified.

    Args:
        x: A tensor of any shape

    Returns:
        num_elements: An integer giving the number of scalar elements in x
    """
    num_elements = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    #   You CANNOT use the built-in functions torch.numel(x) or x.numel().   #
    ##########################################################################
    # Replace "pass" statement with your code
    
    #经验：torch.size类型可以遍历
    num_elements = 1
    size = x.shape
    for length in size:
      num_elements = num_elements*length
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    """
    Returns a Tensor of shape (M, N) filled entirely with the value 3.14

    Args:
        M, N: Positive integers giving the shape of Tensor to create

    Returns:
        x: A tensor of shape (M, N) filled with the value 3.14
    """
    x = None
    ##########################################################################
    #         TODO: Implement this function. It should take one line.        #
    ##########################################################################
    # Replace "pass" statement with your code
    x = torch.full((M,N),3.14)

    #经验：除了ones,zeros,rand基本三种，还有full和arange两种构造器

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    """
    Returns a Tensor of dtype torch.float64 that contains all of the multiples
    of ten (in order) between start and stop, inclusive. If there are no
    multiples of ten in this range then return an empty tensor of shape (0,).

    Args:
        start: Beginning ot range to create.
        stop: End of range to create (stop >= start).

    Returns:
        x: float64 Tensor giving multiples of ten between start and stop
    """
    assert start <= stop
    x = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    x = torch.tensor([ item for item in torch.arange(start,stop) if item %10 == 0],dtype=torch.float64)

    #经验：数据类型，一般用在构造器dtype，.to（）变类型，.new_zeros()保持类型变尺寸，.zeros_like（类型尺寸不变）三种

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given a two-dimensional tensor x, extract and return several subtensors to
    practice with slice indexing. Each tensor should be created using a single
    slice indexing operation.

    The input tensor should not be modified.

    Args:
        x: Tensor of shape (M, N) -- M rows, N columns with M >= 3 and N >= 5.

    Returns:
        A tuple of:
        - last_row: Tensor of shape (N,) giving the last row of x. It should be
          a one-dimensional tensor.
        - third_col: Tensor of shape (M, 1) giving the third column of x. It
          should be a two-dimensional tensor.
        - first_two_rows_three_cols: Tensor of shape (2, 3) giving the data in
          the first two rows and first three columns of x.
        - even_rows_odd_cols: Two-dimensional tensor containing the elements in
          the even-valued rows and odd-valued columns of x.
    """
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    last_row = None
    third_col = None
    first_two_rows_three_cols = None
    even_rows_odd_cols = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    last_row = x[-1,:]
    third_col = x[:,2:3]
    first_two_rows_three_cols = x[:2,:3]
    even_rows_odd_cols = x[0::2,1::2]

    #经验：两种切片[1,:]和[1:2,4]，第一种变成一维，第二种保持维度不退化向量
    #torch的切片还是原来的位置，想要改造必须用.clone()

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    """
    Given a two-dimensional tensor of shape (M, N) with M >= 4, N >= 6, mutate
    its first 4 rows and 6 columns so they are equal to:

    [0 1 2 2 2 2]
    [0 1 2 2 2 2]
    [3 4 3 4 5 5]
    [3 4 3 4 5 5]

    Note: the input tensor shape is not fixed to (4, 6).

    Your implementation must obey the following:
    - You should mutate the tensor x in-place and return it
    - You should only modify the first 4 rows and first 6 columns; all other
      elements should remain unchanged
    - You may only mutate the tensor using slice assignment operations, where
      you assign an integer to a slice of the tensor
    - You must use <= 6 slicing operations to achieve the desired result

    Args:
        x: A tensor of shape (M, N) with M >= 4 and N >= 6

    Returns:
        x
    """
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    x[0:2,0:2] = torch.tensor([[0,1],[0,1]])
    x[0:2,2:6] = torch.tensor([[2,2,2,2],[2,2,2,2]])
    x[2:4,0:4] = torch.tensor([[3,4,3,4],[3,4,3,4]])
    x[2:4,4:6] = torch.tensor([[5,5],[5,5]])
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x

    #interger arrays index
    #经验：一般用于，造新张量，选行列造向量（对角），行列换顺序
    #对于行列操作，一般其中一个用arange构造，另一个选数据
def shuffle_cols(x: Tensor) -> Tensor:
    """
    Re-order the columns of an input tensor as described below.

    Your implementation should construct the output tensor using a single
    integer array indexing operation. The input tensor should not be modified.

    Args:
        x: A tensor of shape (M, N) with N >= 3

    Returns:
        A tensor y of shape (M, 4) where:
        - The first two columns of y are copies of the first column of x
        - The third column of y is the same as the third column of x
        - The fourth column of y is the same as the second column of x
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    idx = [0,0,2,1]
    y = x[:,idx]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def reverse_rows(x: Tensor) -> Tensor:
    """
    Reverse the rows of the input tensor.

    Your implementation should construct the output tensor using a single
    integer array indexing operation. The input tensor should not be modified.

    Your implementation may not use torch.flip.

    Args:
        x: A tensor of shape (M, N)

    Returns:
        y: Tensor of shape (M, N) which is the same as x but with the rows
            reversed - the first row of y should be equal to the last row of x,
            the second row of y should be equal to the second to last row of x,
            and so on.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code

    #经验：如何借助range和列表解析式构造逆序数组
    idx = [x.shape[0]-i-1 for i in range(x.shape[0])]
    y = x[idx,:]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def take_one_elem_per_col(x: Tensor) -> Tensor:
    """
    Construct a new tensor by picking out one element from each column of the
    input tensor as described below.

    The input tensor should not be modified, and you should only access the
    data of the input tensor using a single indexing operation.

    Args:
        x: A tensor of shape (M, N) with M >= 4 and N >= 3.

    Returns:
        A tensor y of shape (3,) such that:
        - The first element of y is the second element of the first column of x
        - The second element of y is the first element of the second column of x
        - The third element of y is the fourth element of the third column of x
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    idx0 = torch.tensor([1,0,3])
    idx1 = torch.arange(3)

    y = x[idx0,idx1]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y

#要学会用列表造独热编码，很重要！！经典应用
#x[n] = c then y[n, c] = 1
#上面是换一个元素的重要等式，应用integer array index可以向量化改变
def make_one_hot(x: List[int]) -> Tensor:
    """
    Construct a tensor of one-hot-vectors from a list of Python integers.

    Your implementation should not use any Python loops (including
    comprehensions).

    Args:
        x: A list of N integers

    Returns:
        y: Tensor of shape (N, C) and where C = 1 + max(x) is one more than the
            max value in x. The nth row of y is a one-hot-vector representation
            of x[n]; in other words, if x[n] = c then y[n, c] = 1; all other
            elements of y are zeros. The dtype of y should be torch.float32.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    y = torch.zeros((len(x),1+max(x)),dtype = torch.float32)
    idx0,idx1 = torch.arange(len(x)),x
    y[idx0,idx1] = torch.tensor(x,dtype=torch.float32)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y

  #bool tensor index
  #用来对tensor中的部分满足条件的数值进行筛选并索引成新tensor
def sum_positive_entries(x: Tensor) -> Tensor:
    """
    Return the sum of all the positive values in the input tensor x.

    For example, given the input tensor

    x = [[ -1   2   0 ],
         [  0   5  -3 ],
         [  8  -9   0 ]]

    This function should return sum_positive_entries(x) = 2 + 5 + 8 = 15

    Your output should be a Python integer, *not* a PyTorch scalar.

    Your implementation should not modify the input tensor, and should not use
    any explicit Python loops (including comprehensions). You should access
    the data of the input tensor using only a single comparison operation and a
    single indexing operation.

    Args:
        x: A tensor of any shape with dtype torch.int64

    Returns:
        pos_sum: Python integer giving the sum of all positive values in x
    """
    pos_sum = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    sum_mask = x>0
    pos_sum = sum(x[sum_mask])
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return pos_sum


def reshape_practice(x: Tensor) -> Tensor:
    """
    Given an input tensor of shape (24,), return a reshaped tensor y of shape
    (3, 8) such that

    y = [[x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
         [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
         [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]]]

    You must construct y by performing a sequence of reshaping operations on
    x (view, t, transpose, permute, contiguous, reshape, etc). The input
    tensor should not be modified.

    Args:
        x: A tensor of shape (24,)

    Returns:
        y: A reshaped version of x of shape (3, 8) as described above.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code

    #先寻找在一起的块，然后想办法用交换重新组合
    #（a,b,c）：换a，b--将第一块所有行分配到新的每个块的第一行，以此类推

    ##新鲜经验：使用维度技巧，分x，y，z...方向来确定交换的值位置
    #简单讲：x：列向下方向，y：行向右方向，z竖直向下
    #注意dim和shape是维度对应的
    y = x.reshape(2,3,4).transpose(0,1).reshape(3,8)

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def zero_row_min(x: Tensor) -> Tensor:
    """
    Return a copy of the input tensor x, where the minimum value along each row
    has been set to 0.

    For example, if x is:
    x = torch.tensor([
          [10, 20, 30],
          [ 2,  5,  1]])

    Then y = zero_row_min(x) should be:
    torch.tensor([
        [0, 20, 30],
        [2,  5,  0]
    ])

    Your implementation shoud use reduction and indexing operations. You should
    not use any Python loops (including comprehensions). The input tensor
    should not be modified.

    Args:
        x: Tensor of shape (M, N)

    Returns:
        y: Tensor of shape (M, N) that is a copy of x, except the minimum value
            along each row is replaced with 0.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    col_idx = x.argmin(dim=1)
    row_idx = torch.arange(x.shape[-2])
    y = x.clone()
    y[row_idx,col_idx] = 0

    ###经验：shape属性返回（x，x，..，行，列）
    ###经验：dim=0,1...的意义随着张量维度的变化而变化，dim=0永远是最外层维度
    ###经验：dim=-1，-2永远是（行）和（列）方向维度
    ###经验：！记得x，y，z轴堆叠张量，y轴为-1维度！！！
    ###经验：minarg如果操作行，本质返回索引是对应的列

    ###!!!!shape的数值，是对应维度方向有几个向量，方向！！如
    #shape[-1]是行有几个元素（对应维度元素个数）


    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def batched_matrix_multiply(
    x: Tensor, y: Tensor, use_loop: bool = True
) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    Depending on the value of use_loop, this calls to either
    batched_matrix_multiply_loop or batched_matrix_multiply_noloop to perform
    the actual computation. You don't need to implement anything here.

    Args:
        x: Tensor of shape (B, N, M)
        y: Tensor of shape (B, M, P)
        use_loop: Whether to use an explicit Python loop.

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)


def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    This implementation should use a single explicit loop over the batch
    dimension B to compute the output.

    Args:
        x: Tensor of shaper (B, N, M)
        y: Tensor of shape (B, M, P)

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    # Replace "pass" statement with your code

    #经验：shape和dim是相同的
    #z = x.new_zeros([x.shape[0],x.shape[1],y.shape[2]])
    #注释的是不用列表和stack堆叠，直接for不断索引计算得到每个结果矩阵
    #经验：对于batch数据，先计算出所有低维结果后stack成更高维度
    z = []
    for i in range(x.shape[0]):
      z.append(torch.mm(x[i],y[i]))
    z = torch.stack(z,dim=0)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return z


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    This implementation should use no explicit Python loops (including
    comprehensions).

    Hint: torch.bmm

    Args:
        x: Tensor of shaper (B, N, M)
        y: Tensor of shape (B, M, P)

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    # Replace "pass" statement with your code
    z = torch.bmm(x,y)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return z


def normalize_columns(x: Tensor) -> Tensor:
    """
    Normalize the columns of the matrix x by subtracting the mean and dividing
    by standard deviation of each column. You should return a new tensor; the
    input should not be modified.

    More concretely, given an input tensor x of shape (M, N), produce an output
    tensor y of shape (M, N) where y[i, j] = (x[i, j] - mu_j) / sigma_j, where
    mu_j is the mean of the column x[:, j].

    Your implementation should not use any explicit Python loops (including
    comprehensions); you may only use basic arithmetic operations on tensors
    (+, -, *, /, **, sqrt), the sum reduction function, and reshape operations
    to facilitate broadcasting. You should not use torch.mean, torch.std, or
    their instance method variants x.mean, x.std.

    Args:
        x: Tensor of shape (M, N).

    Returns:
        y: Tensor of shape (M, N) as described above. It should have the same
            dtype as the input x.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    #经验：！！向量本质上矩阵乘法没有行列之分，但是广播时默认为一行，列向广播
    #经验：广播最好设置好维度，不要降维

    mean_x = torch.sum(x,dim=-2,keepdim=True)/x.shape[-1]
    std_x = torch.sqrt(torch.sum((x-mean_x)**2,dim=-2,keepdim=True)/(x.shape[-1]-1))
    y = (x-mean_x)/std_x
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on CPU.

    You don't need to implement anything for this function.

    Args:
        x: Tensor of shape (A, B), on CPU
        w: Tensor of shape (B, C), on CPU

    Returns:
        y: Tensor of shape (A, C) as described above. It should not be in GPU.
    """
    y = x.mm(w)
    return y


def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on GPU.

    Specifically, given two input tensors this function should:
    (1) move each input tensor to the GPU;
    (2) perform matrix multiplication between the GPU tensors;
    (3) move the result back to CPU

    When you move the tensor to GPU, use the "your_tensor.cuda()" operation

    Args:
        x: Tensor of shape (A, B), on CPU
        w: Tensor of shape (B, C), on CPU

    Returns:
        y: Tensor of shape (A, C) as described above. It should not be in GPU.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code

    #经验：！！转换gup后，产生一个在gpu的新的张量，原来的变量名还保存cpu张量
    x_gpu = x.cuda()
    w_gpu = w.cuda()
    y_gpu = torch.mm(x_gpu,w_gpu)
    y = y_gpu.cpu()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_mean_tensors(xs: List[Tensor], ls: Tensor) -> Tensor:
    """
    Compute mean of each tensor in a given list of tensors.

    Specifically, the input is a list of N tensors, (1 <= N <= 10000). The i-th
    tensor in this list has length Ki, (1 <= Ki <= 10000). Return a tensor of
    shape (N, ) whose i-th element gives the mean of i-th tensor in input list.
    You may assume that all tensors are on the same device (CPU or GPU).

    Your implementation should not use any explicit Python loops (including
    comprehensions).

    Args:
        xs: List of N 1-dimensional tensors.
        ls: Length of tensors in `xs`. An int64 Tensor of same length as `xs`
            with `ls[i]` giving the length of `xs[i]`.

    Returns:
        y: Tensor of shape (N, ) with `y[i]` giving the mean of `xs[i]`.
    """

    y = None
    ##########################################################################
    # TODO: Implement this function without using `for` loops and store the  #
    # mean values as a tensor in `y`.                                        #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_get_uniques(x: torch.Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get unique values and first occurrence from an input tensor.

    Specifically, the input is 1-dimensional int64 Tensor with length N. This
    tensor contains K unique values (not necessarily consecutive). Your
    implementation must return two tensors:
    1. uniques: int64 Tensor of shape (K, ) - giving K uniques from input.
    2. indices: int64 Tensor of shape (K, ) - giving indices of the first
       occurring unique values.

    Concretely, this should hold True: x[indices[i]] = uniques[i] 

    Your implementation should not use any explicit Python loops (including
    comprehensions), and should not require more than O(N) memory. Creating
    new tensors larger than input tensor is not allowed. If you wish to
    create new tensors like input tensor, use `input.clone()`.

    You may use `torch.unique`, but you will receive half credit for that.

    Args:
        x: Tensor of shape (N, ) with K <= N unique values.

    Returns:
        uniques and indices: Se description above.
    """

    uniques, indices = None, None
    ##########################################################################
    # TODO: Implement this function without using `for` loops and within     #
    # O(N) memory.                                                           #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return uniques, indices
