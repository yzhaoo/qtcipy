import TensorCrossInterpolation as TCI
using LinearAlgebra
import TensorCrossInterpolation: nrows, ncols, addpivot!, MatrixCI, evaluate


function diagonal_inverse(row,col,data,shape)

	ci = MatrixCI(Complex,shape[1],shape[2])
#	ci = MatrixCI(row,col,data)
#            rowindices, colindices,
#            A[:, colindices], A[rowindices, :]
#        )
        return ci
end

row = [1,2,3]
col = [1,2,3]
data = [1.,1.,1.]
shape = [3,3]
diagonal_inverse(row,col,data,shape)




