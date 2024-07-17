import QuanticsGrids as QG
using QuanticsTCI
using QuanticsTCI: quanticscrossinterpolate

function initialize_interpolator_1d(f, xmin, xmax,nb,tol)
    R = nb # number of bits
    N = 2^R # size of the grid
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=false)
    ci, ranks, errors = quanticscrossinterpolate(Float64, f, qgrid;
    tolerance=tol,
    normalizeerror=true, # Normalize the error by the maximum sample value,
    loginterval=1) # Log the error every `loginterval` iterations)
    return ci, ranks, errors, qgrid
end

# this is commented until the parallelization is done

#
#function initialize_interpolator_1d_batch(fbatch, xmin, xmax,nb,tol; batch_size=6)
#    # input of fbatch is [x1,x2,..,x_batch], and returns
#    # fbatch([x1,x2,..,x_batch]) = [f(x1),...,f(x_batch)]
#    # where f is the original function that takes one point and returns
#    # a single point
#    R = nb # number of bits
#    N = 2^R # size of the grid
#    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=false)
#    ci, ranks, errors = quanticscrossinterpolate_batch(Float64, fbatch, qgrid;
#    batch_size = 10, # how many points at the same time does fbatch return
#    tolerance=tol,
#    normalizeerror=true, # Normalize the error by the maximum sample value,
#    loginterval=1) # Log the error every `loginterval` iterations)
#    return ci, ranks, errors, qgrid
#end
#



function initialize_interpolator_2d(f, xmin, xmax,ymin,ymax,nb,tol)
    R = nb # number of bits
    N = 2^R # size of the grid
    qgrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax))
    ci, ranks, errors = quanticscrossinterpolate(Float64, f, qgrid;
    tolerance=tol,
    normalizeerror=true, # Normalize the error by the maximum sample value,
    loginterval=1) # Log the error every `loginterval` iterations)
    return ci, ranks, errors, qgrid
end





function initialize_interpolator_3d(f, xmin, xmax,ymin,ymax,zmin,zmax,nb,tol)
    R = nb # number of bits
    N = 2^R # size of the grid
    qgrid = QG.DiscretizedGrid{3}(R, (xmin, ymin, zmin), (xmax, ymax, zmax))
    ci, ranks, errors = quanticscrossinterpolate(Float64, f, qgrid;
    tolerance=tol,
    normalizeerror=true, # Normalize the error by the maximum sample value,
    loginterval=1) # Log the error every `loginterval` iterations)
    return ci, ranks, errors, qgrid
end








function call_qtci_1d(xs,qgrid,ci)
    yci = map(xs) do x
        # Convert a coordinate in the original coordinate system to the corresponding grid index
        i = QG.origcoord_to_grididx(qgrid, x)
        ci(i)
    end
end


function call_qtci_2d(x,y,qgrid,ci)
     # Convert a coordinate in the original coordinate system to the corresponding grid index
     out = ci(QG.origcoord_to_grididx(qgrid, (x, y)))
     return out
end


function call_qtci_3d(x,y,z,qgrid,ci)
     # Convert a coordinate in the original coordinate system to the corresponding grid index
     out = ci(QG.origcoord_to_grididx(qgrid, (x, y, z)))
     return out
end




function integrate_qtci_1d(ci,xmin,xmax,R)
    QuanticsTCI.sum(ci) * (xmax - xmin) / 2^R
end



function integrate_qtci_2d(ci,xmin,xmax,ymin,ymax,R)
    QuanticsTCI.sum(ci) * (xmax - xmin) / 2^R * (ymax - ymin) / 2^R
end


function integrate_qtci_3d(ci,xmin,xmax,ymin,ymax,zmin,zmax,R)
    QuanticsTCI.sum(ci) * (xmax - xmin) / 2^R * (ymax - ymin) / 2^R * (zmax - zmin) / 2^R
end





function evaluated_points(ci)
    evaluated = QuanticsTCI.cachedata(ci)
    xs_evaluated = collect(keys(evaluated))
    fs_evaluated = [evaluated[x] for x in xs_evaluated]
    return xs_evaluated,fs_evaluated
end
