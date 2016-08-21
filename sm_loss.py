import sys
import numpy
import scipy.optimize

def sm_loss(params, data):
    data_size, dim = data.shape
    mean = params[:dim][numpy.newaxis]
    cov_inv = params[dim:].reshape((dim, dim))
    data_nomean = data - numpy.matmul(numpy.ones((data_size, 1)), mean)
    dc_m = numpy.matmul(data_nomean, cov_inv.T)
    loss = numpy.sum(dc_m * dc_m) / 2 / data_size - numpy.trace(cov_inv)
    mean_dev = numpy.matmul(numpy.matmul(cov_inv.T, cov_inv),\
        (mean - numpy.sum(data, 0) / data_size).T)
    
    #covi_dev = -numpy.eye(dim, dim)
    covi_dev = numpy.zeros((dim, dim))
    for i in xrange(data_size):
        covi_dev += numpy.matmul(cov_inv, numpy.matmul(data_nomean[i, :][numpy.newaxis].T, data_nomean[i, :][numpy.newaxis])) \
            + numpy.matmul(numpy.matmul(data_nomean[i, :][numpy.newaxis].T, data_nomean[i, :][numpy.newaxis]), cov_inv)
    covi_dev /= data_size * 2.0
    covi_dev -= numpy.eye(dim, dim)
    grad = numpy.zeros(params.size)
    grad[:dim] = mean_dev.reshape((dim,))
    grad[dim:] = covi_dev.reshape((dim * dim, ))
    return loss, grad

def check_dev(dim=5, data_size=100):
    # will not working
    data = numpy.random.rand(data_size, dim)
    mean = numpy.random.rand(1, dim)
    cov_inv = numpy.random.rand(dim, dim)
    cov_inv = cov_inv + cov_inv.T
    params = numpy.zeros((dim + dim * dim,))
    params[:dim] = mean.reshape((dim, ))
    params[dim:] = cov_inv.reshape((dim * dim, ))
    loss, grad = sm_loss(params, data)
    delta = 1e-8
    grad_numeric = numpy.zeros((params.size,))
    for i in range(params.size):
        p_p = params.copy()
        p_p[i] += delta
        p_n = params.copy()
        p_n[i] -= delta
        loss_p, _ = sm_loss(p_p, data)
        loss_n, _ = sm_loss(p_n, data)
        grad_numeric[i] = (loss_p - loss_n) / 2 / delta
    print numpy.linalg.norm(grad - grad_numeric)

def test_sm():
    #check_dev()
    dim = 5
    data_size = 100000
    mean = numpy.random.rand(dim)
    cov = numpy.random.rand(dim, dim)
    while numpy.linalg.det(cov) < 0.7:
        cov = numpy.random.rand(dim, dim)
    cov = numpy.matmul(cov, cov.T)
    cov_inv = numpy.linalg.inv(cov)
    params_true = numpy.zeros(dim + dim * dim)
    params_true[:dim] = mean
    params_true[dim:] = cov_inv.reshape((dim * dim, ))
    
    #print mean, cov, cov_inv
    data = numpy.random.multivariate_normal(mean, cov, data_size)
    print data.shape
    
    # the initial guess
    params = numpy.zeros(dim + dim * dim)
    mean_init = numpy.random.rand(dim)
    cov_init = numpy.random.rand(dim, dim)
    while numpy.linalg.det(cov_init) < 0.7:
        cov_init = numpy.random.rand(dim, dim)
    cov_init = numpy.matmul(cov_init, cov_init.T)
    cov_init_inv = numpy.linalg.inv(cov_init)
    params[:dim] = mean_init
    params[dim:] = cov_init_inv.reshape((dim * dim, ))
    
    opt = {'maxiter': 2000, 'disp': True, 'xtol': 1e-8, 'maxcor': 10, 'factr': 20, \
        'ftol': 1e-12, }
    out_res = scipy.optimize.minimize(sm_loss, params, args=(data),\
        method='L-BFGS-B', jac=True, options=opt)
    params_res = out_res.x
    mean_res = params_res[:dim]
    cov_inv_res = params_res[dim:]
    print mean
    print mean_res
    print numpy.log10(numpy.linalg.norm(params_res - params_true)**2)
    
def main():
    test_sm()

if __name__ == '__main__':
    main()

