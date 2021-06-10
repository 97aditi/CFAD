 # from __future__ import print_function, division

import time
import numpy as np
from pymanopt.solvers.solver import Solver
# from pymanopt.solvers.linesearch_wolfe import linesearch_wolfe
from pymanopt.solvers.linesearch import LineSearchBackTracking


if not hasattr(__builtins__, "xrange"):
    xrange = range


class LBFGS(Solver):

    def __init__(self, numgrad=20, *args, **kwargs):
        super(LBFGS, self).__init__(*args, **kwargs)
        self.numgrad = numgrad  
        self.linesearch = LineSearchBackTracking()


    def solve(self, problem, x=None):
        fxlist = []

        man = problem.manifold
        verbosity = problem.verbosity

        cost = problem.cost
        grad = problem.grad
        hess = problem.hess

        if x is None:
            x = man.rand()

    
         # Initialize solution and companion measures: f(x), fgrad(x)
        fx = cost(x)
        fgradx = grad(x)
        norm_grad = man.norm(x, fgradx)

        lsmem = {};

        if verbosity >= 2:
            print(" iter\t\t   cost val\t    grad. norm")

        iter = 0
        desc_dir = -1/norm_grad * fgradx

        #Initialize the Hessian
        H = 1
        stepsize = np.infty
        x_all = [x]
        desc_dir_all = []
        grad_diff_all = []
        ddgd_all = []
        gd_all = []
        # Expc_all = []
        # Expci_all = []

        # # ** Display:
        # if verbosity >= 1:
        #     print("Optimizing...")
        # if verbosity >= 2:
        #     print("{:44s}f: {:+.6e}   |grad|: {:.6e}".format(
        #         " ", float(fx), norm_grad))
        costevals = 1

        
        # self._start_optlog()

        #Start iterating until stopping criterion triggers

        while True:
            # Initializations
            time0 = time.time()

            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, fx, norm_grad))
                fxlist.append(fx)


            stop_reason = self._check_stopping_criterion(
                time0, gradnorm=norm_grad, iter=iter, costevals = costevals, stepsize = stepsize)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

            df0 = man.inner(x, fgradx, desc_dir)

            if df0 > 0:
                if verbosity >= 1:
                    print(['Line search warning: got an ascent direction (df0 = %2e), went the other way.\n'], df0);

                desc_dir = -desc_dir;
                df0 = -df0;


            # print("X: "+str(x)+" "+"cost: "+str(fx)+" "+"grad: "+str(norm_grad))
            # stepsize, newx, lsmem, lsstats = self.linesearch.search(problem, x, desc_dir, fx, df0, lsmem)
            stepsize, newx = self.linesearch.search(cost, man, x, desc_dir, fx, df0)

            # if 'grad' in lsmem:
            #     newfx = lsmem['cost'];
            #     newfgradx = lsmem['grad'];
            # else:
            newfx = cost(newx)
            timeg = time.time()
            newfgradx = grad(newx)
            # print("Time taken for grad: "+str(time.time()-timeg))
            
            # costevals = costevals + lsstats['costevals']
            newnorm_grad = man.norm(newx, newfgradx)

            #STORAGE TODO

            #Using previous and new information to update
            gradC = man.transp(x, newx, fgradx)
            grad_diff = newfgradx-gradC
            
            #Multiplying the stepsize in descent direction
            desc_dir_step = stepsize*desc_dir

            #Parallel transport descent to the new point
            desc_dir_step = man.transp(x, newx, desc_dir_step)
            #Update the previous saved info
            time1 = time.time()
            grad_diff_all, desc_dir_all, x_all, gd_all, ddgd_all, H = self.lbfgs_update(newx, man, grad_diff, desc_dir_step, grad_diff_all, desc_dir_all, self.numgrad, x_all, gd_all,ddgd_all, H, verbosity)

            if H == 0:
                break


            if len(gd_all)==0:
                desc_dir = 1/newnorm_grad*newfgradx
            else:
                desc_dir = self.desc_dir_cal(newfgradx, man, grad_diff_all, \
            desc_dir_all, x_all, ddgd_all, gd_all, len(gd_all) , H)


            #Change search direction because it is gradient descend
            desc_dir = -1*desc_dir
            
            #Update iterate info
            x = newx;
            fx = newfx;
            fgradx = newfgradx;
            norm_grad = newnorm_grad;
            
            #iter is the number of iterations we have accomplished.
            iter = iter + 1

            # print("Time taken: "+str(time.time()-time0))
        return x, fxlist

        


    def lbfgs_update(self, x, M, grad_diff, desc_dir, grad_diff_all,desc_dir_all,num_corr, x_all, gd_all, ddgd_all, Hdiag, verbosity):
        gd = M.inner(x,desc_dir,grad_diff)

        if gd > 1e-10:
            num = len(desc_dir_all);
            dd = M.inner(x,desc_dir, desc_dir)
            gg = M.inner(x,grad_diff, grad_diff)
            ddgd = dd / gd
            if num < num_corr:
                desc_dir_all.append(desc_dir)
                grad_diff_all.append(grad_diff)
                x_all.append(x)
                gd_all.append(gd)
                ddgd_all.append(ddgd)

            else:
                desc_dir_all = desc_dir_all[1:num]
                desc_dir_all.append(desc_dir)
                grad_diff_all = grad_diff_all[1:num]
                grad_diff_all.append(grad_diff)
                x_all = x_all[1:num+1]
                x_all.append(x)
                gd_all = gd_all[1:num]
                gd_all.append(gd)
                ddgd_all = ddgd_all[1:num]
                ddgd_all.append(ddgd)

            #Initial Hessian
            Hdiag = gd / gg;
        else:
            if verbosity:
                print('inner(gradient,descend) is small ... Remove Memory\n');

            #Cleaning History
            desc_dir_all = [];
            grad_diff_all = [];
            x_all = [x];
            gd_all = [];
            ddgd_all = [];
            #Hdiag = M.norm(x, desc_dir);
            Hdiag = 1;

        return grad_diff_all, desc_dir_all, x_all, gd_all, ddgd_all, Hdiag


    def desc_dir_cal(self, p, M, grad_diff_all, desc_dir_all,x_all, ddgd, gd, corrections , Hdiag):

        coef = M.inner(x_all[corrections], p, desc_dir_all[corrections-1]) / gd[corrections-1]
        p_prev = p - coef*grad_diff_all[corrections-1]

        #Doing an inverse retraction from a point to the previous one
        p_invtransp = M.transp(x_all[corrections],x_all[corrections-1],p_prev)
        if corrections >1:
            vec_prec= self.desc_dir_cal(p_invtransp, M, grad_diff_all, desc_dir_all, \
            x_all, ddgd, gd, corrections-1, Hdiag)
        else:
            vec_prec = Hdiag*p_prev
        
        vec_new = M.transp(x_all[corrections-1],x_all[corrections],vec_prec)

        coef = M.inner(x_all[corrections],vec_new,grad_diff_all[corrections-1])
        p_desc = vec_new-(coef/gd[corrections-1]) * desc_dir_all[corrections-1]
        p_desc =  p_desc + ddgd[corrections-1]*p

        return p_desc











