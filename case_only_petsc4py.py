from petsc4py import PETSc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank=MPI.COMM_WORLD.Get_rank()
size = comm.Get_size()

n = 10  # Global matrix size

A = PETSc.Mat().create(comm=comm)
A.setSizes([10, 10])
A.setType('aij')  # Sparse matrix
A.setUp()


for i in range(10):
    A[i, i] = 2.0
    if i > 0:
        A[i, i - 1] = -1.0
    if i < n - 1:
        A[i, i + 1] = -1.0

# Assemble the matrix
A.assemble()

Istart, Iend= A.getOwnershipRange()

#    A is 
#    [2    -1     0     0     0     0     0     0     0     0
#    -1     2    -1     0     0     0     0     0     0     0
#     0    -1     2    -1     0     0     0     0     0     0
#     0     0    -1     2    -1     0     0     0     0     0
#     0     0     0    -1     2    -1     0     0     0     0
#     0     0     0     0    -1     2    -1     0     0     0
#     0     0     0     0     0    -1     2    -1     0     0
#     0     0     0     0     0     0    -1     2    -1     0
#     0     0     0     0     0     0     0    -1     2    -1
#     0     0     0     0     0     0     0     0    -1     2]

# creat right vector 

b = PETSc.Vec().createMPI(10,5,PETSc.COMM_WORLD)

for m in range(10):
    b[m]=1 

ksp = PETSc.KSP().create(PETSc.COMM_WORLD)  
ksp.setOperators(A)  
ksp.setType("preonly")  

pc = ksp.getPC() 
pc.setType("lu") 
x = A.createVecRight()


def my_setValue(u, m, beta):
    # u.setValue(m, u.getValue(m) + beta)   
    u.assemble() 
    Istart, Iend= u.getOwnershipRange()
    if Istart <= m<Iend:  
        local_index = m - Istart
        u.setValue(m, u.getArray()[local_index] + beta)
    return

def my_setValue2(u, m, xishu,beta):
    # u.setValue(m, u.getValue(m)*xishu +beta)   
    u.assemble() 
    Istart, Iend= u.getOwnershipRange()
    if Istart <= m<Iend:  
        local_index = m - Istart
        u.setValue(m, u.getArray()[local_index]*xishu + beta)
    return

def my_setValue_mult(u,n,m, beta):
    # u.setValue(n, u.getValue(m) * beta)   
    u.assemble() 
    Istart, Iend= u.getOwnershipRange()
    if Istart <= m<Iend:  
        local_index = m - Istart
        u.setValue(n, u.getArray()[local_index] *beta)
    return

def my_indx0(v,m): 
    #return: v[m] Variables are local 
    cc=0 
    Istart, Iend= v.getOwnershipRange() 
    if Istart <= m<Iend:   
        local_index = m - Istart  
        cc=v.getArray()[local_index] 
    return  cc

def my_indx(v,m): 
    #return : v[m] Variables are global
    cc=0
    ranges= v.getOwnershipRanges() 
    indices = np.where(m >= ranges)[-1] 
    rank_root=indices[-1]
    Istart, Iend= v.getOwnershipRange() 
    if Istart <= m<Iend:   
        local_index = m - Istart 
        cc=v.getArray()[local_index] 

    cc=comm.bcast(cc, root=rank_root)
    return  cc

def hou_gmres_para(A,b,maxit):
    # Non reboot gmres algorithm 
    # apply left preconditioner
    # maxit : Krylov Subspace dimension = Maximum Number Of Iterations
    # note: The algorithm is not complete,This is just the first version.
    pc_g = PETSc.PC().create() 
    pc_g.setOperators(A) 
    pc_g.setType('mat') 

    n = b.getSize() 
    x = b.duplicate() 
    v = b.duplicate() 

    U=[] 
    J=[] 
    R = PETSc.Mat().create(PETSc.COMM_WORLD)  
    R.setSizes([maxit, maxit])  
    R.setType('aij')  # Use sparse storage 
    R.setUp()  

    w = PETSc.Vec().createMPI(maxit+1,R.block_size,PETSc.COMM_WORLD)   
    
    b.assemble() 
    r = b.copy() 
  
    u = r.copy()  
    normr = r.norm()  

    beta = np.sign(r.getArray()[0]) * normr    

    my_setValue(u,0,beta)  

    u.assemble()    
    u.normalize()  #r.normlize(r)
    U.append(u)  

    my_setValue(w,0,beta)  


    for m in range(maxit):
        v=U[m].copy() 
        
        v.scale(-2*my_indx(v,m))   
        my_setValue(v,m,1)

        for k in range(m-1,-1,-1):  
            Utemp = U[k].copy()  
            v.assemble()   
            v -= Utemp * (2 * v.dot(Utemp)) 

        v.assemble() 
        v.normalize() 
        vk=v.copy() 
        pc_g.apply(vk,v)

        for k in range(m+1): 
            Utemp = U[k].copy() 
            v = v-Utemp * (2 * v.dot(Utemp))# 

        u = v.copy() 
        u.setValues(range(m+1),np.zeros(m+1)) 
        u.assemble() 
        alpha = u.norm() 

        
        if alpha != 0:
            alpha= alpha*np.sign(my_indx(v,m))
            
            my_setValue(u,m+1,alpha)
                
            u.assemble()
            u.normalize()
            U.append(u)
            
            indices = range(m + 2, n) 
            values = np.zeros(len(indices)) 
            v.setValues(indices, values)      
            v.setValues(m+1, -alpha) 

        for colJ in range(m):
            v.assemble()

            tmpv = my_indx(v,colJ) 
            #v[colJ] = np.conj(J[colJ][0]) * v[colJ] + np.conj(J[colJ][1]) * v[colJ + 1]
            b_tmp=np.conj(J[colJ][1]) * my_indx(v,colJ+1) 
            my_setValue2(v,colJ,np.conj(J[colJ][0]),b_tmp)
            v.assemble()
            #v[colJ + 1] = -J[colJ][1] * tmpv + J[colJ][0] * v[colJ + 1]
            my_setValue2(v,colJ + 1,J[colJ][0],-J[colJ][1] * tmpv)


        if m != v.getSize(): 
            
            vm=my_indx(v,m)  
            vm2=my_indx(v,m+1)  
            rho = np.linalg.norm(np.array([vm, vm2]))  
    
            tmpvm_vm2=PETSc.Vec().createSeq(2)   
            tmpvm_vm2.setValues(0,vm/rho)  
            tmpvm_vm2.setValues(1,vm2/rho)   
            J.append(tmpvm_vm2)  #J是global 
            
            my_setValue_mult(w,m+1,m,-J[m][1]) #w[m + 1] = -J[m][1] * w[m] 
            my_setValue_mult(w,m,m,np.conj(J[m][0]))#w[m] = np.conj(J[m][0]) * w[m] 

            v.setValues(m,rho) #v[m] = rho 
            v.setValues(m+1,0) #v[m + 1] = 0
        ##R(:,m) = v(1:maxit);
        v.assemble()
        r_start,r_end = R.getOwnershipRange()

        # 
        v_astmp= PETSc.Vec().createMPI(maxit,R.block_size,PETSc.COMM_WORLD)

        for kk in range(size):
            is1 = PETSc.IS().createStride(r_end-r_start,r_start, 1)  #length:  begin:  step:
            sct = PETSc.Scatter().create(v,is1,v_astmp,None)

        sct.scatter(v,v_astmp)   # v_astmp[:]=v[1:size()]
        
        
        v_astmp.assemble()
        v_local = v_astmp.getArray()
        local_rows = list(range(r_start, r_end))

        v_local_indices=[]  
        if r_start<=m:   
            v_local_indices = list(range(min(m-r_start+1, r_end - r_start)))

        if v_local_indices:
            R.setValues(local_rows[:len(v_local_indices)], [m], v_local[v_local_indices])
        R.assemble()
    
    # w_b=w(1:m+1) 
    
    w_b= PETSc.Vec().createMPI(maxit,R.block_size,PETSc.COMM_WORLD)

    for kk in range(size):
        is1 = PETSc.IS().createStride(r_end-r_start,r_start, 1) #length:  begin:  step:
        sct = PETSc.Scatter().create(w,is1,w_b,None)
    sct.scatter(w,w_b)  
     
    w_b.scale(-1)
    
    y = R.getVecRight()
    ksp_m2= PETSc.KSP().create()
    ksp_m2.setType(PETSc.KSP.Type.PREONLY)
    pc0 = ksp_m2.getPC()
    pc0.setType(PETSc.PC.Type.LU)
    ksp_m2.setOperators(R)
    R.assemble()
    ksp_m2.solve(w_b, y)


    #additive = U[m] * (-2 * y[m] * np.conj(U[m][m]))   
    #additive[m] =additive[m]+ y[m] 
    additive=U[m].copy()    
    y_m=my_indx(y,m)
    additive.scale((-2 *  y_m* np.conj(my_indx(additive,m))))
    my_setValue(additive,m,y_m)

    
    for k in range(m-1, -1, -1):
        additive.assemble()
        y_k=my_indx(y,k)
        my_setValue(additive,k,y_k)
        additive.assemble()
        # y.axpy(alph,x) Compute and store y = ɑ·x + y.
        #additive -= U[k] * (2 * additive.dot(U[k]))
        UK=U[k].copy()
        additive.axpy(-2 * additive.dot(UK),UK)


    x.axpy(1, additive)

    return x



def hou_gmres_para_pre(A,b,P,maxit):
    # Non reboot gmres algorithm 
    # apply left preconditioner
    # maxit : Krylov Subspace dimension = Maximum Number Of Iterations
    # note: The algorithm is not complete,This is just the first version.
    # Because there is no absolutely correct residual judgment
    # The convergence conditions of ksp require additional attention.
    # Otherwise, it will lead to incorrect solutions
    ksp_p = PETSc.KSP().create(PETSc.COMM_WORLD)  
    ksp_p.setOperators(P)
    ksp_p.setType("gmres")
    ksp_p.setInitialGuessNonzero(False)
    ksp_p.getPC().setType('bjacobi')
    opts = PETSc.Options()  
    opts["ksp_rtol"] = 1.0e-10     
    opts["ksp_atol"] = 1.0e-6     
    opts["ksp_max_it"] =2000       
    ksp_p.setFromOptions()

    pc_g = PETSc.PC().create() 
    pc_g.setOperators(A) 
    pc_g.setType('mat') 

    n = b.getSize() 
    x = b.duplicate() 
    v = b.duplicate() 

    U=[] 
    J=[] 
    R = PETSc.Mat().create(PETSc.COMM_WORLD)  
    R.setSizes([maxit, maxit])  
    R.setType('aij')  # Use sparse storage 
    R.setUp()  

    w = PETSc.Vec().createMPI(maxit+1,R.block_size,PETSc.COMM_WORLD)   
    
    b.assemble() 
    pre_rtmp = b.copy() 
    
    # apply preconditioner  
    r = b.duplicate() 
    ksp_p.solve(pre_rtmp,r)
    #
    u = r.copy()  
    normr = r.norm()  

    beta = np.sign(r.getArray()[0]) * normr    

    my_setValue(u,0,beta)  

    u.assemble()    
    u.normalize()  #r.normlize(r)
    U.append(u)  

    my_setValue(w,0,beta)  


    for m in range(maxit):
        v=U[m].copy() 
        
        v.scale(-2*my_indx(v,m))  
        
        #v.setValue(m, v.getValue(m) + 1)    
        my_setValue(v,m,1)

        for k in range(m-1,-1,-1):  
            Utemp = U[k].copy()  
            v.assemble()   
            v -= Utemp * (2 * v.dot(Utemp)) 

        v.assemble() 
        v.normalize() 
        vk=v.copy() 
        pc_g.apply(vk,v)

        #apply preconditioner
        pre_vtmp = v.copy()
        ksp_p.solve(pre_vtmp,v)
        #
        for k in range(m+1): 
            Utemp = U[k].copy() 
            v = v-Utemp * (2 * v.dot(Utemp))# 

        u = v.copy() 
        u.setValues(range(m+1),np.zeros(m+1)) 
        u.assemble() 
        alpha = u.norm() 

        if alpha != 0:
            alpha= alpha*np.sign(my_indx(v,m))
            
            my_setValue(u,m+1,alpha)
                
            u.assemble()
            u.normalize()
            U.append(u)
            
            indices = range(m + 2, n) 
            values = np.zeros(len(indices)) 
            v.setValues(indices, values)      
            v.setValues(m+1, -alpha) 

        for colJ in range(m):
            v.assemble()

            tmpv = my_indx(v,colJ) 
            #v[colJ] = np.conj(J[colJ][0]) * v[colJ] + np.conj(J[colJ][1]) * v[colJ + 1]
            b_tmp=np.conj(J[colJ][1]) * my_indx(v,colJ+1) 
            
            my_setValue2(v,colJ,np.conj(J[colJ][0]),b_tmp)

            v.assemble()
            #v[colJ + 1] = -J[colJ][1] * tmpv + J[colJ][0] * v[colJ + 1]
            my_setValue2(v,colJ + 1,J[colJ][0],-J[colJ][1] * tmpv)


        if m != v.getSize(): 
            #v.assemble() 
            vm=my_indx(v,m)  
            vm2=my_indx(v,m+1)  

            rho = np.linalg.norm(np.array([vm, vm2]))  
        
            tmpvm_vm2=PETSc.Vec().createSeq(2)   
            tmpvm_vm2.setValues(0,vm/rho)  
            tmpvm_vm2.setValues(1,vm2/rho)   
            J.append(tmpvm_vm2) 
            
            my_setValue_mult(w,m+1,m,-J[m][1]) #w[m + 1] = -J[m][1] * w[m] 
            my_setValue_mult(w,m,m,np.conj(J[m][0]))#w[m] = np.conj(J[m][0]) * w[m] 

            v.setValues(m,rho) #v[m] = rho 
            v.setValues(m+1,0) #v[m + 1] = 0
        
        ##R(:,m) = v(1:maxit);
        v.assemble()
        print(v.norm())
        r_start,r_end = R.getOwnershipRange()

   
        v_astmp= PETSc.Vec().createMPI(maxit,R.block_size,PETSc.COMM_WORLD)

        for kk in range(size):
            is1 = PETSc.IS().createStride(r_end-r_start,r_start, 1) #length:  begin:  step:
            sct = PETSc.Scatter().create(v,is1,v_astmp,None)

        sct.scatter(v,v_astmp)   # v_astmp[:]=v[1:size()]
        
        # R=(:m,m) = v_astmp(:m)
        v_astmp.assemble()
        v_local = v_astmp.getArray()
        local_rows = list(range(r_start, r_end))

        v_local_indices=[]  
        if r_start<=m:   
            v_local_indices = list(range(min(m-r_start+1, r_end - r_start)))

        if v_local_indices:
            R.setValues(local_rows[:len(v_local_indices)], [m], v_local[v_local_indices])
        R.assemble()
    
    # w_b=w(1:m+1) 
    
    w_b= PETSc.Vec().createMPI(maxit,R.block_size,PETSc.COMM_WORLD)

    for kk in range(size):
        is1 = PETSc.IS().createStride(r_end-r_start,r_start, 1) #length:  begin:  step:
        sct = PETSc.Scatter().create(w,is1,w_b,None)
    sct.scatter(w,w_b)  
     
    w_b.scale(-1)
    
    y = R.getVecRight()
    ksp_m2= PETSc.KSP().create()
    ksp_m2.setType(PETSc.KSP.Type.PREONLY)
    pc0 = ksp_m2.getPC()
    pc0.setType(PETSc.PC.Type.LU)
    ksp_m2.setOperators(R)
    R.assemble()
    ksp_m2.solve(w_b, y)

    #additive = U[m] * (-2 * y[m] * np.conj(U[m][m]))   
    #additive[m] =additive[m]+ y[m] 
    additive=U[m].copy()    
    y_m=my_indx(y,m)
    additive.scale((-2 *  y_m* np.conj(my_indx(additive,m))))
    my_setValue(additive,m,y_m)

    for k in range(m-1, -1, -1):
        additive.assemble()
        y_k=my_indx(y,k)
        my_setValue(additive,k,y_k)
        additive.assemble()
        # y.axpy(alph,x) Compute and store y = ɑ·x + y.
        #additive -= U[k] * (2 * additive.dot(U[k]))
        UK=U[k].copy()
        additive.axpy(-2 * additive.dot(UK),UK)

    x.axpy(1, additive)
    
    
    ## rv=norm(inv(P)*(a*x-b))
    r1 = b.duplicate() 
    rv = b.duplicate()  
    pc_g.apply(x,r1)  # r1=A*x
    r1.axpy(-1, b)  # r1=r1-b=A*x-b

    ksp_p.solve(r1,rv)
    norm_rv=rv.norm()

    return x,norm_rv

x=hou_gmres_para(A,b,3)

print(f'rank{rank}',x.array)







