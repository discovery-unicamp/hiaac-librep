import numpy as np
import sklearn.neighbors as nb
from sklearn import manifold
from sklearn import preprocessing

from scipy import sparse, linalg


#==============================================================================
#                   Farthest Point Sampling
#==============================================================================
def FPS(W,num_FPS):

    Numpts = W.shape[0]
    indices_FPS = []
    indices_FPS.append(np.random.randint(0,Numpts))
    
    Dsk = np.zeros([num_FPS,Numpts])
         
    for itern in range(0,num_FPS-1):
        # Search the max value different from inf and nan in Dsk
        # first compute the distances:
        Dsk[itern,:] = sparse.csgraph.dijkstra(W,directed = False,indices=indices_FPS[itern])
        # print(itern, indices_FPS[itern], Dsk)
        indices_FPS.append(np.argmax(np.ndarray.min(Dsk[0:itern+1,:],0)))
        # print(itern, indices_FPS)
    Dsk[itern+1,:] = sparse.csgraph.dijkstra(W,directed = False,indices=indices_FPS[-1])

    Landmarks = {'indices':indices_FPS,'Ds': Dsk,'W':W}
    return Landmarks

#==============================================================================
#                   Standard ISOMAP
#==============================================================================
def ISOMAP(Data,numNN):

    W = nb.kneighbors_graph(Data, numNN, mode='distance',
                                   metric='minkowski', p=2, metric_params=None, 
                                   include_self=False, n_jobs=-1)

#    nbrs = nb.NearestNeighbors(n_neighbors=8, algorithm='kd_tree').fit(Data)
#    W = nbrs.kneighbors_graph(Data,mode='distance')

    Numpts = W.shape[0]
    Dss = np.power(sparse.csgraph.dijkstra(W,directed = False),2)

    H = np.identity(Numpts) - (1.0/Numpts)*np.outer(np.ones(Numpts),np.ones(Numpts))
    
    DsH = np.multiply(H.dot(Dss).dot(H), -0.5)
    
    EigVal, EigVec = linalg.eigh(DsH, eigvals=(Numpts-3,Numpts-1))
   
    ISOMAP_Embedding = np.diag(np.sqrt(EigVal)).dot(EigVec.T).T

    return ISOMAP_Embedding 



#==============================================================================
#                       Landmark-ISOMAP
#==============================================================================
def L_ISOMAP(Data,numNN,Landmarks):
    
    if len(Landmarks)==1:
       print('Yes') 
       W = nb.kneighbors_graph(Data, numNN, mode='distance',
                                       metric='minkowski', p=2, metric_params=None, 
                                       include_self=False, n_jobs=-1)
       Landmarks = FPS(W,Landmarks[0])
    
    
    Dsk = np.power(Landmarks['Ds'],2)
    indices_FPS = Landmarks['indices']       
    
    
    Ns = len(indices_FPS)
    Numpts = Data.shape[0]
        
    Ds_FPS = Dsk[:,indices_FPS]
    H = np.identity(Ns) - (1.0/Ns)*np.outer(np.ones(Ns),np.ones(Ns))
    
    DsH = (-0.5)*np.matmul(np.matmul(H,Ds_FPS),H)
    EigVal, EigVec = linalg.eigh(DsH, eigvals=(Ns-2,Ns-1))
    
    Lkhash=np.diag(np.reciprocal(np.sqrt(EigVal))).dot(EigVec.T)
    print(Lkhash.shape)
#    lk1 = (1.0/np.sqrt(EigVal[0]))*EigVec[:,0]; lk2 = (1.0/np.sqrt(EigVal[1]))*EigVec[:,1];
#    Lkhash = np.vstack((lk1,lk2))    
    
    Delmu = np.matlib.repmat(np.expand_dims(Ds_FPS.sum(1),axis=1),1,Numpts)
    
    Embedding_LandmarkISOMAP = ((-0.5)*np.matmul(Lkhash,Dsk) + (0.5)*np.matmul(Lkhash,Delmu)).T
    
#    Embedding_LandmarkISOMAP = ((-0.5)*np.matmul(Lkhash,Dsk-Delmu)).T
    
    return Embedding_LandmarkISOMAP
    
#==============================================================================
#                       Landmark-ISOMAP - OS Extention
#==============================================================================
def L_ISOMAP_OSExt(Data,Landmarks,DataExt):

    W = Landmarks['W']    
    DsExt = np.zeros((DataExt.shape[0],Data.shape[0]+1))
    nbrs = nb.NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(Data)
    distances, indices = nbrs.kneighbors(DataExt)
    
    for ii in range(DataExt.shape[0]):
       dist_ii = distances[ii,:]
       ind_col_ii = indices[ii,:]
       ind_row_ii = np.zeros(ind_col_ii.shape)
       
       Wr = sparse.coo_matrix((dist_ii, (ind_row_ii, ind_col_ii)), shape=(1, Data.shape[0])).tocsr()
       Wtemp=sparse.vstack([W,Wr])
       Wr = sparse.hstack([Wr,sparse.csr_matrix((1, 1), dtype=np.float64)]).transpose()
       Wtemp=sparse.hstack([Wtemp,Wr])
       DsExt[ii,:] = sparse.csgraph.dijkstra(Wtemp,directed = False,indices=Data.shape[0])

    
    Dsk = np.power(Landmarks['Ds'],2)
    indices_FPS = Landmarks['indices']   
    
    DsExt = DsExt[:,0:Data.shape[0]]
    DsExt = np.power(DsExt[:,indices_FPS],2).T
        
    Ns = len(indices_FPS)
        
    Ds_FPS = Dsk[:,indices_FPS]
    H = np.identity(Ns) - (1.0/Ns)*np.outer(np.ones(Ns),np.ones(Ns))
    
    DsH = (-0.5)*np.matmul(np.matmul(H,Ds_FPS),H)
    EigVal, EigVec = linalg.eigh(DsH, eigvals=(Ns-2,Ns-1))
    
    Lkhash=np.diag(np.reciprocal(np.sqrt(EigVal))).dot(EigVec.T)
    print(Lkhash.shape)
#    lk1 = (1.0/np.sqrt(EigVal[0]))*EigVec[:,0]; lk2 = (1.0/np.sqrt(EigVal[1]))*EigVec[:,1];
#    Lkhash = np.vstack((lk1,lk2))    
    
    Delmu = np.matlib.repmat(np.expand_dims(Ds_FPS.mean(1),axis=1),1,DataExt.shape[0])
    
    Embedding_LandmarkISOMAP = ((-0.5)*np.matmul(Lkhash,DsExt) + (0.5)*np.matmul(Lkhash,Delmu)).T
    
#    Embedding_LandmarkISOMAP = ((-0.5)*np.matmul(Lkhash,Dsk-Delmu)).T
    
    return Embedding_LandmarkISOMAP

#==============================================================================
#                      C -Isomap - OS Extention
#==============================================================================
def C_ISOMAP_OSExt(Data,Landmarks,DataExt):

    NumberOfNeighbours = 15
    W = Landmarks['W']    
    DsExt = np.zeros((DataExt.shape[0],Data.shape[0]+1))
    nbrs = nb.NearestNeighbors(n_neighbors=NumberOfNeighbours, algorithm='kd_tree').fit(Data)
    distances, indices = nbrs.kneighbors(DataExt)
    
    for ii in range(DataExt.shape[0]):
       dist_ii = distances[ii,:]
       ind_col_ii = indices[ii,:]
       ind_row_ii = np.zeros(ind_col_ii.shape)
       
       Wr = sparse.coo_matrix((dist_ii, (ind_row_ii, ind_col_ii)), shape=(1, Data.shape[0])).tocsr()
       Wtemp=sparse.vstack([W,Wr])
       Wr = sparse.hstack([Wr,sparse.csr_matrix((1, 1), dtype=np.float64)]).transpose()
       Wtemp=sparse.hstack([Wtemp,Wr]);
       Wtemp=sparse.csr_matrix(Wtemp);

       
       NPts_new = Wtemp.shape[0]
       MeanRowVector = Wtemp.sum(1)/NumberOfNeighbours;
       rows,cols = Wtemp.nonzero();
       D_Matrix_data = np.divide(Wtemp[rows,cols].transpose(),np.sqrt(np.multiply(MeanRowVector[rows],MeanRowVector[cols])))
       rows = rows.squeeze();cols = cols.squeeze(); 
       D_Matrix_data = np.asarray(D_Matrix_data).squeeze()
       Wprime = sparse.csr_matrix((D_Matrix_data, (rows, cols)), (NPts_new, NPts_new));

       DsExt[ii,:] = sparse.csgraph.dijkstra(Wprime,directed = False,indices=Data.shape[0])

    
    Dsk = np.power(Landmarks['Ds'],2)
    indices_FPS = Landmarks['indices']   
    
    DsExt = DsExt[:,0:Data.shape[0]]
    DsExt = np.power(DsExt[:,indices_FPS],2).T
        
    Ns = len(indices_FPS)
        
    Ds_FPS = Dsk[:,indices_FPS]
    H = np.identity(Ns) - (1.0/Ns)*np.outer(np.ones(Ns),np.ones(Ns))
    
    DsH = (-0.5)*np.matmul(np.matmul(H,Ds_FPS),H)
    EigVal, EigVec = linalg.eigh(DsH, eigvals=(Ns-2,Ns-1))
    
    Lkhash=np.diag(np.reciprocal(np.sqrt(EigVal))).dot(EigVec.T)
    print(Lkhash.shape)
#    lk1 = (1.0/np.sqrt(EigVal[0]))*EigVec[:,0]; lk2 = (1.0/np.sqrt(EigVal[1]))*EigVec[:,1];
#    Lkhash = np.vstack((lk1,lk2))    
    
    Delmu = np.matlib.repmat(np.expand_dims(Ds_FPS.sum(1),axis=1),1,DataExt.shape[0])
    
    Embedding_LandmarkISOMAP = ((-0.5)*np.matmul(Lkhash,DsExt) + (0.5)*np.matmul(Lkhash,Delmu)).T
    
#    Embedding_LandmarkISOMAP = ((-0.5)*np.matmul(Lkhash,Dsk-Delmu)).T
    
    return Embedding_LandmarkISOMAP

#==============================================================================
#                       L-SMACOF
#==============================================================================
def L_SMACOF(Data,numNN,Landmarks,Optparams):
    
    if len(Landmarks)==1:
        
       W = nb.kneighbors_graph(Data, numNN, mode='distance',
                                       metric='minkowski', p=2, metric_params=None, 
                                       include_self=False, n_jobs=-1)
       Landmarks = FPS(W,Landmarks[0])
        
    
    Dsk = np.power(Landmarks['Ds'],2)
    indices_FPS = Landmarks['indices']       
    
    Numpts = Data.shape[0]
    
    Ds_FPS = Dsk[:,indices_FPS]

    Embedding_SMACOF_FPS = manifold.MDS(n_components=2, metric=True, n_init=8, 
                                                  max_iter=Optparams['numIter'], verbose=0, 
                                                  eps=Optparams['tol'],n_jobs=-1, random_state=None, 
                                                  dissimilarity='precomputed').fit_transform(Ds_FPS)
    
    Embedding_SMACOF_FPS = preprocessing.scale(Embedding_SMACOF_FPS, with_std = False, with_mean = True)
    Lkhash = np.linalg.pinv(Embedding_SMACOF_FPS)
    
    Delmu = np.matlib.repmat(np.expand_dims(Ds_FPS.sum(1),axis=1),1,Numpts)
    
    Embedding_LandmarkSMACOF = ((-0.5)*np.matmul(Lkhash,Dsk) + (0.5)*np.matmul(Lkhash,Delmu)).T
    
    return Embedding_LandmarkSMACOF




