// Started from https://github.com/jalan/spam=
//https://stuff.mit.edu/afs/sipb/project/python/src/python-numeric-22.0/doc/www.pfdubois.com/numpy/html2/numpy-13.html
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

/* Low level routines */
void compute_rtransfer(int nr, int central_nmu, double *S_C1, double *S_C2, 
        double *expdy, double *Jw, double *Hw,
        double *Jmat,  double *Hmat){
    /* Array Strides for incremental and weight matrices  */
    int sti = nr - 1;
    double exptau=1;
    double *Iinner_C =  (double*)malloc(nr * (nr + central_nmu) * sizeof(double));
    if (Iinner_C==NULL) return; //This shouldn't happen...
    int i,j,k;

    //Zero the Iinner array
    for (k=0;k<nr*(nr+central_nmu);k++) Iinner_C[k]=0.0;

    // We have to do our first 3-dimensional loop here, considering inwards going rays
    //first (mu <= 0), then outwards going rays (mu>0)
    for (j=0;j<nr;j++){
        //i is the y value.
        for (i=0;i<central_nmu+j+1;i++){
            //Integrating outwards along the incoming ray.
            //Reset Temporary optical depth
            exptau = 1;
            for (k=0;k<nr-j-1;k++){
                Jmat[j*nr+j+k]   += exptau*S_C1[i*sti+j+k]*Jw[i*nr+j];
                Jmat[j*nr+j+k+1] += exptau*S_C2[i*sti+j+k]*Jw[i*nr+j];
                Hmat[j*nr+j+k]   -= exptau*S_C1[i*sti+j+k]*Hw[i*nr+j];
                Hmat[j*nr+j+k+1] -= exptau*S_C2[i*sti+j+k]*Hw[i*nr+j];
                //Inner Boundary and mu=0 rays
                if ((j==0) || (i==central_nmu+j)){
                    Iinner_C[i*nr+j+k]   += exptau*S_C1[i*sti+j+k];
                    Iinner_C[i*nr+j+k+1] += exptau*S_C2[i*sti+j+k];
                    }
                exptau *= expdy[i*sti+j+k];
            }
        }
    }

    //We have to do our next 3-dimensional loop here, for outwards going rays.
    for (j=0;j<nr;j++){
        //i is the y value. We have already covered the mu=0 ray, so don't
        //need to consider it again.
        for (i=0;i<central_nmu+j;i++){
            //Integrating inwards along the outgoing ray.
            exptau = 1;
            for (k=0;k<j;k++){
                Jmat[j*nr+j-k]   += exptau*S_C1[i*sti+j-k-1]*Jw[i*nr+j];
                Jmat[j*nr+j-k-1] += exptau*S_C2[i*sti+j-k-1]*Jw[i*nr+j];
                Hmat[j*nr+j-k]   += exptau*S_C1[i*sti+j-k-1]*Hw[i*nr+j];
                Hmat[j*nr+j-k-1] += exptau*S_C2[i*sti+j-k-1]*Hw[i*nr+j];
                exptau *= expdy[i*sti+j-k-1];
            }
            //Now we're at the inner boundary, or about to go outwards, 
            //and have to apply that boundary condition. 
            if (i < central_nmu){
                Jmat[j*nr] += 2*exptau*Jw[i*nr+j];
                Hmat[j*nr] += 2*exptau*Hw[i*nr+j];
                for (k=0;k<nr;k++){
                    Jmat[j*nr+k] -= exptau*Iinner_C[i*nr+k]*Jw[i*nr+j];
                    Hmat[j*nr+k] -= exptau*Iinner_C[i*nr+k]*Hw[i*nr+j];
                }
            } else {
                for (k=i-central_nmu;k<nr;k++){
                    Jmat[j*nr+k] += exptau*Iinner_C[i*nr+k]*Jw[i*nr+j];
                    Hmat[j*nr+k] += exptau*Iinner_C[i*nr+k]*Hw[i*nr+j];
                }
            }
        }
    }
    free(Iinner_C);   
}

/* Python level routines */
static PyObject* codex2rtError;

static PyObject* codex2rt_compute_rtransfer(PyObject* self, PyObject* args) {
    PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL, *arg4=NULL, *arg5=NULL, *out1=NULL, *out2=NULL;
    PyArrayObject *a_S_C1=NULL, *a_S_C2=NULL, *a_expdy=NULL, *a_Jw=NULL, *a_Hw=NULL, *a_Jmat=NULL, *a_Hmat=NULL;
    npy_intp nr, central_nmu;

    /* First, parse objects */
    if (!PyArg_ParseTuple(args, "OOOOOO!O!", &arg1, &arg2, &arg3, &arg4, &arg5,
        &PyArray_Type, &out1, &PyArray_Type, &out2)) return NULL;

    /* Now convert to Numpy arrays. hopefully this is super-quick if
    they are already arrays!*/
    a_S_C1 = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (a_S_C1 == NULL) return NULL;
    a_S_C2 = (PyArrayObject *)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (a_S_C2 == NULL) goto fail;
    a_expdy = (PyArrayObject *)PyArray_FROM_OTF(arg3, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (a_expdy == NULL) goto fail;
    a_Jw = (PyArrayObject *)PyArray_FROM_OTF(arg4, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (a_Jw == NULL) goto fail;
    a_Hw = (PyArrayObject *)PyArray_FROM_OTF(arg5, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (a_Hw == NULL) goto fail;
#if NPY_API_VERSION >= 0x0000000c
    a_Jmat = (PyArrayObject *)PyArray_FROM_OTF(out1, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    a_Jmat = (PyArrayObject *)PyArray_FROM_OTF(out1, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (a_Jmat == NULL) goto fail;
#if NPY_API_VERSION >= 0x0000000c
    a_Hmat = (PyArrayObject *)PyArray_FROM_OTF(out2, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    a_Hmat = (PyArrayObject *)PyArray_FROM_OTF(out2, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (a_Hmat == NULL) goto fail;

    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */
    /* Can also use:
    void *PyArray_DATA(PyArrayObject *arr)*/
    
    /* Check number of dimensions for arrays */
    if ( (PyArray_NDIM(a_S_C1) != 2) || (PyArray_NDIM(a_S_C2) != 2) ||
         (PyArray_NDIM(a_expdy) != 2) || (PyArray_NDIM(a_Jw) != 2) ||
         (PyArray_NDIM(a_Hw) != 2) || (PyArray_NDIM(a_Jmat) != 2) ||
         (PyArray_NDIM(a_Hmat) != 2) ){
        PyErr_SetString(
            codex2rtError, "All 5 input and 2 output arrays must have 2 dimensions");
        goto fail;
    }
            
    /* Find the dimensions and check them */
    nr = PyArray_DIM(a_Jw,1);
    central_nmu = PyArray_DIM(a_Jw,0) - nr;
    if ( (PyArray_DIM(a_Hw,1) != nr) || (PyArray_DIM(a_Hw,0) != nr+central_nmu) ) {
        PyErr_SetString(
            codex2rtError, "Both Jw and Hw must have (nr + central_nmu, nr) dimensions.");
        goto fail;
    }
 
    if ( (PyArray_DIM(a_S_C1,1) != nr-1) || (PyArray_DIM(a_S_C1,0) != nr+central_nmu-1) ||
         (PyArray_DIM(a_S_C2,1) != nr-1) || (PyArray_DIM(a_S_C2,0) != nr+central_nmu-1) ||
         (PyArray_DIM(a_expdy,1) != nr-1) || (PyArray_DIM(a_S_C2,0) != nr+central_nmu-1) ){    
        PyErr_SetString(
            codex2rtError, "S_C1, S_C2 and expdy must have (nr + central_nmu - 1, nr-1) dimensions.");
        goto fail;
    }

    //Now do the low level computation.
    compute_rtransfer(nr, central_nmu, (double *)PyArray_DATA(a_S_C1), (double *)PyArray_DATA(a_S_C2), 
        (double *)PyArray_DATA(a_expdy), (double *)PyArray_DATA(a_Jw), (double *)PyArray_DATA(a_Hw),
        (double *)PyArray_DATA(a_Jmat), (double *)PyArray_DATA(a_Hmat));

    Py_DECREF(a_S_C1);
    Py_DECREF(a_S_C2);
    Py_DECREF(a_expdy);
    Py_DECREF(a_Jw);
    Py_DECREF(a_Hw);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_ResolveWritebackIfCopy(a_Jmat);
    PyArray_ResolveWritebackIfCopy(a_Hmat);
#endif
    Py_XDECREF(a_Jmat);
    Py_XDECREF(a_Hmat);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_DECREF(a_S_C1);
    Py_DECREF(a_S_C2);
    Py_DECREF(a_expdy);
    Py_DECREF(a_Jw);
    Py_DECREF(a_Hw);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_DiscardWritebackIfCopy(a_Jmat);
    PyArray_DiscardWritebackIfCopy(a_Hmat);
#endif
    Py_XDECREF(a_Jmat);
    Py_XDECREF(a_Hmat);
    return NULL;
}


static PyObject* codex2rt_multi_rtransfer(PyObject* self, PyObject* args) {
    Py_RETURN_NONE;
}



static PyMethodDef codex2rtMethods[] = {
    {
        "compute_rtransfer",
        codex2rt_compute_rtransfer,
        METH_VARARGS,
        "Compute radiative transfer for a single wavelength.\n\n"
        "Return an (nr x nr) radiative transfer matrix for J and H",
    },
    {
        "multi_rtransfer",
        codex2rt_multi_rtransfer,
        METH_VARARGS,
        "Compute radiative transfer for a many wavelengths.\n\n"
        "Return an (nw x nr x nr) radiative transfer matrix for J and H",
    },
    {NULL, NULL, 0, NULL},  // sentinel
};

static PyModuleDef codex2rtModule = {
    PyModuleDef_HEAD_INIT,
    "codex2rt",
    "Radiative transfer module for CODEX2.",
    -1,
    codex2rtMethods,
};

PyMODINIT_FUNC PyInit_codex2rt() {
    PyObject* module;

    module = PyModule_Create(&codex2rtModule);
    if (module == NULL) {
        return NULL;
    }
    codex2rtError = PyErr_NewException("codex2rt.Error", NULL, NULL);
    Py_INCREF(codex2rtError);
    PyModule_AddObject(module, "Error", codex2rtError);
    import_array();
    return module;
}
