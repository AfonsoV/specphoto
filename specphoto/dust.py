import numpy as np


## Calzetti 2000 dust attenuation curve
def klambda(lambda_array):
    RV = 4.05
    sellong = (lambda_array>=0.63) & (lambda_array<=2.2)
    selshort = (lambda_array>=0.12) & (lambda_array<0.63)
    klambda = np.zeros_like(lambda_array)
    klambda[sellong] = 2.659 * ( -1.857 + 1.040/lambda_array[sellong]) + RV
    klambda[selshort] = 2.659 * ( - 2.156 + 1.509/lambda_array[selshort]\
                                  - 0.198/lambda_array[selshort]**2\
                                  + 0.011/lambda_array[selshort]**3)\
                        + RV
    return klambda

def klambda_ind(lam):
    RV = 4.05
    if lam>=0.63 and lam<=2.2:
        return 2.659 * ( -1.857 + 1.040/lam) + RV
    elif lam>=0.12 and lam<0.63:
        return 2.659 * ( -2.156 + 1.509/lam - 0.198/lam**2 + 0.011/lam**3) + RV


def uv_bump(lam,B):
    f1 = (lam*0.035)**2
    f2 = (lam**2 -0.2175**2)
    return B * f1/(f2**2+f1)

def klambda_salim_highz(lam):
    ## see Salim et al. 2018
    ## https://arxiv.org/pdf/1804.05850.pdf
    ## Table 1, Eqs 8 and 9
    a0 = -4.01
    a1 = 2.46
    a2 = -0.128
    a3 = 0.0098
    B = 2.27
    RV = 2.88
    lam_max = 2.12
    klambda = a0 + a1/lam + a2/lam**2 + a3/lam**3 + uv_bump(lam,B) + RV

    if isinstance(lam,float):
        if klambda<0:
            return 0
    else:
        klambda[klambda<0]=0

    return klambda
