import numpy as np

def adam_step(param, grad, m, v, t,
              lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param= np.array(param)
    grad=np.array(grad)
    m=np.array(m)
    v=np.array(v)
    m_comp=beta1*m+(1-beta1)*grad
    v_comp=beta2*v+(1-beta2)*grad*grad
    m_new=m_comp/(1-beta1**t)
    v_new=v_comp/(1-beta2**t)
    param_new=param-lr*(m_new/(np.sqrt(v_new)+eps))
    return (param_new,m_comp,v_comp)

   
    

