Contributions of element-wise Alternative Least Squares (eALS):

1. Design L_2 and add it to L_eALS to assign a non-uniform weight to missing data based on item popularity i.e. s_k.

2. Utilizing the pre-computed cache i.e. sum of V^T_k * V_k for all items k, sum of U^T_k * U_k for all users k to reduce computational complexity.


Ref: https://arxiv.org/pdf/1708.05024.pdf

Ref: http://csse.szu.edu.cn/staff/panwk/recommendation/OCCF/eALS.pdf