import numpy as np

def gauss_seidel(A, b, x0, Eppara, maxit):
    ne = len(b) # medir o tamanho do vetor de P no tempo anterior mais 8/3*eta*rx*Pw(ou P0), número de equações do sistema
    x = np.zeros(ne) if x0 is None else np.array(x0) # vetor de x com zeros se não for dado chute inicial, e vetor x0 se for fornecido chute inicial 

    iter = 0 
    Epest = np.linspace(100,100,ne) # vetor de erros relativos

    while np.max(Epest) >= Eppara and iter <= maxit: # enquanto o valor máximo do vetor de erros for maior ou igual ao critério de parada e o número de interações for menor que o máximo
        x_old = np.copy(x) # cria um vetor igual ao vetor de x, para guardar os valores encontrados e comparar 

        for i in range(ne): # loop nas equações do sistema 
            sum1 = np.dot(A[i, :i], x[:i]) # soma dos produtos escalares dos coeficientes da linha i da matriz de A pelo valores de x até a posição i
            sum2 = np.dot(A[i, i+1:], x_old[i+1:]) # soma dos produtos escalares dos coeficientes da linha i da matriz A pelos valores de x_old após a posição i
            x[i] = (b[i] - sum1 - sum2)/A[i,i] # atualiza x usando Gauss-Seidel
        
        Epest = np.abs((x-x_old)/x) * 100 # erro relativo 

        iter += 1 # número de interações 

    return x