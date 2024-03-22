from sympy import symbols, Function, Eq, Derivative, exp, dsolve, simplify

# Definindo as variáveis
t = symbols('t')
y = Function('y')(t)

# Definindo a equação diferencial
eq = Eq(Derivative(y, t), y * t**3 - 1.5 * y)

# Encontrando a solução geral
solucao = simplify(dsolve(eq, y))

print(solucao)