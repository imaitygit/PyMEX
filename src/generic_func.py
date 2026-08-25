import numpy as np
import sys
from functools import partial
import sympy as sp
print_f = partial(print, flush=True)

def unit2super(m,A):
  """
  Returns the supercell lattice vectors of a 
  ginven unit-cell lattice vectors.
  @input
    m: array of integers (for supercell)
    A: Unit-cell lattice vectors
  """
  S = np.zeros((3,3))
  np.fill_diagonal(S,m)
  return np.matmul(A,S)


def ang2crys(A, pos):
  """
  Converts position(s) of an atom (or series of atoms)
  to crystal coordinates
  @input
    A: Lattice vectors
    pos: positions in ansgtrom
  """
  Ainv = np.linalg.inv(A)
  if len(pos.shape) == 1:
    pos_c = np.dot(pos, Ainv)
    return pos_c
  elif len(pos.shape) == 2:
    pos_c = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[0]):
      pos_c[i] = np.dot(pos[i], Ainv)
    return pos_c
  else:
    print_f("Unrecongnized data format!")
    print_f("Exiting...")
    sys.exit()


def crys2ang(A, pos_c):
  """
  Converts position(s) of an atom (or series of atoms)
  from crystal coordinates to angstroms.
  @input
    A: Lattice vectors
    pos_c: positions in crystal coordinates
  """
  if len(pos_c.shape) == 1:
    pos = np.dot(pos_c, A)
    return pos
  elif len(pos_c.shape) == 2:
    pos = np.zeros((pos_c.shape[0], pos_c.shape[1]))
    for i in range(pos_c.shape[0]):
      pos[i] = np.dot(pos_c[i], A)
    return pos
  else:
    print_f("Unrecongnized data format!")
    print_f("Exiting...")
    sys.exit()

# line counter
def linecounter(num_l, L):
  """
  Computes the number of lines to that writes/reads L
  elements/points with num_l of elements per line.
  """
  rem = L%num_l
  if rem != 0.0:
    l = L//num_l + 1
  else:
    l = L//num_l
  return l


# Chek unitarity of a matrix
def is_unitary(M):
  """
  Checks if a matrix, M is unitary or not.
  Works for both square and rectangular M.
  @input
   M: Matrix
  @output
    bool-True if the matrix is unitary
  """
  M = np.matrix(M)
  I = M.H * M
  return np.allclose(np.eye(I.shape[0]), I)


def LMatrix(
  chi_sym=None,
  chi_val=None,
  epsilon_b_val=None,
  epsilon_t_val=None,
  q_val=None,
):
  """
  Layer matrix for a 2D material interface.

  L(chi, epsilon_b, epsilon_t) = 1/(2*epsilon_t) *
      [[epsilon_t + epsilon_b - chi*q,  epsilon_t - epsilon_b - chi*q],
       [epsilon_t - epsilon_b + chi*q,  epsilon_t + epsilon_b + chi*q]]

  Parameters
  ----------
  chi_sym       : sympy symbol, optional. Defaults to locally defined `chi`.
  chi_val       : float, optional. Numeric value for chi.
  epsilon_b_val : float, optional. Numeric value for epsilon below.
  epsilon_t_val : float, optional. Numeric value for epsilon above.
  q_val         : float, optional. Numeric value for in-plane wavevector.

  Returns
  -------
  sympy.Matrix : Symbolic or numeric 2x2 layer matrix.
  """
  # Define symbolic variables locally
  q, epsilon_b, epsilon_t = sp.symbols('q epsilon_b epsilon_t', real=True)
  _chi = chi_sym if chi_sym is not None else sp.symbols('chi', real=True)

  # Build symbolic layer matrix
  # See paper for details: XXXXX
  L = sp.Matrix([
    [epsilon_t + epsilon_b - _chi * q,  epsilon_t - epsilon_b - _chi * q],
    [epsilon_t - epsilon_b + _chi * q,  epsilon_t + epsilon_b + _chi * q],
  ]) / (2 * epsilon_t)

  # Build substitution dictionary from provided numeric values
  subs_dict = {
    k: v for k, v in {
      q         : q_val,
      _chi      : chi_val,
      epsilon_b : epsilon_b_val,
      epsilon_t : epsilon_t_val,
    }.items()
    if v is not None
  }

  if not subs_dict:
    return L

  L = L.subs(subs_dict)

  # If any value is a float, evaluate to numeric
  if any(isinstance(v, float) for v in subs_dict.values()):
    L = L.evalf()

  return L


def PMatrix(
  d_sym=None,
  d_val=None,
  q_val=None,
):
  """
  Propagation matrix for distance d.

  P(d) = [[exp(-q*d), 0      ],
          [0,         exp(q*d)]]

  To avoid sympy's tendency to convert exp(-xxx) to
  exp(xxx) while simplifying, we define:

  P(d) = [[f, 0],
          [0,   1/f]]

  where f = exp(q*d) is a symbol substituted only at the end.

  Parameters
  ----------
  d_sym : sympy symbol, optional. Defaults to locally defined `d`.
  d_val : float, optional. Numeric value for distance d.
  q_val : float, optional. Numeric value for in-plane wavevector q.

  Returns
  -------
  sympy.Matrix : Symbolic or numeric 2x2 propagation matrix.
  """
  # Define symbolic variables locally
  q = sp.Symbol('q', real=True)
  _d = d_sym if d_sym is not None else sp.Symbol('d', real=True)

  # Normalized symbolic f to avoid exp(q*d) blow-up
  label = str(_d)
  f_sym = sp.Symbol(f'f_{label}')

  # Build symbolic matrix
  P = sp.Matrix([
    [f_sym, 0    ],
    [0,    1/f_sym],
  ])

  # Build substitution dictionary
  # If numeric, substitute f = exp(q*d) directly
  subs_dict = {}
  if q_val is not None:
    subs_dict[q] = q_val
  if d_val is not None:
    subs_dict[_d] = d_val

  if not subs_dict:
    return P

  # Replace f_sym with exp(q*d) only when going numeric
  f_val = sp.exp(subs_dict.get(q, q) * subs_dict.get(_d, _d))
  P = P.subs(f_sym, f_val)

  if any(isinstance(v, (float, int)) for v in subs_dict.values()):
    P = P.evalf()

  return P


def SMatrix(
  e_val=None,
  epsilon_0_val=None,
  epsilon_t_val=None,
  q_val=None,
):
  """
  Source matrix for a point charge at a layer.

  S = e / (2 * epsilon_0 * epsilon_t * q) * [[-1],
                                              [ 1]]

  Parameters
  ----------
  e_val         : float, optional. Numeric value for electron charge.
  epsilon_0_val : float, optional. Numeric value for vacuum permittivity.
  epsilon_t_val : float, optional. Numeric value for epsilon above.
  q_val         : float, optional. Numeric value for in-plane wavevector.

  Returns
  -------
  sympy.Matrix : Symbolic or numeric 2x1 source matrix.
  """
  # Define symbolic variables locally
  q, e, epsilon_0, epsilon_t = sp.symbols('q e epsilon_0 epsilon_t', real=True)

  # Build symbolic matrix
  S = (e / (2 * epsilon_0 * epsilon_t * q)) * sp.Matrix([
    [-1],
    [ 1],
  ])

  # Build substitution dictionary
  subs_dict = {
    k: v for k, v in {
      q         : q_val,
      e         : e_val,
      epsilon_0 : epsilon_0_val,
      epsilon_t : epsilon_t_val,
    }.items()
    if v is not None
  }

  if not subs_dict:
    return S

  S = S.subs(subs_dict)

  if any(isinstance(v, float) for v in subs_dict.values()):
    S = S.evalf()

  return S


def verify_boundary_condition(M_total, phi_0, S_tilde):
  """
  Verify that the boundary condition (A, 0)^T is satisfied.
  Exits program if check fails.
  """
  result = M_total * sp.Matrix([[0], [phi_0]]) + S_tilde
  A_val = sp.simplify(result[0])
  zero_val = sp.simplify(result[1])
  
  if zero_val != 0:
    print(f"ERROR: Boundary condition NOT satisfied for source at layer {1}!")
    print(f"Expected: (A, 0)^T")
    print(f"Got: ({A_val}, {zero_val})^T")
    sys.exit(1)
  
  return None 

def substitute_f_with_exp(phi_dict, d_syms, f_syms, q):
  """
  Substitute f_sym -> exp(-q * d_sym) for all entries in phi_dict.
  Returns a new dict, leaving the original phi_dict unchanged.
  """
  subs = {f: sp.exp(-q * d) for f, d in zip(f_syms, d_syms)}
  return {key: expr.subs(subs) for key, expr in phi_dict.items()}


def _absorb_all_spurious_exponentials(phi_dict_sub, q, d_syms):
    """
    Scan all entries in phi_dict_sub for spurious exp(+) in the numerator
    or exp(-) in the denominator, and absorb them using algebraic cancellation.

    Returns (phi_dict_sub, success) where success=False if any entry could
    not be cleaned — caller should abort the job in that case.
    """

    def _get_exp_args_from_mul(e):
      """Return exp() arguments found as top-level Mul factors."""
      if isinstance(e, sp.Mul):
        return [arg.args[0] for arg in e.args if isinstance(arg, sp.exp)]
      elif isinstance(e, sp.exp):
        return [e.args[0]]
      return []

    def _has_spurious_exponentials(expr, q, d_syms):
      """
      Returns True if expr has exp(+...) as overall numerator factor
      or exp(-...) as overall denominator factor.
      """
      num, denom = sp.fraction(sp.powsimp(expr, combine='exp', force=True))
      sign_subs = {q: sp.Float(1), **{d: sp.Float(1) for d in d_syms}}
      for exp_arg in _get_exp_args_from_mul(num):
        try:
          if float(exp_arg.subs(sign_subs).evalf()) > 1e-10:
            return True
        except:
          pass
      for exp_arg in _get_exp_args_from_mul(denom):
        try:
          if float(exp_arg.subs(sign_subs).evalf()) < -1e-10:
            return True
        except:
          pass
      return False

    def _build_fwd_subs(expr, q, d_syms, f_syms):
      """
      Scan expr for all unique exp() arguments, expand each one,
      extract coefficients of every d_i*q, and build the substitution
      exp(arg) -> product of f_i^(-c_i/2).
      Handles single terms, cross terms like exp(-(d_12+d_23)*q),
      and any integer or rational power n.
      """
      exp_args = set()
      for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.exp):
          exp_args.add(a.args[0])

      fwd_subs = []
      seen = set()
      for arg in exp_args:
        arg_expanded = sp.expand(arg)
        replacement = sp.Integer(1)
        valid = True
        for d, f in zip(d_syms, f_syms):
          c = arg_expanded.coeff(d * q)
          if c != 0:
            replacement *= f**(-c / 2)

        # Check nothing symbolic remains after extracting all d_i*q terms
        remainder = arg_expanded
        for d in d_syms:
          c = arg_expanded.coeff(d * q)
          remainder = remainder - c * d * q
        remainder = sp.simplify(remainder)

        if remainder != 0:
          print_f(f"WARNING: _build_fwd_subs could not decompose exp({arg}), skipping.")
          valid = False

        if valid and sp.exp(arg) not in seen:
          fwd_subs.append((sp.exp(arg),  replacement))
          fwd_subs.append((sp.exp(-arg), 1/replacement))
          seen.add(sp.exp(arg))

      return fwd_subs

    def _absorb_spurious_exponentials(expr, q, d_syms):
      """
      Only acts if spurious exp() factors are detected.
      Scans for all exp(n*d*q) powers present, builds algebraic substitution
      dynamically, uses cancel() to absorb, then verifies numerically.
      Returns (result, success) where success=False means the issue
      persists after absorption and the job should be aborted.
      """
      if not _has_spurious_exponentials(expr, q, d_syms):
        return expr, True

      print_f("Found overall exp(+) in numerator or exp(-) in denominator.")
      print_f("Attempting to absorb to prevent blow-up at large q or d...")

      f_syms_local = [sp.symbols(f'_f{i}', positive=True) for i in range(len(d_syms))]
      fwd_subs  = _build_fwd_subs(expr, q, d_syms, f_syms_local)
      back_subs = [(f, sp.exp(-2*d*q)) for d, f in zip(d_syms, f_syms_local)]

      result = sp.powsimp(
        sp.cancel(expr.subs(fwd_subs)).subs(back_subs),
        combine='exp', force=True
      )

      # Numerical verification — build subs from ALL free symbols in expr
      verify_subs = {}
      for s in expr.free_symbols:
        name = str(s)
        if s == q:
          verify_subs[s] = sp.Float(0.5)
        elif any(str(s) == str(d) for d in d_syms):
          verify_subs[s] = sp.Float(1.0)
        elif name.startswith('epsilon'):
          verify_subs[s] = sp.Float(2.0)
        elif name == 'C':
          verify_subs[s] = sp.Float(1.0)
        elif name.startswith('chi'):
          verify_subs[s] = sp.Float(10.0)
        else:
          verify_subs[s] = sp.Float(1.0)

      try:
        orig_val   = complex(expr.subs(verify_subs))
        result_val = complex(result.subs(verify_subs))
        if abs(orig_val - result_val) > 1e-8:
          print_f("WARNING: Numerical mismatch after absorption — returning original.")
          return expr, False
      except Exception as e:
        print_f(f"WARNING: Numerical verification failed ({e}) — returning original.")
        return expr, False

      if _has_spurious_exponentials(result, q, d_syms):
        print_f("WARNING: Spurious exponentials persist after absorption attempt.")
        return result, False

      print_f("SUCCESS: Spurious exponentials absorbed and verified.")
      return result, True

    # Apply to all entries, track overall success
    success = True
    for key in phi_dict_sub:
      phi_dict_sub[key], key_success = _absorb_spurious_exponentials(
        phi_dict_sub[key], q, d_syms
      )
      if not key_success:
        print_f(f"ERROR: Could not fix spurious exponentials in {key}.")
        success = False

    return phi_dict_sub, success


def multilayer_potential(n=2):
  """
  Symbolic expression for multilayer potential.
  n: number of layers
  """
  # Generic symbols whose parameters will be passed through
  # pymex_yaml input file.
  q, epsilon_b, epsilon_t, e, epsilon_0 = sp.symbols(
      'q epsilon_b epsilon_t e epsilon_0', real=True
  )

  # Depending on number of layers build 
  # polarizabilities and interlayer distances
  # Starts from chi_1, chi_2 for readability (instead of chi_0, chi_1, ...)
  chi_syms = tuple(sp.symbols(f'chi_{i+1}') for i in range(n))
  if n > 1:
    d_names = [f'd_{i+1}{i+2}' for i in range(n-1)]
    d_syms = tuple(sp.symbols(name) for name in d_names)
    f_syms = tuple(sp.Symbol(f'f_{d_names[i]}') for i in range(n-1))
  else:
    d_syms = ()
    f_syms = ()
  
  # List of 2D Layer matrices (i.e., LMatrices)
  # L_0 --> \chi_1 and always with bottom substrate 
  # L_{n-1} --> \chi_n and always with top substrate
  # This is a specific choice common in 2D materials
  L_list = [LMatrix(chi_sym=chi_syms[i],
                    epsilon_b_val=1 if i>0 else None,
                    epsilon_t_val=1 if i<n-1 else None)
            for i in range(n)]
  
  # List of Propagation matrices (i.e., PMatrices)
  P_list = [PMatrix(d_sym=d_syms[i]) for i in range(n-1)]

  # M-matrix (Common structure regardless of source position)
  # M_total = L_n * P_{n-1} * L_{n-1} * ... * P_1 * L_1
  M_total = L_list[-1] 
  for i in reversed(range(len(P_list))):
    M_total = M_total * P_list[i] * L_list[i]
  M_total = sp.simplify(M_total)  
  M11, M12, M21, M22 = M_total[0,0], M_total[0,1], M_total[1,0], M_total[1,1] 

  phi_dict = {}
  if n >= 1:
    for i in range(n):

      # If not topmost layer, epsilon_t = 1 
      S = sp.simplify(SMatrix() if i == n-1 else SMatrix(epsilon_t_val=1))

      # Build Source-related matrix
      # based on the location of the source layer i.
      N = sp.eye(2)
      for k in range(n-1, i, -1):
        N = N * L_list[k] * P_list[k-1]

      S_tilde = sp.simplify(N * S)
      S_tilde1, S_tilde2 = S_tilde[0], S_tilde[1]

      # Solve for phi_11 from lower equation: 0 = M22*phi_0 + S_tilde2
      # This is computed once for each source layer i. 
      phi_0 = -S_tilde2/ M22
      # phi_dict[f"phi_{i+1}1"] = phi_0
      phi_0 = sp.simplify(phi_0)

      # Verify boundary condition when source is at layer 1
      verify_boundary_condition(M_total, phi_0, S_tilde)

      # Propagate upward to get potential at each target layer j
      psi_bottom = sp.Matrix([0, phi_0])

      # Build the potential, phi_ij
      # i: source layer index, j: target layer index
      for j in range(n):
        psi_j = psi_bottom
        for layer in range(j+1):
          psi_j = L_list[layer] * (P_list[layer-1] * psi_j 
                                   if layer > 0 else psi_j)
          if layer == i:
            psi_j = psi_j + S

        phi_ij = sp.simplify(psi_j[0] + psi_j[1])
        phi_dict[f"phi_{i+1}{j+1}"] = phi_ij

  # Manipulations to clean up the expressions
  prefactor = sp.symbols('C', real=True)
  for key in phi_dict:
    phi_dict[key] = phi_dict[key].subs(e/(epsilon_0), prefactor)

  # Substitute f_sym -> exp(-q * d_sym) 
  phi_dict_sub = substitute_f_with_exp(phi_dict, d_syms, f_syms, q)
  # Absorb any remaining exp(+) in numerator or exp(-) in denominator
  phi_dict_sub, exp_absorption_success = _absorb_all_spurious_exponentials(
    phi_dict_sub, q, d_syms)
  
  for key, expr in phi_dict_sub.items():
    phi_dict_sub[key] = sp.powsimp(
      sp.powdenest(expr.rewrite(sp.exp), force=True),
      combine='exp', force=True
    )
  return phi_dict_sub, exp_absorption_success

# n=3
# phi_dict, success = multilayer_potential(n=n)
# if success:
#   for key, expr in phi_dict.items():
#     print_f(f"Potential expression for {key}:")
#     print_f(expr)
# else:
#   print_f(f"{n}-layer fails")

