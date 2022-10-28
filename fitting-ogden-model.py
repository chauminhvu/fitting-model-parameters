import time
import jax
import jax.numpy as jnp
import jaxopt as opt
import pandas as pd
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)


def ogden3(params, x_data):
    """
    The ogden3 function computes the Ogden strain-energy values
    for a given set of material parameters.
    :param x_data [3,]: Stretches vectors
    :param params [9,]: parameters of the ogden model
    :return: The strain-energy of ogden model
    """
    params = params.reshape(3, 3)
    # Get the the parameters
    μ, α, β = params[0, :], params[1, :], params[2, :]
    # Principal stretches
    λ1, λ2, λ3 = x_data[0], x_data[1], x_data[2]
    # The volume ratio
    J = λ1 * λ2 * λ3
    ψ = 0
    for i in range(len(μ)):
        # Ogden-Hill' strain energy function (ABAQUS version)
        ψ += (2 * μ[i] / α[i]**2) * (
            λ1**α[i] + λ2**α[i] + λ3**α[i] - 3.0
            + (J**(-α[i] * β[i]) - 1) / β[i]
        )
    return ψ


@jax.jit
def stress_batched(params, x_batched):
    """
    The stress_batched function computes the stress tensor for a batch of
    principal stretches. The inputs, outputs have shape of [m, 3]

    :param x_batched [m, 3]: Batch of principal stretches
    :return: Batch of principal stress
    """
    return jax.vmap(jax.grad(ogden3, argnums=1),
                    in_axes=(None, 0))(params, x_batched)


@jax.jit
def loss_func(params, x_data, y_data):
    """
    The loss_func function computes the loss of the model.
    It takes in parameters, x_data and y_data as input.
    The function returns a scalar value which is the sum of
    squared errors between predicted and actual values.

    :param params [3, 3]: parameters of the ogden model
    :param x_data [cases]: list of experimental data for principal stretches
    :param y_data [cases]: list of experimental data for princpal stresses
    :return: The sum of the squared differences between the predicted
     and actual values of the first component of 1st piola stresses.
    """
    params = params.reshape(3, 3)
    μ, β = params[0, :], params[2, :]

    # Ground state bulk modulus
    K0 = 0
    for i in range(len(μ)):
        K0 += jnp.array(2*μ[i] * (1/3 + β[i]))
    # predicted stresses
    puc_pred = stress_batched(params, x_data[0])
    pbc_pred = stress_batched(params, x_data[1])
    # losses
    loss1 = jnp.mean((puc_pred - y_data[0])**2)
    loss2 = jnp.mean((pbc_pred - y_data[1])**2)
    # parameters' constraints
    const1 = jax.nn.relu(jnp.sum(μ))
    const2 = jax.nn.relu(K0)
    # Use parameters' constraints as regularization terms
    losses = loss1 + loss2 + 1e-5 * (const1 + const2)
    return losses


def r_squared(y, y_hat):
    """
    The r_squared function computes the proportion of variance in the dependent
    variable that is explained by the independent variable. The r_squared
    function takes two arguments: y, which is a vector of dependent variables,
    and y_hat, which is a vector of predicted values for those dependent
    variables. It returns the proportion (between 0 and 1) of variance in y
    that can be explained by y_hat.

    :param y: Calculate the mean of y
    :param y_hat: Calculate the mean of y
    :return: The percentage of the variance in y that is explained by x
    """
    y_bar = y.mean()
    ss_tot = ((y - y_bar)**2).sum()
    ss_res = ((y - y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)


def plot_data(x, y_exp, y_pred, case, title):
    """
    The plot_data function plots the experimental data and the fitted models
    for a given case. The function takes in four arguments: x, y_exp, y_pred, and
    case. The x argument is an array of arrays containing stretch values for each
    experimental case. The y_exp argument is an array of arrays containing
    principal stress values for each experimental case.
    Finally, the title argument contains a string that will be used as plot title.
    
    :param x: list of experimental principal stretches for many cases
    :param y_exp: List of experimental principal stress
    :param y_pred: List of predicted principal stress
    :param case: Distinguish between the two cases in the plot
    :param title: Set the title of the plot
    :return: None
    """
    plt.title(f'{title}', fontsize=12)
    plt.xlabel(r'Stretch in loading dir., $\lambda$ [-]')
    plt.ylabel(r'Principal stress, $p_1$ [MPa]')

    # plot experimental data
    for i in range(len(y_exp)):
        plt.scatter(x[i], y_exp[i], edgecolors=f'k', facecolors='none',
                    label=r"$p_1^{{{0}}}$ (Exp.)".format(case[i]))

    #  plot fitted models
    for j in range(len(y_pred)):
        k = j % 2  # idx of x, either 0 or 1
        source = "paper" if j < 2 else "Opt"
        plt.plot(x[k], y_pred[j], color=f"C{j}",
                 label=r"$p_1^{{{0}}}$ ({1})".format(case[k], source))

    # set up
    plt.grid(0.25)
    plt.legend(loc='lower right')
    type = "paper" if len(y_pred) < 3 else "opt"
    plt.savefig(f'ogden3_{type}.png', dpi=300)
    return None


# Experimental data (Table 1, 152) [A. Kossa, 2016]
# doi.org/10.1016/j.polymertesting.2016.05.014
# Uniaxial compression
expUA = pd.read_csv('UAexp_polyethylene_foam.csv', index_col=False)
# Biaxial compression
expBA = pd.read_csv('BAexp_polyethylene_foam.csv', index_col=False)

# stretches in the loading direction
λuc = jnp.array(expUA['λ_UAexp'].values)
λbc = jnp.array(expBA['λ_BAexp'].values)

# Transverse stretches
λTuc = jnp.array(expUA['λT_UAexp'].values)
λTbc = jnp.array(expBA['λT_BAexp'].values)

# Principal 1st Piola Kirchhof stress (engineering stress)
puc = jnp.array(expUA['p_UAexp_MPa'].values)  # MPa
pbc = jnp.array(expBA['p_BAexp_MPa'].values)  # MPa

# Transverse stresses are zero
pTuc = jnp.zeros_like(puc)
pTbc = jnp.zeros_like(pbc)

# Collect experimental data
# Uniaxial compression case
λuc_vec = jnp.column_stack((λuc, λTuc, λTuc))
puc_vec = jnp.column_stack((puc, pTuc, pTuc))

# Biaxial compression case
λbc_vec = jnp.column_stack((λbc, λbc, λTbc))
pbc_vec = jnp.column_stack((pbc, pbc, pTbc))

# Reivew [Berezvai, 2016]'s results
params_Berezvai = jnp.array([[0.00064, 0.084, 0.0085],
                             [2.63, 4.78, 0.0058],
                             [1.214, 0.0072, 0.6]])
puc_paper = stress_batched(params_Berezvai.flatten(), λuc_vec)
pbc_paper = stress_batched(params_Berezvai.flatten(), λbc_vec)
r2uc = r_squared(puc, puc_paper[:, 0])
r2bc = r_squared(pbc, pbc_paper[:, 0])
print('R2 accuracy of UC (paper)', r2uc)
print('R2 accuracy of BC (paper)', r2bc)

# Since the paramester from [Berezvai, 2016] is not considered
# simutaniously 2 dataset. We will optimize new ones.
#
# Optimized parameters for 2 dataset simutaniously
x_data = [λuc_vec, λbc_vec]
y_data = [puc_vec, pbc_vec]

# Initialize params
key = jax.random.PRNGKey(2022)
params0 = jax.random.uniform(key, (9,))

# Use the ones from [Berezvai, 2016] as initial guess
# params0 = params_Berezvai

start = time.time()
optimizer = opt.ScipyMinimize(method='BFGS', fun=loss_func,
                              tol=10e-12, maxiter=1000)
opt_para = optimizer.run(params0, x_data=x_data, y_data=y_data).params

print(f"\noptimization time: {time.time() - start}")
print(f"Optimized para: \n {opt_para.reshape(3,3)}")

# Check conditions of the paramters
params = opt_para.reshape(3, 3)
μ, α, β = params[0, :], params[1, :], params[2, :]
K0 = sum([2*μ[i] * (1/3 + β[i]) for i in range(len(μ))])
print('\nCheck conditions of the paramters:')
print('Initial shear muodulus μ0 = sum μ > 0; μ0 = ', μ.sum())
print('Initial bulk muodulus K0 = sum 2μ_i(1/3 + β_i) > 0; K0 = ', K0)
print('Initial shear muodulus β > -0.33; β = ', β >= -1/3)

assert μ.sum() >= 0, "Initial shear modulus μ0 must > 0"
assert β.all() >= -1/3, "βi must > -1/3"
assert K0 >= 0, "Initial bulk modulus K0 must > 0"

# principal stresses based on the obtained params
opt_puc_vec = stress_batched(opt_para, x_data[0])
opt_pbc_vec = stress_batched(opt_para, x_data[1])

# Performance of the obtained params
opt_r2uc = r_squared(puc, opt_puc_vec[:, 0])
opt_r2bc = r_squared(pbc, opt_pbc_vec[:, 0])
print('R2 accuracy of UC (opt)', opt_r2uc)
print('R2 accuracy of BC (opt)', opt_r2bc)

# Plot comparisions
plot_data([λuc, λbc], [puc, pbc],
          [puc_paper[:, 0], pbc_paper[:, 0], opt_puc_vec[:, 0],
              opt_pbc_vec[:, 0]], ['UC', 'BC'],
          'Fitted optimal params (Ogden) on closed-cell polyethylene foam material')
