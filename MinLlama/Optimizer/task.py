from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """
    AdamW optimizer with bias correction and weight decay decoupled from learning rate.
    Implements the algorithm from:
    "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015)
    "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
    Args:
        params: Iterable of model parameters
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square
              (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-6)
        weight_decay: Weight decay coefficient (L2 penalty) (default: 0.0)
        correct_bias: Whether to apply bias correction (default: True)
    Otimizador AdamW com correção de viés e regularização de peso desacoplada da taxa de aprendizado.
    Implementa o algoritmo de:
    "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015)
    "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
    Argumentos:
        params: Iterável de parâmetros do modelo
        lr: Taxa de aprendizado (padrão: 1e-3)
        betas: Coeficientes para calcular as médias móveis do gradiente e seu quadrado
        (padrão: (0.9, 0.999))
        eps: Termo adicionado ao denominador para estabilidade numérica (padrão: 1e-6)
        weight_decay: Coeficiente de regularização de peso (penalidade L2) (padrão: 0.0)
        correct_bias: Indica se a correção de viés deve ser aplicada (padrão: True)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Args:
            closure: A callable that reevaluates the model and returns the loss.
        Returns:
            loss: The loss value if closure is provided, None otherwise.
        Executa uma única etapa de otimização.
        Argumentos:
            closure: Uma função que reavalia o modelo e retorna a perda.
        Retorna:
            loss: O valor da perda se closure for fornecido; caso contrário, None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # ============================================================
                # TODO: implement state creation and parameters initialization
                # ============================================================
                # Access hyperparameters from the `group` dictionary
                # =============================================================
                # TODO: implementar a criação de estado e a inicialização de parâmetros
                # ============================================================
                # Acessar hiperparâmetros do dicionário `group`
                beta1, beta2 = group["betas"]

                if len(state) == 0:
                    # Initialize state
                    # step: number of iterations (for bias correction)
                    # momentum_t: first moment estimate (mean of gradients)
                    # rms_t: second moment estimate (uncentered variance of gradients)
                    # Inicializar estado
                    # passo: número de iterações (para correção de viés)
                    # momentum_t: estimativa do primeiro momento (média dos gradientes)
                    # rms_t: estimativa do segundo momento (variância não centrada dos gradientes)
                    state["step"] = 0
                    state["momentum_t"] = torch.zeros_like(p.data)  # m_t
                    state["rms_t"] = torch.zeros_like(p.data)  # v_t

                # ============================================================
                # TODO: update first and second moments of the gradients
                # ============================================================
                # Increment step counter
                # =============================================================
                # TODO: atualizar primeiro e segundo momentos dos gradientes
                # ============================================================
                # Incrementar contador de passos
                state["step"] += 1
                step = state["step"]

                # Get current moments
                momentum_t = state["momentum_t"]  # m_t
                rms_t = state["rms_t"]  # v_t

                # Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                # Atualizar estimativa do primeiro momento enviesado: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                momentum_t.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                # Atualizar estimativa enviesada do segundo momento bruto: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                rms_t.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # ============================================================
                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                # ============================================================
                # Compute bias-corrected estimates
                # ============================================================
                # TODO: Correção de viés
                # Observe que estamos usando a "versão eficiente" fornecida em
                # https://arxiv.org/abs/1412.6980
                # =============================================================
                # Calcular estimativas com correção de viés
                if group["correct_bias"]:
                    # Bias correction for m_t: m_hat = m_t / (1 - beta1^t)
                    # Bias correction for v_t: v_hat = v_t / (1 - beta2^t)
                    # Correção de viés para m_t: m_hat = m_t / (1 - beta1^t)
                    # Correção de viés para v_t: v_hat = v_t / (1 - beta2^t)
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    momentum_hat = momentum_t / bias_correction1
                    rms_hat = rms_t / bias_correction2
                else:
                    # Without bias correction (less common, but allowed)
                    # Sem correção de viés (menos comum, mas permitido)
                    momentum_hat = momentum_t
                    rms_hat = rms_t

                # ============================================================
                # Update parameters
                # ============================================================
                # Efficient version of the update:
                # theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
                # =============================================================
                # Parâmetros de atualização
                # =============================================================
                # Versão eficiente da atualização:

                # theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
                # theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
                alpha = group["lr"]
                eps = group["eps"]

                # Compute parameter update
                param_update = momentum_hat / (torch.sqrt(rms_hat) + eps)

                # Apply the update  # Calcular atualização de parâmetros
                params = p.data - alpha * param_update

                # ============================================================
                # TODO: Add weight decay after the main gradient-based updates
                # Please note that the learning rate should be incorporated
                # into this update
                # ============================================================
                # Weight decay is applied DECOUPLED from the gradient update
                # This is the key difference from L2 regularization
                # Formula: theta = theta - alpha * weight_decay * theta
                # ============================================================
                # TODO: Adicionar regularização de peso após as atualizações principais baseadas em gradiente
                # Observe que a taxa de aprendizado deve ser incorporada
                # a esta atualização
                # =============================================================
                # A regularização de peso é aplicada DESACOPLADA da atualização do gradiente
                # Esta é a principal diferença em relação à regularização L2
                # Fórmula: theta = theta - alpha * weight_decay * theta
                if group["weight_decay"] > 0:
                    params = params - alpha * group["weight_decay"] * p.data

                # Update parameter tensor in-place # Atualizar tensor de parâmetros no local
                p.data.copy_(params)

        return loss




# from typing import Callable, Iterable, Tuple
#
# import torch
# from torch.optim import Optimizer
#
#
# class AdamW(Optimizer):
#     def __init__(
#             self,
#             params: Iterable[torch.nn.parameter.Parameter],
#             lr: float = 1e-3,
#             betas: Tuple[float, float] = (0.9, 0.999),
#             eps: float = 1e-6,
#             weight_decay: float = 0.0,
#             correct_bias: bool = True,
#     ):
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
#         super().__init__(params, defaults)
#
#     def step(self, closure: Callable = None):
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
#
#                 # State should be stored in this dictionary
#                 state = self.state[p]
#
#                 # TODO: implement state creation and parameters initialization: step, momentum_t, rms_t, beta_1, beta_2
#
#                 # Access hyperparameters from the `group` dictionary
#                 alpha = group["lr"]
#
#                 # TODO: update first and second moments of the gradients
#
#                 # TODO: Bias correction
# # Please note that we are using the "efficient version" given in
# # https://arxiv.org/abs/1412.6980
#
#                 # Update parameters
#                 params = p - alpha_t * momentum_t / (torch.sqrt(rms_t) + group['eps'])
#
#                 # TODO: Add weight decay after the main gradient-based updates
# # Please note that the learning rate should be incorporated into this update
#
#         return loss
