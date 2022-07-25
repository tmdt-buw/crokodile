import torch
from utils.nn import NeuralNetwork, Rescale, Sawtooth
import numpy as np

class DHT_Transform(torch.nn.Module):
    def __init__(self, theta=None, d=None, a=0, alpha=0, proximal=False, upscale_dim=False):
        super(DHT_Transform, self).__init__()

        assert theta is not None or d is not None, "Transformation can only have one degree of freedom"

        self.proximal = proximal

        if theta is not None:
            self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float32), requires_grad=False)
        if d is not None:
            self.d = torch.nn.Parameter(torch.tensor(d, dtype=torch.float32), requires_grad=False)
        self.a = torch.nn.Parameter(torch.tensor(a, dtype=torch.float32), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)

        self.theta_cos = torch.nn.Parameter(torch.tensor([[1, 0, 0, 0],
                                                          [0, 1, 0, 0],
                                                          [0, 0, 0, 0],
                                                          [0, 0, 0, 0]], dtype=torch.float32), requires_grad=False)
        self.theta_sin = torch.nn.Parameter(torch.tensor([[0, -1, 0, 0],
                                                          [1, 0, 0, 0],
                                                          [0, 0, 0, 0],
                                                          [0, 0, 0, 0]], dtype=torch.float32), requires_grad=False)
        self.theta_const = torch.nn.Parameter(torch.tensor([[0, 0, 0, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 1, 0],
                                                            [0, 0, 0, 1]], dtype=torch.float32), requires_grad=False)

        self.d_ = torch.nn.Parameter(torch.tensor([[0, 0, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 1],
                                                   [0, 0, 0, 0]], dtype=torch.float32), requires_grad=False)
        self.d_const = torch.nn.Parameter(torch.eye(4, 4), requires_grad=False)

        self.a_ = torch.nn.Parameter(torch.tensor([[0, 0, 0, 1],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [0, 0, 0, 0]], dtype=torch.float32), requires_grad=False)
        self.a_const = torch.nn.Parameter(torch.eye(4, 4), requires_grad=False)

        self.alpha_cos = torch.nn.Parameter(torch.tensor([[0, 0, 0, 0],
                                                          [0, 1, 0, 0],
                                                          [0, 0, 1, 0],
                                                          [0, 0, 0, 0]], dtype=torch.float32), requires_grad=False)
        self.alpha_sin = torch.nn.Parameter(torch.tensor([[0, 0, 0, 0],
                                                          [0, 0, -1, 0],
                                                          [0, 1, 0, 0],
                                                          [0, 0, 0, 0]], dtype=torch.float32), requires_grad=False)
        self.alpha_const = torch.nn.Parameter(torch.tensor([[1, 0, 0, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 1]], dtype=torch.float32), requires_grad=False)

        if upscale_dim:
            network_structure = [("linear", 10), ("tanh", None)] * 4 + [("linear", 1)]
            self.theta_upscale_dim = NeuralNetwork(1, network_structure)
            self.d_upscale_dim = NeuralNetwork(1, network_structure)
            self.a_upscale_dim = NeuralNetwork(1, network_structure)
            self.alpha_upscale_dim = NeuralNetwork(1, network_structure)

    def forward(self, x=None):

        if hasattr(self, "theta"):
            theta = self.theta.expand(len(x), 1)
        else:
            theta = x

        if hasattr(self, "d"):
            d = self.d.expand(len(x), 1)
        else:
            d = x

        alpha = self.alpha.expand(len(x), 1)
        a = self.a.expand(len(x), 1)

        # theta = torch.remainder(theta, torch.tensor(2 * np.pi))
        # alpha = torch.remainder(alpha, torch.tensor(2 * np.pi))

        if hasattr(self, "theta_upscale_dim"):
            theta = self.theta_upscale_dim(theta)
        if hasattr(self, "d_upscale_dim"):
            d = self.d_upscale_dim(d)
        if hasattr(self, "a_upscale_dim"):
            a = self.a_upscale_dim(a)
        if hasattr(self, "alpha_upscale_dim"):
            alpha = self.alpha_upscale_dim(alpha)

        transformation_theta = torch.einsum("xy,bz->bxy", self.theta_cos, theta.cos()) + \
                               torch.einsum("xy,bz->bxy", self.theta_sin, theta.sin()) + self.theta_const
        transformation_d = torch.einsum("xy,bz->bxy", self.d_, d) + self.d_const
        transformation_alpha = torch.einsum("xy,bz->bxy", self.alpha_cos, alpha.cos()) + \
                               torch.einsum("xy,bz->bxy", self.alpha_sin, alpha.sin()) + self.alpha_const
        transformation_a = torch.einsum("xy,bz->bxy", self.a_, a) + self.a_const

        if self.proximal:
            transformation = torch.einsum(f"bij,bjk,bkl,blm->bim",
                                          transformation_alpha, transformation_a,
                                          transformation_theta, transformation_d)
        else:
            transformation = torch.einsum(f"bij,bjk,bkl,blm->bim",
                                          transformation_theta, transformation_d,
                                          transformation_a, transformation_alpha)

        return transformation


class DHT_Model(torch.nn.Module):
    def __init__(self, dht_params=None, upscale_dim=False):
        super().__init__()

        self.transformations = torch.nn.ModuleList([])

        self.active_joints_mask = [int("theta" not in dht_param or "d" not in dht_param) for dht_param in dht_params]

        for dht_param in dht_params:
            self.transformations.append(DHT_Transform(**dht_param, upscale_dim=upscale_dim))

        self.scaling_mask = torch.nn.Parameter(torch.zeros(4,4, dtype=bool), requires_grad=False)
        self.scaling_mask[:3,-1] = True

        self.scaling = torch.nn.Parameter(torch.ones(len(dht_params), requires_grad=True))

        self.pose_init = torch.nn.Parameter(torch.eye(4, 4), requires_grad=False)

    def forward(self, params):

        pose = torch.eye(4, 4, device=params.device).unsqueeze(0).repeat(params.shape[0], 1, 1)

        poses = []

        for transformation, param in zip(self.transformations, params.split(self.active_joints_mask, 1)):
            transformation = transformation(param)

            pose = torch.einsum("bij,bjk->bik", pose, transformation)

            poses.append(pose)

        out = torch.stack(poses, 1)

        return out

def get_dht_model(dht_params, joint_limits, upscale_dim=False):
    return torch.nn.Sequential(
        Rescale(-1, 1, joint_limits[:,0], joint_limits[:,1]),
        # Sawtooth(-np.pi, np.pi, -np.pi, np.pi),
        DHT_Model(dht_params, upscale_dim=upscale_dim)
    )

if __name__ == '__main__':

    # theta = np.random.rand()
    # d = np.random.rand()
    # a = np.random.rand()
    # alpha = np.random.rand()
    # print(d, a, alpha)
    #
    # # oracle = DHT_Transform_Translation(theta, a, alpha)
    # oracle = DHT_Transform_Rotation(d, a, alpha)
    #
    # model = DHT_Transform()
    #
    # print([p for p in model.parameters()])
    #
    # loss_function = torch.nn.MSELoss()
    # loss_function_mode = torch.nn.BCELoss()
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    #
    # for ii in range(100_000):
    #     x = torch.rand(64, 1)
    #
    #     transformation_oracle = oracle(x)
    #     transformation_model = model(x)
    #
    #     optimizer.zero_grad()
    #     loss = loss_function(transformation_model, transformation_oracle)
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     if ii % 1000 == 0:
    #         print("Step", ii)
    #         print(loss.item())
    #
    #         print(theta, d, a, alpha)
    #         print(model.theta.data, model.d.data, model.a.data, model.alpha.data)

    from dht import DHT_Model as DHT_Model_static
    import numpy as np
    from matplotlib import pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = [0, 0, 0, 0, 0]
    dht_params = np.random.rand(len(modes), 4)

    dht_params[:, modes] = None
    # dht_params[:, 1] = 0
    # dht_params[:, 3] = 0
    print(dht_params)

    # dht_params = [
    #     [None, 0., .5, 0.],
    #     [None, 0., .5, 0.],
    # ]

    # theta = np.random.rand(len(modes))torch.tensor([[.5, .5], [.0, .0]]).to(device)

    oracle = DHT_Model_static(dht_params).to(device)

    # positions = output[:, :, :3, -1]

    model = DHT_Model(modes).to(device)

    # print([p for p in model.parameters()])

    loss_function = torch.nn.MSELoss()
    loss_function_mode = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

    for ii in range(100_000):
        x = torch.rand(64, len(modes)).to(device) * 2 - 1
        x *= np.pi

        transformation_oracle = oracle(x)
        transformation_model = model(x)

        tcp_oracle = transformation_oracle[:, -1, :3, :]
        tcp_model = transformation_model[:, -1, :3, :]

        optimizer.zero_grad()
        loss = loss_function(tcp_model, tcp_oracle)

        loss.backward()
        optimizer.step()


        def plot_line(ax, pointA, pointB, color=None):
            ax.plot([pointA[0], pointB[0]], [pointA[1], pointB[1]], zs=[pointA[2], pointB[2]], c=color)


        if ii % 1_000 == 0:
            print("Step", ii)
            print(loss.item())

            # print(theta, d, a, alpha)
            param_tensor = []

            for transformation in model.transformations:
                params = torch.stack((transformation.theta.data, transformation.d.data, transformation.a.data,
                                      transformation.alpha.data))
                param_tensor.append(params)

            param_tensor = torch.stack(param_tensor)

            print(param_tensor)

            fig = plt.figure(figsize=(15, 15))

            for irow in range(9):
                ax = fig.add_subplot(3, 3, irow + 1, projection='3d')
                ax_model = ax_oracle = ax
                # ax_model = fig.add_subplot(3, 2, irow * 2 + 1, projection='3d')
                # ax_oracle = fig.add_subplot(3, 2, irow * 2 + 2, projection='3d')

                # ax_model.set_axis_off()
                # ax_oracle.set_axis_off()

                kp_oracle = transformation_oracle[irow, :, :3, -1].cpu().detach().numpy()
                kp_oracle = np.concatenate([np.zeros((1, 3)), kp_oracle])

                kp_oracle_link_lengths = np.sqrt(np.power(kp_oracle[1:] - kp_oracle[:-1], 2).sum(-1))
                kp_oracle_total_length = kp_oracle_link_lengths.sum()

                prev_length = 0

                for pointA, pointB in zip(kp_oracle[:-1], kp_oracle[1:]):
                    color_start = prev_length / kp_oracle_total_length
                    link_length = np.sqrt(np.power(pointB - pointA, 2).sum(-1))
                    color_end = (prev_length + link_length) / kp_oracle_total_length

                    points = np.linspace(pointA, pointB, int(np.ceil(10 * link_length / kp_oracle_total_length)))
                    colors = np.linspace(color_start, color_end, len(points))[:-1]
                    colors = plt.get_cmap("autumn")(colors)

                    for pA, pB, color in zip(points[:-1], points[1:], colors):
                        ax.plot([pA[0], pB[0]], [pA[1], pB[1]], zs=[pA[2], pB[2]], c=color)

                    prev_length += link_length

                kp_model = transformation_model[irow, :, :3, -1].cpu().detach().numpy()
                kp_model = np.concatenate([np.zeros((1, 3)), kp_model])

                kp_model_link_lengths = np.sqrt(np.power(kp_model[1:] - kp_model[:-1], 2).sum(-1))
                kp_model_total_length = kp_model_link_lengths.sum()

                prev_length = 0

                for pointA, pointB in zip(kp_model[:-1], kp_model[1:]):
                    color_start = prev_length / kp_model_total_length
                    link_length = np.sqrt(np.power(pointB - pointA, 2).sum(-1))
                    color_end = (prev_length + link_length) / kp_model_total_length

                    points = np.linspace(pointA, pointB, int(np.ceil(10 * link_length / kp_model_total_length)))
                    colors = np.linspace(color_start, color_end, len(points))[:-1]
                    colors = plt.get_cmap("autumn")(colors)

                    for pA, pB, color in zip(points[:-1], points[1:], colors):
                        ax.plot([pA[0], pB[0]], [pA[1], pB[1]], zs=[pA[2], pB[2]], c=color)

                    prev_length += link_length

                # plot_line(ax_model, [0, 0, 0], kp_model[0], plt.get_cmap("autumn")(0.))
                # plot_line(ax_oracle, [0, 0, 0], kp_oracle[0], plt.get_cmap("winter")(0.))
                #
                # for ilink in range(len(kp_oracle) - 1):
                #     plot_line(ax_model, kp_model[ilink], kp_model[ilink + 1], plt.get_cmap("autumn")((ilink + 1) / len(kp_oracle)))
                #     plot_line(ax_oracle, kp_oracle[ilink], kp_oracle[ilink + 1], plt.get_cmap("winter")((ilink + 1) / len(kp_oracle)))

                # print("Link", ilink)
                # print(kp_model[ilink])
                # print(kp_oracle[ilink])

                # for ilink in range(len(kp_oracle)):
                #     plot_line(ax_oracle, kp_oracle[ilink], kp_oracle[ilink + 1])
                #
                #     ax_model.quiver(
                #         kp_model[ilink][0], kp_model[ilink][1], kp_model[ilink][2],  # <-- starting point of vector
                #         v[0] - mean_x, v[1] - mean_y, v[2] - mean_z,  # <-- directions of vector
                #         color='red', alpha=.8, lw=3,
                #     )

            plt.show()

            #     for link in
            #
            #
