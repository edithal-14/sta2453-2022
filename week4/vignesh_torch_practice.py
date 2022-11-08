import torch
import matplotlib.pyplot as plt
import seaborn as sns

def loss_function(output, target):
    return (output - target) **2

def model(input):
    # Objective function
    x = input
    return 5 * x**2 + 3 * x + 7

def main():
    # Plot objective function
    xs = torch.linspace(-3, 3, 10000)
    ys = model(xs)
    sns.lineplot(x=xs, y=ys).set_title("Objective function")
    plt.savefig("vignesh_torch_practice.png")
    plt.clf()

    x_star = 4 + torch.randn(1) * 2
    # Set gradient tracking
    x_star.requires_grad = True
    num_iter = 1000
    lr = 0.3
    optim = torch.optim.Adam([x_star], lr=lr)
    for _ in range(num_iter):
        optim.zero_grad()
        output = model(x_star)
        loss = loss_function(output, 20)
        loss.backward()
        optim.step()
    print(x_star)

if __name__ == "__main__":
    main()