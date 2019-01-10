import torch

N, D_in, D_out, H = 64, 1000, 10, 100

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for t in range(500):
    y_predicted = model(x)
    loss = loss_fn(y, y_predicted)

    print(t, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


