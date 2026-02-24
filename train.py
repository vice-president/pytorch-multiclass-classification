import torch
from torch import nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=2500, n_features=30, n_informative=18, n_classes=4, n_clusters_per_class=1, random_state=7
)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

model = nn.Sequential(nn.Linear(30, 128), nn.ReLU(), nn.Linear(128, 4))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 41):
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = loss_fn(val_logits, y_val).item()
        preds = torch.argmax(val_logits, dim=1)
        acc = (preds == y_val).float().mean().item()
    if epoch % 5 == 0:
        print(f"epoch={epoch:02d} train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")
