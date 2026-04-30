import torch

def quantile_loss(pred, target, quantiles=[0.05, 0.5, 0.95]):
    loss = 0.0
    for i, q in enumerate(quantiles):
        errors = target - pred[:, :, i]
        loss += torch.max(q * errors, (q - 1) * errors).mean()
    return loss / len(quantiles)

def train_model(model, train_loader, val_loader, logger, num_epochs=10, patience=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = quantile_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                val_loss += quantile_loss(outputs, batch_y).item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        logger.info(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_val_loss, None